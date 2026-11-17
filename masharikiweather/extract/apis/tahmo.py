"""
TAHMO DataHub Weather Station Extraction Engine.

Downloads high-resolution ground station observations from the Trans-African
Hydro-Meteorological Observatory (TAHMO) DataHub (https://datahub.tahmo.org).

Key Design Principles:
1. Non-Destructive Storage: Multi-sensor observations (e.g. rh_S001402, rh_S001403)
   are never averaged or collapsed; each sensor stream is preserved in dedicated columns.
2. Quality Control: Observations with quality flag != 1 are converted to NaN for controlled feeds.
3. Spatiotemporal Subsetting: Automated station discovery and filtering by bounding box.
4. Output Persistence: Stored as standardized Apache Parquet files on local disk.
"""

from __future__ import annotations

import concurrent.futures
import gc
import logging
import random
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util import Retry

from .base import (
    AFRICA_BBOX,
    EAST_AFRICA_BBOX,
    KENYA_BBOX,
    BaseExtractor,
    SpatialBoundingBox,
    TemporalRange,
)
from .credentials import CredentialsError, CredentialsManager, parse_date, parse_date_range

logger = logging.getLogger("MasharikiWeather.TAHMO")

API_BASE_URL = "https://datahub.tahmo.org"
MAX_REQUEST_DAYS = 365
DATASETS = ("controlled", "raw")
SENSOR_COL_PATTERN = re.compile(r"^(?P<variable>[a-z]{2,3})_(?P<sensor>\w+)$")

# Bidirectional Cross-Source Harmonization Registry for TAHMO
TAHMO_SHORT_TO_CANONICAL: Dict[str, str] = {
    "te": "2m_temperature",
    "temp": "2m_temperature",
    "temperature": "2m_temperature",
    "rh": "relative_humidity",
    "pr": "total_precipitation",
    "precip": "total_precipitation",
    "ws": "wind_speed",
    "wd": "wind_from_direction",
    "sr": "surface_solar_radiation_downwards",
    "rad": "surface_solar_radiation_downwards",
    "ap": "surface_pressure",
    "pres": "surface_pressure",
}

TAHMO_CANONICAL_TO_SHORT: Dict[str, str] = {
    "2m_temperature": "te",
    "relative_humidity": "rh",
    "total_precipitation": "pr",
    "wind_speed": "ws",
    "wind_from_direction": "wd",
    "surface_solar_radiation_downwards": "sr",
    "surface_pressure": "ap",
}

# Standard PeakWeather parameters catalog inventory for TAHMO
DEFAULT_TAHMO_PARAMETERS: List[Dict[str, Any]] = [
    {
        "parameter": "2m_temperature",
        "canonical_name": "2m_temperature",
        "standard_name": "air_temperature",
        "units": "degC",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Near-surface dry bulb air temperature at ~2m elevation.",
    },
    {
        "parameter": "relative_humidity",
        "canonical_name": "relative_humidity",
        "standard_name": "relative_humidity",
        "units": "%",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Relative humidity measured at ~2m.",
    },
    {
        "parameter": "total_precipitation",
        "canonical_name": "total_precipitation",
        "standard_name": "precipitation_amount",
        "units": "mm",
        "physical_type": "PRECIPITATION",
        "resampling_method": "sum",
        "description": "Cumulative liquid precipitation measured by tipping bucket or optical rain gauge.",
    },
    {
        "parameter": "surface_solar_radiation_downwards",
        "canonical_name": "surface_solar_radiation_downwards",
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "units": "W m-2",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Downwelling shortwave solar radiation flux.",
    },
    {
        "parameter": "surface_pressure",
        "canonical_name": "surface_pressure",
        "standard_name": "surface_air_pressure",
        "units": "hPa",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Atmospheric surface pressure.",
    },
    {
        "parameter": "wind_speed",
        "canonical_name": "wind_speed",
        "standard_name": "wind_speed",
        "units": "m s-1",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Horizontal scalar wind speed at ~2m mast.",
    },
    {
        "parameter": "wind_from_direction",
        "canonical_name": "wind_from_direction",
        "standard_name": "wind_from_direction",
        "units": "degrees",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Direction from which the wind is blowing (0-360 degrees).",
    },
]


def canonicalize_tahmo_variable(var: str) -> str:
    """Map raw TAHMO short code to canonical descriptive name."""
    return TAHMO_SHORT_TO_CANONICAL.get(var, var)


def tahmo_canonical_to_short(var: str) -> str:
    """Map canonical descriptive name to primary TAHMO short code."""
    return TAHMO_CANONICAL_TO_SHORT.get(var, var)


class TAHMOExtractor(BaseExtractor):
    """
    Client and extraction engine for TAHMO ground station observations.
    """

    extractor_name: str = "tahmo"
    network_name: str = "tahmo"

    def __init__(
        self,
        output_dir: Union[str, Path] = "./data/tahmo",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        env_file: Optional[Union[str, Path]] = None,
        dataset: str = "controlled",
        request_delay: float = 0.1,
        max_workers: Optional[int] = 4,
    ) -> None:
        """
        Initialize TAHMO ground station extraction engine with authentication and output cache.

        Parameters
        ----------
        output_dir : Union[str, Path], default "./data/tahmo"
            Directory path where extracted station observations and metadata will be saved.
        api_key : Optional[str]
            TAHMO DataHub API key.
        api_secret : Optional[str]
            TAHMO DataHub API secret.
        env_file : Optional[Union[str, Path]]
            Optional path to custom .env file for BYOK credential resolution.
        dataset : str, default "controlled"
            Dataset tier ("controlled" or "raw").
        request_delay : float, default 0.1
            Inter-request pacing delay in seconds between HTTP queries to protect legacy servers.
        max_workers : Optional[int], default 4
            Maximum number of concurrent worker threads for batch extraction. Kept at a maximum
            of 4 to protect legacy TAHMO server infrastructure while accelerating retrieval.
            Pass 1 or None for serial execution.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, received: {dataset}")

        self.credentials = CredentialsManager.get_tahmo_credentials(
            api_key=api_key, api_secret=api_secret, env_file=env_file
        )
        self.api_key = self.credentials["api_key"]
        self.api_secret = self.credentials["api_secret"]
        self.env_file = env_file
        self.auth = HTTPBasicAuth(self.credentials["api_key"], self.credentials["api_secret"])
        self.request_delay = max(0.0, float(request_delay))
        self.max_workers = max_workers
        self._local = threading.local()
        self.failed_stations: Dict[str, str] = {}

    def _get_thread_session(self) -> requests.Session:
        """Get or create thread-local requests.Session to guarantee thread safety across concurrent workers."""
        if not hasattr(self._local, "session"):
            session = requests.Session()
            session.auth = self.auth
            adapter = HTTPAdapter(
                pool_connections=8,
                pool_maxsize=8,
                max_retries=Retry(
                    total=3,
                    backoff_factor=1.0,
                    status_forcelist=[429, 500, 502, 503, 504],
                ),
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._local.session = session
        return self._local.session

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> Dict[str, Any]:
        """
        Execute authenticated HTTP GET request with retries, exponential backoff,
        and respectful inter-request delay to protect legacy TAHMO server infrastructure.
        """
        url = f"{API_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
        session = self._get_thread_session()
        delay = 1.0
        for attempt in range(1, retries + 1):
            try:
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                resp = session.get(url, params=params, timeout=60)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code in (401, 403):
                    raise CredentialsError("TAHMO DataHub", f"Authentication failed (HTTP {resp.status_code})")
                elif resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_time = float(retry_after) if retry_after else (delay * 2.0 + random.uniform(0.1, 0.5))
                    logger.warning("TAHMO rate limit (HTTP 429). Server-respecting backoff for %.1f s", sleep_time)
                    time.sleep(sleep_time)
                else:
                    logger.warning("TAHMO request %s failed with HTTP %d (attempt %d/%d)", url, resp.status_code, attempt, retries)
            except Exception as e:
                if attempt == retries:
                    raise
                logger.warning("Network retry on %s: %s", url, e)
            time.sleep(delay + random.uniform(0.1, 0.4))
            delay *= 2.0
        raise RuntimeError(f"Failed to fetch {url} after {retries} attempts.")

    def get_stations(
        self,
        bbox: Optional[SpatialBoundingBox] = None,
    ) -> pd.DataFrame:
        """
        Retrieve list of available weather stations, optionally filtered by spatial bounding box.

        Mathematical docstring:
        Output table shape: (N_stations, C_metadata)
        where columns include ['station_code', 'name', 'lat', 'lon', 'elevation', 'country', 'status'].
        """
        raw = self._request("services/assets/v2/stations", params={"sort": "code"})
        stations_list = raw.get("data", raw) if isinstance(raw, dict) else raw
        rows: List[Dict[str, Any]] = []
        for s in stations_list:
            if not isinstance(s, dict):
                continue
            loc = s.get("location") or {}
            lat = loc.get("latitude")
            lon = loc.get("longitude")
            elev = loc.get("elevation")

            if lat is None or lon is None:
                continue

            lat_f = float(lat)
            lon_f = float(lon)

            if bbox is not None and not bbox.contains(lat_f, lon_f):
                continue

            country_code = loc.get("countrycode") or s.get("countrycode") or s.get("country")
            rows.append({
                "station_code": s.get("code"),
                "name": s.get("name"),
                "lat": lat_f,
                "lon": lon_f,
                "elevation": float(elev) if elev is not None else np.nan,
                "country": str(country_code).upper() if country_code else None,
                "status": s.get("status"),
            })

        df = pd.DataFrame(rows)
        return df.sort_values(by="station_code").reset_index(drop=True)

    def _parse_measurements_fast(
        self,
        payload: Dict[str, Any],
        variables: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Vectorized parser converting raw TAHMO InfluxDB JSON into a non-destructive flat DataFrame.

        Mathematical docstring:
        Input:
            payload: Dict[str, Any] containing InfluxDB measurement series of shape (N_raw_obs, C_fields)
                where C_fields includes ['time', 'variable', 'sensor', 'value', 'quality'].
            variables: Optional sequence of C_req requested canonical or short parameter codes.
        Output:
            flat_df: pd.DataFrame of shape (N_valid_obs, 3) with columns ['time', 'col', 'value']
                where 'col' is formatted as '<variable>_<sensor>' or '<variable>'.
        """
        if not isinstance(payload, dict) or "results" not in payload:
            return pd.DataFrame(columns=["time", "col", "value"])

        frames: List[pd.DataFrame] = []
        for res in payload.get("results", []):
            if "error" in res:
                logger.warning("TAHMO InfluxDB reported series error: %s", res["error"])
                continue
            for s in res.get("series", []):
                cols = s.get("columns", [])
                vals = s.get("values", [])
                if not cols or not vals:
                    continue

                raw_df = pd.DataFrame(vals, columns=cols)
                if "time" not in raw_df.columns or "value" not in raw_df.columns or "variable" not in raw_df.columns:
                    continue

                if variables:
                    raw_df = raw_df[raw_df["variable"].isin(variables)]
                if raw_df.empty:
                    continue

                # Quality control filter: non-quality 1 values set to NaN
                if self.dataset == "controlled" and "quality" in raw_df.columns:
                    raw_df.loc[raw_df["quality"] != 1, "value"] = np.nan

                # Coerce values safely to numeric float32
                raw_df["value"] = pd.to_numeric(raw_df["value"], errors="coerce").astype(np.float32)

                # Defensive sensor column naming (never create 'pr_None' or 'pr_nan')
                if "sensor" in raw_df.columns:
                    has_sensor = (
                        raw_df["sensor"].notna()
                        & (raw_df["sensor"].astype(str).str.strip() != "")
                        & (raw_df["sensor"].astype(str) != "None")
                        & (raw_df["sensor"].astype(str) != "nan")
                    )
                    raw_df["col"] = np.where(
                        has_sensor,
                        raw_df["variable"].astype(str) + "_" + raw_df["sensor"].astype(str),
                        raw_df["variable"].astype(str),
                    )
                else:
                    raw_df["col"] = raw_df["variable"].astype(str)

                frames.append(raw_df[["time", "col", "value"]])

        if not frames:
            return pd.DataFrame(columns=["time", "col", "value"])

        return pd.concat(frames, ignore_index=True)

    def extract_station(
        self,
        station_code: str,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        variables: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Download measurements for a single station with non-destructive sensor preservation.

        Mathematical docstring:
        Output table shape: (T_timestamps, C_sensors)
        where column headers are named '<variable>_<sensor_id>' or '<variable>'
        to guarantee zero data loss.
        """
        s_date, e_date = parse_date_range(start_date, end_date)
        if s_date >= e_date:
            raise ValueError(f"start_date ({s_date}) must be prior to end_date ({e_date})")

        endpoint = f"services/measurements/v2/stations/{station_code}/measurements/{self.dataset}"
        flat_frames: List[pd.DataFrame] = []

        filter_vars = None if (variables is None or variables == "all" or (isinstance(variables, (list, tuple)) and "all" in variables)) else variables
        if filter_vars is not None:
            filter_vars = [tahmo_canonical_to_short(v) for v in filter_vars]
        cur = s_date
        while cur < e_date:
            next_step = min(cur + timedelta(days=MAX_REQUEST_DAYS), e_date)
            params = {
                "start": cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": next_step.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            if filter_vars and len(filter_vars) == 1:
                params["variable"] = filter_vars[0]

            resp_json = self._request(endpoint, params=params)
            chunk_df = self._parse_measurements_fast(resp_json, filter_vars)
            if not chunk_df.empty:
                flat_frames.append(chunk_df)
            cur = next_step + timedelta(seconds=1)

        if not flat_frames:
            return pd.DataFrame()

        combined = pd.concat(flat_frames, ignore_index=True)
        combined["time"] = pd.to_datetime(combined["time"], utc=True)
        combined = combined.drop_duplicates(subset=["time", "col"], keep="last")
        pivoted = combined.pivot(index="time", columns="col", values="value").astype(np.float32)
        pivoted.index.name = "time"
        return pivoted

    def _parse_measurements_payload(
        self,
        payload: Dict[str, Any],
        variables: Optional[Sequence[str]],
        series_holder: Dict[str, List[Any]],
    ) -> None:
        """Extract measurement rows respecting quality control flags."""
        results = payload.get("results", [])
        for res in results:
            for s in res.get("series", []):
                cols = s.get("columns", [])
                vals = s.get("values", [])
                if not cols or not vals:
                    continue

                idx_t = cols.index("time")
                idx_q = cols.index("quality")
                idx_v = cols.index("variable")
                idx_s = cols.index("sensor")
                idx_val = cols.index("value")

                for row in vals:
                    var_code = row[idx_v]
                    if variables and var_code not in variables:
                        continue
                    q_flag = row[idx_q]
                    val = row[idx_val] if q_flag == 1 or self.dataset == "raw" else np.nan
                    sensor_id = row[idx_s]
                    timestamp = row[idx_t]

                    series_holder.setdefault(var_code, []).append((timestamp, val, sensor_id, q_flag))

    def _build_non_destructive_dataframe(self, series_holder: Dict[str, List[Any]]) -> pd.DataFrame:
        """
        Assemble observations into non-destructive DataFrame.
        When multiple sensors measure the same variable, each is preserved
        in a distinct '<variable>_<sensor>' column.
        """
        series_list: List[pd.DataFrame] = []
        for var_code, obs_list in series_holder.items():
            if not obs_list:
                continue

            sensors = list({x[2] for x in obs_list if x[2]})
            if len(sensors) > 1:
                for sensor in sensors:
                    sub = [x for x in obs_list if x[2] == sensor]
                    idx = pd.to_datetime([x[0] for x in sub], utc=True)
                    s = pd.Series([x[1] for x in sub], index=idx, dtype=np.float32)
                    s = s.loc[~s.index.duplicated(keep="last")]
                    series_list.append(s.to_frame(f"{var_code}_{sensor}"))
            else:
                idx = pd.to_datetime([x[0] for x in obs_list], utc=True)
                col_name = f"{var_code}_{sensors[0]}" if sensors else var_code
                s = pd.Series([x[1] for x in obs_list], index=idx, dtype=np.float32)
                s = s.loc[~s.index.duplicated(keep="last")]
                series_list.append(s.to_frame(col_name))

        if not series_list:
            return pd.DataFrame()

        merged = pd.concat(series_list, axis=1, sort=True)
        merged.index.name = "time"
        return merged

    def generate_metadata_catalogs(
        self,
        stations_df: Optional[pd.DataFrame] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path]:
        """
        Generate standardized metadata catalogs per PeakWeather specification:
        - installation.parquet: Station inventory (station_id, name, latitude, longitude, elevation, etc.)
        - parameters.parquet: Meteorological parameters inventory (parameter, units, physical type, etc.)

        Mathematical docstring:
        Output catalogs:
          installation: (N_stations, C_metadata) -> {output_dir}/metadata/installation.parquet
          parameters: (N_parameters, C_param_info) -> {output_dir}/metadata/parameters.parquet

        Parameters
        ----------
        stations_df : Optional[pd.DataFrame]
            Pre-queried station metadata DataFrame. If None, queries get_stations().
        output_dir : Optional[Union[str, Path]]
            Directory to save metadata catalogs. Defaults to self.output_dir / "metadata".

        Returns
        -------
        Tuple[Path, Path]
            Paths to (installation.parquet, parameters.parquet).
        """
        meta_dir = Path(output_dir) if output_dir is not None else (self.output_dir / "metadata")
        meta_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build installation catalog
        df_st = stations_df if stations_df is not None else self.get_stations()
        install_rows: List[Dict[str, Any]] = []
        for _, row in df_st.iterrows():
            st_id = str(row.get("station_code") or row.get("station_id") or row.get("id") or "")
            install_rows.append({
                "station_id": st_id,
                "station_name": str(row.get("name") or row.get("station_name") or st_id),
                "latitude": float(row.get("lat") or row.get("latitude") or 0.0),
                "longitude": float(row.get("lon") or row.get("longitude") or 0.0),
                "elevation": float(row.get("elevation") or row.get("elev") or 0.0),
                "network": "tahmo",
                "country": str(row.get("country")) if row.get("country") else "KE",
                "start_date": pd.to_datetime(row.get("start_date") or pd.NaT),
                "end_date": pd.to_datetime(row.get("end_date") or pd.NaT),
                "sensor_count": int(row.get("sensor_count") or 1),
            })
        install_df = pd.DataFrame(install_rows).drop_duplicates(subset=["station_id"])
        install_path = meta_dir / "installation.parquet"
        install_df.to_parquet(install_path, index=False)

        # 2. Build parameters catalog
        param_df = pd.DataFrame(DEFAULT_TAHMO_PARAMETERS)
        param_path = meta_dir / "parameters.parquet"
        param_df.to_parquet(param_path, index=False)

        logger.info(
            "Generated metadata catalogs at %s: installation.parquet (%d stations), parameters.parquet (%d variables)",
            meta_dir, len(install_df), len(param_df),
        )
        return install_path, param_path

    def open_zarr(
        self,
        zarr_path: Union[str, Path],
        **kwargs: Any,
    ) -> Any:
        """
        Lazily open a TAHMO Zarr dataset from disk.

        Parameters
        ----------
        zarr_path : Union[str, Path]
            Path to existing TAHMO Zarr store.

        Returns
        -------
        xr.Dataset
            Lazy xarray Dataset.
        """
        import xarray as xr
        return xr.open_zarr(str(zarr_path), **kwargs)

    def extract(
        self,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        stations: Optional[Sequence[str]] = None,
        bbox: Optional[SpatialBoundingBox] = None,
        variables: Optional[Sequence[str]] = None,
        save_parquet: bool = True,
        save_zarr: bool = False,
        zarr_name: Optional[str] = None,
        generate_metadata: bool = True,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Union[pd.DataFrame, Path]]:
        """
        Execute batch station extraction with legacy-server-respecting concurrency.

        Mathematical docstring:
        Input:
            start_date, end_date: Temporal boundaries
            stations: Sequence of N_target station codes or None
        Output:
            results: Dict[str, Union[pd.DataFrame, Path]] mapping station_code -> Path or DataFrame

        Parameters
        ----------
        start_date : Union[str, datetime, pd.Timestamp]
            Initial observation timestamp.
        end_date : Union[str, datetime, pd.Timestamp]
            Terminal observation timestamp.
        stations : Optional[Sequence[str]]
            Specific station codes to extract.
        bbox : Optional[SpatialBoundingBox]
            Bounding box to filter stations if stations is None.
        variables : Optional[Sequence[str]]
            List of variable codes to extract.
        save_parquet : bool, default True
            Whether to write each station's observations to a local Parquet file.
        save_zarr : bool, default False
            Whether to compile extracted stations into a unified multi-station Zarr store.
        zarr_name : Optional[str]
            Optional filename for compiled Zarr store.
        generate_metadata : bool, default True
            Whether to generate installation.parquet and parameters.parquet catalogs.
        max_workers : Optional[int], default None
            Maximum number of concurrent worker threads. If None, defaults to `self.max_workers`
            (default 4). Kept at a maximum of 4 to respect legacy TAHMO server infrastructure
            and avoid rate limits. Pass 1 for serial execution.

        Returns
        -------
        Dict[str, Union[pd.DataFrame, Path]]
            Dictionary mapping station code to DataFrame or Parquet Path.
        """
        t_range = self.validate_temporal_range(start_date, end_date)
        if stations is not None:
            self.validate_stations(stations, strict=True)
        if variables is not None:
            self.validate_variables(variables, strict=True)
        self.validate_temporal_bounds(t_range, strict=False)

        stations_df = self.get_stations(bbox=bbox or AFRICA_BBOX) if stations is None else None
        if stations is None:
            target_codes = stations_df["station_code"].tolist() if stations_df is not None else []
        else:
            target_codes = list(stations)

        if generate_metadata:
            self.generate_metadata_catalogs(stations_df=stations_df)

        effective_workers = self.max_workers if max_workers is None else max_workers
        logger.info("Extracting TAHMO observations for %d stations (max_workers=%s)...", len(target_codes), effective_workers)
        results: Dict[str, Union[pd.DataFrame, Path]] = {}

        s_dt, e_dt = parse_date_range(start_date, end_date)
        s_str = s_dt.strftime("%Y%m%d")
        e_str = e_dt.strftime("%Y%m%d")

        dataframes: Dict[str, pd.DataFrame] = {}

        def _fetch_single_station(code: str) -> Tuple[str, Optional[pd.DataFrame], Optional[Path]]:
            parquet_path = self.output_dir / f"{code}_{s_str}_{e_str}.parquet"
            if save_parquet and parquet_path.exists() and not save_zarr:
                logger.info("Station %s already extracted: %s", code, parquet_path)
                return code, None, parquet_path

            try:
                df = self.extract_station(
                    station_code=code,
                    start_date=start_date,
                    end_date=end_date,
                    variables=variables,
                )
                if df.empty:
                    logger.info("No data returned for station %s.", code)
                    return code, None, None

                saved_p = None
                if save_parquet:
                    df.reset_index().to_parquet(parquet_path, index=False)
                    saved_p = parquet_path
                return code, df, saved_p
            except Exception as e:
                logger.error("Failed to extract station %s: %s", code, e)
                self.failed_stations[code] = str(e)
                return code, None, None

        if effective_workers and effective_workers > 1 and len(target_codes) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {executor.submit(_fetch_single_station, c): c for c in target_codes}
                for fut in concurrent.futures.as_completed(futures):
                    code, df, p = fut.result()
                    if save_zarr and df is not None:
                        dataframes[code] = df
                    if p is not None:
                        results[code] = p
                    elif df is not None and not save_parquet:
                        results[code] = df
        else:
            for code in target_codes:
                code, df, p = _fetch_single_station(code)
                if save_zarr and df is not None:
                    dataframes[code] = df
                if p is not None:
                    results[code] = p
                elif df is not None and not save_parquet:
                    results[code] = df

        if save_zarr and dataframes:
            import xarray as xr
            # Convert collection of station dataframes to station xarray dataset
            name = zarr_name or f"tahmo_stations_{s_str}_{e_str}.zarr"
            zarr_path = self.output_dir / name

            # Align on master time index
            all_times = pd.DatetimeIndex([])
            for df_st in dataframes.values():
                all_times = all_times.union(df_st.index)
            all_times = all_times.sort_values()

            st_codes = sorted(dataframes.keys())
            var_set: Set[str] = set()
            for df_st in dataframes.values():
                var_set.update(df_st.columns)

            data_vars: Dict[str, Tuple[Tuple[str, str], np.ndarray]] = {}
            for v in sorted(var_set):
                matrix = np.full((len(all_times), len(st_codes)), np.nan, dtype=np.float32)
                for s_idx, code in enumerate(st_codes):
                    df_st = dataframes[code]
                    if v in df_st.columns:
                        s_series = df_st[v]
                        locs = all_times.get_indexer(s_series.index)
                        valid_locs = locs[locs >= 0]
                        matrix[valid_locs, s_idx] = s_series.values[:len(valid_locs)].astype(np.float32)
                data_vars[v] = (("time", "station"), matrix)

            ds_zarr = xr.Dataset(
                data_vars=data_vars,
                coords={"time": all_times, "station": st_codes},
            )
            ds_zarr.to_zarr(str(zarr_path), mode="w", consolidated=True)
            results["_zarr"] = zarr_path
            logger.info("Compiled multi-station TAHMO Zarr store: %s", zarr_path)

        return results
