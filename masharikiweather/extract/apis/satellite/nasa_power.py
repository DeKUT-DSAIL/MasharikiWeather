"""
NASA POWER Solar and Meteorological Satellite Data API Wrapper for MasharikiWeather.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_CACHE_DIR,
    _BaseDatasetWrapper,
    _normalize_locations,
    _parse_date,
    _safe_request,
)

logger = logging.getLogger("MasharikiWeather.NASAPower")

# Parameter catalog with descriptions, units, canonical aliases, and typical data latency
NASA_POWER_CATALOG: Dict[str, Dict[str, Any]] = {
    # --- Solar Irradiance & Radiation ---
    "ALLSKY_SFC_SW_DWN": {
        "aliases": ["ghi", "allsky_ghi", "solar_radiation", "solar_rad", "rad", "surface_solar"],
        "description": "All Sky Surface Shortwave Downward Irradiance (GHI)",
        "units": "W/m^2 (hourly) or kW-hr/m^2/day (daily)",
        "community": "RE",
        "typical_latency_days": "5 to 7 days",
    },
    "CLRSKY_SFC_SW_DWN": {
        "aliases": ["clrsky_ghi", "clearsky_ghi", "clearsky_solar"],
        "description": "Clear Sky Surface Shortwave Downward Irradiance",
        "units": "W/m^2 (hourly) or kW-hr/m^2/day (daily)",
        "community": "RE",
        "typical_latency_days": "5 to 7 days",
    },
    "ALLSKY_SFC_SW_DNI": {
        "aliases": ["dni", "direct_normal_irradiance"],
        "description": "All Sky Surface Direct Normal Irradiance",
        "units": "W/m^2 (hourly) or kW-hr/m^2/day (daily)",
        "community": "RE",
        "typical_latency_days": "5 to 7 days",
    },
    "ALLSKY_SFC_SW_DIFF": {
        "aliases": ["dhi", "diffuse", "diffuse_irradiance"],
        "description": "All Sky Surface Diffuse Horizontal Irradiance",
        "units": "W/m^2 (hourly) or kW-hr/m^2/day (daily)",
        "community": "RE",
        "typical_latency_days": "5 to 7 days",
    },
    "ALLSKY_KT": {
        "aliases": ["kt", "clearness_index"],
        "description": "Insolation Clearness Index (0 to 1)",
        "units": "dimensionless",
        "community": "RE",
        "typical_latency_days": "5 to 7 days",
    },
    "ALLSKY_SFC_UV_INDEX": {
        "aliases": ["uv_index", "uvi"],
        "description": "All Sky Surface Ultraviolet Index",
        "units": "dimensionless",
        "community": "RE",
        "typical_latency_days": "5 to 7 days",
    },
    "ALLSKY_SFC_PAR_TOT": {
        "aliases": ["par", "photosynthetically_active_radiation"],
        "description": "All Sky Surface PAR Total",
        "units": "W/m^2",
        "community": "RE",
        "typical_latency_days": "5 to 7 days",
    },
    "TOA_SW_DWN": {
        "aliases": ["toa", "toa_dwn", "extraterrestrial_radiation"],
        "description": "Top-Of-Atmosphere Shortwave Downward Flux",
        "units": "W/m^2 (hourly) or kW-hr/m^2/day (daily)",
        "community": "RE",
        "typical_latency_days": "5 to 7 days",
    },

    # --- Temperature & Thermal ---
    "T2M": {
        "aliases": ["t2m", "temp", "temperature"],
        "description": "Temperature at 2 Meters",
        "units": "°C",
        "community": "RE",
        "typical_latency_days": "3 to 4 days",
    },
    "T2M_MAX": {
        "aliases": ["tmax", "max_temp", "t2m_max"],
        "description": "Maximum Temperature at 2 Meters",
        "units": "°C",
        "community": "RE",
        "typical_latency_days": "3 to 4 days",
    },
    "T2M_MIN": {
        "aliases": ["tmin", "min_temp", "t2m_min"],
        "description": "Minimum Temperature at 2 Meters",
        "units": "°C",
        "community": "RE",
        "typical_latency_days": "3 to 4 days",
    },
    "T2MDEW": {
        "aliases": ["t2mdew", "dew_point", "dewpoint"],
        "description": "Dew/Frost Point at 2 Meters",
        "units": "°C",
        "community": "RE",
        "typical_latency_days": "3 to 4 days",
    },
    "T2M_RANGE": {
        "aliases": ["temp_range", "t2m_range"],
        "description": "Temperature Range at 2 Meters (Tmax - Tmin)",
        "units": "°C",
        "community": "RE",
        "typical_latency_days": "3 to 4 days",
    },
    "TS": {
        "aliases": ["ts", "skin_temp", "surface_temperature"],
        "description": "Earth Skin Temperature",
        "units": "°C",
        "community": "RE",
        "typical_latency_days": "3 to 4 days",
    },

    # --- Moisture, Humidity & Precipitation ---
    "RH2M": {
        "aliases": ["rh2m", "rh", "relative_humidity"],
        "description": "Relative Humidity at 2 Meters",
        "units": "%",
        "community": "RE",
        "typical_latency_days": "3 to 4 days",
    },
    "QV2M": {
        "aliases": ["qv2m", "specific_humidity"],
        "description": "Specific Humidity at 2 Meters",
        "units": "g/kg",
        "community": "RE",
        "typical_latency_days": "3 to 4 days",
    },
    "PRECTOTCORR": {
        "aliases": ["precip", "precipitation", "rain", "prectotcorr"],
        "description": "Corrected Total Precipitation",
        "units": "mm/hour (hourly) or mm/day (daily)",
        "community": "RE",
        "typical_latency_days": "3 to 4 days",
    },

    # --- Wind & Atmospheric Pressure ---
    "WS10M": {
        "aliases": ["ws10m", "wind_speed", "wind"],
        "description": "Wind Speed at 10 Meters",
        "units": "m/s",
        "community": "RE",
        "typical_latency_days": "3 to 4 days",
    },
    "WD10M": {
        "aliases": ["wd10m", "wind_direction", "wind_dir"],
        "description": "Wind Direction at 10 Meters",
        "units": "Degrees",
        "community": "RE",
        "typical_latency_days": "3 to 4 days",
    },
    "WS50M": {
        "aliases": ["ws50m", "wind_speed_50m"],
        "description": "Wind Speed at 50 Meters",
        "units": "m/s",
        "community": "RE",
        "typical_latency_days": "3 to 4 days",
    },
    "PS": {
        "aliases": ["ps", "surface_pressure", "pressure"],
        "description": "Surface Pressure",
        "units": "kPa",
        "community": "RE",
        "typical_latency_days": "3 to 4 days",
    },

    # --- Cloud Cover ---
    "CLOUD_AMT": {
        "aliases": ["cloud_amt", "cloud_cover", "cloud"],
        "description": "Cloud Amount",
        "units": "%",
        "community": "RE",
        "typical_latency_days": "3 to 4 days",
    },

    # --- Agroclimatology & Soil Moisture ---
    "GWETROOT": {
        "aliases": ["gwetroot", "soil_moisture_root", "root_zone_moisture"],
        "description": "Root Zone Soil Wetness (0 to 1)",
        "units": "dimensionless",
        "community": "AG",
        "typical_latency_days": "3 to 5 days",
    },
    "GWETTOP": {
        "aliases": ["gwettop", "soil_moisture_top", "surface_soil_moisture"],
        "description": "Surface Soil Wetness (0 to 1)",
        "units": "dimensionless",
        "community": "AG",
        "typical_latency_days": "3 to 5 days",
    },
    "EVLAND": {
        "aliases": ["evland", "evaporation"],
        "description": "Evaporation Land",
        "units": "mm/day",
        "community": "AG",
        "typical_latency_days": "3 to 5 days",
    },
}

# Construct reverse lexicon: canonical alias -> NASA POWER parameter code
NASA_POWER_LEXICON: Dict[str, str] = {}
for code, meta in NASA_POWER_CATALOG.items():
    NASA_POWER_LEXICON[code.lower()] = code
    for alias in meta["aliases"]:
        NASA_POWER_LEXICON[alias.lower()] = code

DEFAULT_POWER_VARS = ["ghi", "clrsky_ghi", "dni", "dhi", "kt", "cloud_amt", "t2m"]

# Maximum parameters per single API query (NASA enforced limits)
MAX_PARAMS_PER_QUERY = {
    "hourly": 15,
    "daily": 20,
    "monthly": 20,
    "climatology": 20,
}


def list_parameters() -> List[str]:
    """Return all supported friendly parameter aliases."""
    return sorted(list(NASA_POWER_LEXICON.keys()))


def search_parameters(query: Optional[str] = None) -> pd.DataFrame:
    """
    Search the NASA POWER parameter dictionary by keyword or view all parameters.

    Parameters
    ----------
    query : str, optional
        Keyword to search across parameter codes, aliases, descriptions, or units.
        If None or empty, returns the full catalog.

    Returns
    -------
    pd.DataFrame
        Table with parameter code, primary alias, description, units, community, and typical latency.
    """
    rows = []
    q = (query or "").lower().strip()
    for code, meta in NASA_POWER_CATALOG.items():
        aliases_str = ", ".join(meta["aliases"])
        desc = meta["description"]
        units = meta["units"]
        comm = meta["community"]
        latency = meta.get("typical_latency_days", "3 to 5 days")
        searchable_text = f"{code} {aliases_str} {desc} {units} {comm} {latency}".lower()

        if not q or q in searchable_text:
            rows.append({
                "code": code,
                "primary_alias": meta["aliases"][0] if meta["aliases"] else code.lower(),
                "all_aliases": aliases_str,
                "description": desc,
                "units": units,
                "community": comm,
                "typical_latency": latency,
            })
    return pd.DataFrame(rows)


def get_parameter_latency() -> pd.DataFrame:
    """
    Return a comprehensive overview of expected data latency across variable streams.

    Returns
    -------
    pd.DataFrame
        Table summarizing variable groups, codes, units, and expected release latencies.
    """
    rows = []
    for code, meta in NASA_POWER_CATALOG.items():
        if "SW" in code or code in ("ALLSKY_KT", "ALLSKY_SFC_UV_INDEX", "ALLSKY_SFC_PAR_TOT"):
            cat = "Solar Irradiance (CERES / GMAO)"
        elif "GWET" in code or code == "EVLAND":
            cat = "Agroclimatology / Soil (GMAO Land)"
        else:
            cat = "Surface Meteorology (GMAO GEOS)"

        rows.append({
            "code": code,
            "primary_alias": meta["aliases"][0] if meta["aliases"] else code.lower(),
            "category": cat,
            "typical_latency": meta.get("typical_latency_days", "3 to 5 days"),
            "units": meta["units"],
            "description": meta["description"],
        })
    return pd.DataFrame(rows)


def get_latest_available_dates(
    lat: float = -0.398,
    lon: float = 36.958,
    variables: Optional[List[str]] = None,
    resolution: str = "daily",
    community: str = "RE",
    lookback_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Probe the live NASA POWER API to discover the exact latest date populated
    with valid (non-fill) data for each requested variable at the given coordinate.

    Important Science Latency Note:
    -------------------------------
    - Surface Meteorology (temp, wind, rain, rh, pressure):
      Derived from NASA GMAO GEOS FP-IT assimilation model.
      Latency: ~3 to 4 days behind real-time (both daily and hourly).
    - Daily Solar Irradiance (daily ghi, dni, dhi, kt):
      Supported by GMAO GEOS fast radiation estimates in NRT mode.
      Latency: ~5 to 7 days behind real-time.
    - Hourly Solar Irradiance (hourly ghi, dni, dhi, kt, cloud_amt):
      Derived from the NASA CERES SYN1deg multi-satellite product
      (Terra + Aqua + 5 geostationary satellites).
      Latency: ~2 to 3 months (~60 to 90 days) behind real-time due to
      rigorous cross-satellite calibration and monthly batch releases.

    Parameters
    ----------
    lat, lon : float
        Target latitude and longitude. Defaults to DeKUT Nyeri (-0.398, 36.958).
    variables : list of str, optional
        Variables to probe. Defaults to core weather and solar variables.
    resolution : str, default 'daily'
        'daily' or 'hourly'.
    community : str, default 'RE'
        'RE', 'AG', or 'SB'.
    lookback_days : int, optional
        Days to probe backwards from today's UTC timestamp.
        Defaults to 21 days for daily, 90 days for hourly.

    Returns
    -------
    pd.DataFrame
        Table with 'variable', 'code', 'latest_available_date', 'latency_days', 'units', and 'latest_value'.
    """
    res_clean = resolution.lower().strip()
    if lookback_days is None:
        lookback_days = 90 if res_clean == "hourly" else 21

    vars_requested = variables or (
        ["temp", "tmax", "tmin", "rain", "wind", "rh", "ghi"]
        if res_clean == "daily"
        else ["temp", "wind", "rain", "rh", "pressure", "ghi", "dni", "dhi"]
    )

    # Map to NASA codes
    nasa_params: List[str] = []
    param_to_alias: Dict[str, str] = {}
    for v in vars_requested:
        canon = v.lower().strip()
        code = NASA_POWER_LEXICON.get(canon, v.upper())
        nasa_params.append(code)
        param_to_alias[code] = canon

    nasa_params = sorted(list(set(nasa_params)))

    now = datetime.now(timezone.utc)
    end_str = now.strftime("%Y%m%d")
    start_str = (now - timedelta(days=lookback_days)).strftime("%Y%m%d")

    base_endpoint = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        if res_clean == "hourly"
        else "https://power.larc.nasa.gov/api/temporal/daily/point"
    )
    query_params = {
        "parameters": ",".join(nasa_params),
        "community": community.upper(),
        "longitude": f"{lon:.4f}",
        "latitude": f"{lat:.4f}",
        "start": start_str,
        "end": end_str,
        "format": "JSON",
    }
    if res_clean == "hourly":
        query_params["time-standard"] = "UTC"

    url = f"{base_endpoint}?{urllib.parse.urlencode(query_params)}"

    raw_bytes = _safe_request(url)
    data = json.loads(raw_bytes.decode("utf-8"))
    param_dict = data.get("properties", {}).get("parameter", {})

    rows = []
    for code, ts_dict in param_dict.items():
        valid = {dt: val for dt, val in ts_dict.items() if val is not None and float(val) != -999.0}
        alias = param_to_alias.get(code, code.lower())
        meta = NASA_POWER_CATALOG.get(code, {})
        units = meta.get("units", "unknown")

        if valid:
            latest_str = sorted(valid.keys())[-1]
            if len(latest_str) == 10:  # Hourly: YYYYMMDDHH
                latest_dt = datetime.strptime(latest_str, "%Y%m%d%H").replace(tzinfo=timezone.utc)
                latest_fmt = f"{latest_str[:4]}-{latest_str[4:6]}-{latest_str[6:8]} {latest_str[8:]}:00 UTC"
            else:  # Daily: YYYYMMDD
                latest_dt = datetime.strptime(latest_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                latest_fmt = f"{latest_str[:4]}-{latest_str[4:6]}-{latest_str[6:]}"

            latency = (now - latest_dt).days
            rows.append({
                "variable": alias,
                "code": code,
                "latest_available_date": latest_fmt,
                "latency_days": latency,
                "latest_value": valid[latest_str],
                "units": units,
            })
        else:
            rows.append({
                "variable": alias,
                "code": code,
                "latest_available_date": f"No valid data in {lookback_days}-day window",
                "latency_days": None,
                "latest_value": None,
                "units": units,
            })

    return pd.DataFrame(rows)


class NASAPower(_BaseDatasetWrapper):
    """
    Client for querying NASA POWER solar and meteorological telemetry.
    No API key required (public domain data).

    Supported resolutions:
    - 'hourly': Hourly solar radiation & weather
    - 'daily': Daily aggregated metrics (default)
    - 'monthly': Monthly averages
    - 'climatology': Multi-year climatological normals

    Supported communities:
    - 'RE': Renewable Energy (default, includes solar and wind)
    - 'AG': Agroclimatology (includes soil moisture, evapotranspiration)
    - 'SB': Sustainable Buildings
    """

    BASE_URLS = {
        "hourly": "https://power.larc.nasa.gov/api/temporal/hourly/point",
        "daily": "https://power.larc.nasa.gov/api/temporal/daily/point",
        "monthly": "https://power.larc.nasa.gov/api/temporal/monthly/point",
        "climatology": "https://power.larc.nasa.gov/api/temporal/climatology/point",
    }

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        rate_limit_delay: float = 0.2,
        max_workers: int = 4,
    ) -> None:
        super().__init__(
            cache_dir=cache_dir,
            rate_limit_delay=rate_limit_delay,
            max_workers=max_workers,
        )

    def probe_latest_date(
        self,
        lat: float,
        lon: float,
        variables: Optional[List[str]] = None,
        community: str = "RE",
    ) -> pd.DataFrame:
        """Probe the live NASA POWER API for latest available dates."""
        return get_latest_available_dates(lat=lat, lon=lon, variables=variables, community=community)

    def fetch_point(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None,
        resolution: str = "hourly",
        community: str = "RE",
        location_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch time series for a single (lat, lon) point.
        Automatically chunks parameters into batches if the list exceeds NASA API limits.
        """
        res_key = resolution.lower().strip()
        if res_key not in self.BASE_URLS:
            raise ValueError(
                f"Unknown resolution '{resolution}'. Supported: {list(self.BASE_URLS.keys())}"
            )

        loc_label = location_name or f"{lat:.3f}_{lon:.3f}"
        if variables == "all" or (isinstance(variables, (list, tuple)) and "all" in variables):
            vars_requested = list(NASA_POWER_CATALOG.keys())
        else:
            vars_requested = variables or DEFAULT_POWER_VARS

        # Resolve canonical NASA parameter names
        nasa_params: List[str] = []
        param_to_canonical: Dict[str, str] = {}
        for v in vars_requested:
            canon = v.lower().strip()
            nasa_p = NASA_POWER_LEXICON.get(canon, v.upper())
            nasa_params.append(nasa_p)
            param_to_canonical[nasa_p] = canon

        nasa_params = sorted(list(set(nasa_params)))

        # Check local cache
        cache_key = {
            "source": "nasa_power",
            "res": res_key,
            "community": community.upper(),
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "start": start_date,
            "end": end_date,
            "params": nasa_params,
        }
        cache_file = self._get_cache_path("power", cache_key)
        if self._is_cache_valid(cache_file):
            try:
                cached_df = pd.read_parquet(cache_file)
                return cached_df.to_dict(orient="records")
            except Exception:
                pass

        # Split into chunks if parameters exceed maximum per request
        max_chunk = MAX_PARAMS_PER_QUERY.get(res_key, 15)
        param_chunks = [
            nasa_params[i : i + max_chunk] for i in range(0, len(nasa_params), max_chunk)
        ]

        records: List[Dict[str, Any]] = []
        base_url = self.BASE_URLS[res_key]

        # Format dates depending on temporal resolution
        s_clean = start_date.replace("-", "")
        e_clean = end_date.replace("-", "")
        if res_key in ("monthly", "climatology"):
            s_param = s_clean[:4]
            e_param = e_clean[:4]
        else:
            s_param = s_clean[:8]
            e_param = e_clean[:8]

        for chunk in param_chunks:
            query_dict = {
                "parameters": ",".join(chunk),
                "community": community.upper(),
                "longitude": f"{lon:.4f}",
                "latitude": f"{lat:.4f}",
                "format": "JSON",
            }
            if res_key == "hourly":
                query_dict["time-standard"] = "UTC"
            if res_key != "climatology":
                query_dict["start"] = s_param
                query_dict["end"] = e_param

            url = f"{base_url}?{urllib.parse.urlencode(query_dict)}"

            time.sleep(self.rate_limit_delay)
            raw_bytes = _safe_request(url)
            data = json.loads(raw_bytes.decode("utf-8"))

            param_dict = data.get("properties", {}).get("parameter", {})

            for p_name, ts_dict in param_dict.items():
                canon_var = param_to_canonical.get(p_name, p_name.lower())
                for t_str, val in ts_dict.items():
                    if val is None or float(val) == -999.0:
                        continue

                    # Parse timestamp by string length
                    if len(t_str) == 10:  # Hourly: YYYYMMDDHH
                        dt = pd.Timestamp(
                            f"{t_str[:4]}-{t_str[4:6]}-{t_str[6:8]} {t_str[8:10]}:00", tz="UTC"
                        )
                    elif len(t_str) == 8:  # Daily: YYYYMMDD
                        dt = pd.Timestamp(
                            f"{t_str[:4]}-{t_str[4:6]}-{t_str[6:8]} 00:00:00", tz="UTC"
                        )
                    elif len(t_str) == 6:  # Monthly: YYYYMM
                        # Key 13 represents the annual average; skip for monthly time series
                        if t_str[4:6] == "13":
                            continue
                        dt = pd.Timestamp(
                            f"{t_str[:4]}-{t_str[4:6]}-01 00:00:00", tz="UTC"
                        )
                    else:
                        dt = pd.Timestamp(f"2000-01-01 00:00:00", tz="UTC")

                    records.append({
                        "time": dt,
                        "location": loc_label,
                        "lat": lat,
                        "lon": lon,
                        "variable": canon_var,
                        "value": float(val),
                    })

        if records:
            try:
                pd.DataFrame(records).to_parquet(cache_file, index=False)
            except Exception:
                pass

        return records

    def query(
        self,
        locations: Optional[Union[Dict[str, float], Tuple[float, float], List[Any], str]] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        region: Optional[str] = None,
        variables: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        resolution: str = "hourly",
        community: str = "RE",
        format: str = "wide",
    ) -> pd.DataFrame:
        """
        Query NASA POWER data across one or more locations.
        Defaults to Kenya/Africa priority when no location is given.
        """
        locs = _normalize_locations(locations=locations, lat=lat, lon=lon, region=region)
        s_date = _parse_date(start_date, "2024-01-01")
        e_date = _parse_date(end_date, "2024-01-07")

        all_records: List[Dict[str, Any]] = []

        if len(locs) == 1:
            loc = locs[0]
            all_records = self.fetch_point(
                lat=loc["lat"],
                lon=loc["lon"],
                start_date=s_date,
                end_date=e_date,
                variables=variables,
                resolution=resolution,
                community=community,
                location_name=loc["name"],
            )
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self.fetch_point,
                        lat=loc["lat"],
                        lon=loc["lon"],
                        start_date=s_date,
                        end_date=e_date,
                        variables=variables,
                        resolution=resolution,
                        community=community,
                        location_name=loc["name"],
                    )
                    for loc in locs
                ]
                for f in as_completed(futures):
                    try:
                        all_records.extend(f.result())
                    except Exception as e:
                        logger.warning(f"NASA POWER point query failed: {e}")

        return self._reshape_output(all_records, format=format)


def query_nasa_power(
    locations: Optional[Union[Dict[str, float], Tuple[float, float], List[Any], str]] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    region: Optional[str] = None,
    variables: Optional[List[str]] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    resolution: str = "hourly",
    community: str = "RE",
    format: str = "wide",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Convenience function to query NASA POWER solar irradiance and meteorology in 1-2 lines.
    Zero API key required.
    """
    client = NASAPower(cache_dir=cache_dir)
    return client.query(
        locations=locations,
        lat=lat,
        lon=lon,
        region=region,
        variables=variables,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
        community=community,
        format=format,
    )


# Aliases
NasaPower = NASAPower
NasaPowerWrapper = NASAPower
NASAPowerWrapper = NASAPower
get_nasa_power_data = query_nasa_power
NASA_POWER_PARAMETERS = NASA_POWER_CATALOG
