"""
Copernicus CDS In-Situ Surface Observations Extraction Engine.

Queries in-situ ground station meteorological observations from the Copernicus
Climate Data Store ('insitu-observations-surface-land').
Handles bounding box subsetting, annual sequential splitting to prevent CDS quotas,
un-nests raw long-format byte-string payloads onto CF-compliant wide tables,
and saves parsed observations as local Parquet files.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import (
    AFRICA_BBOX,
    BaseExtractor,
    SpatialBoundingBox,
    TemporalRange,
)
from .credentials import CredentialsError, CredentialsManager, parse_date, parse_date_range

logger = logging.getLogger("MasharikiWeather.CDSInSitu")

DEFAULT_INSITU_VARIABLES = [
    "accumulated_precipitation",
    "air_pressure",
    "air_pressure_at_sea_level",
    "air_temperature",
    "dew_point_temperature",
    "fresh_snow",
    "snow_depth",
    "snow_water_equivalent",
    "wind_from_direction",
    "wind_speed",
]

# Bidirectional Cross-Source Harmonization Registry for CDS In-Situ
CDS_INSITU_RAW_TO_CANONICAL: Dict[str, str] = {
    "air_temperature": "2m_temperature",
    "dew_point_temperature": "2m_dewpoint_temperature",
    "accumulated_precipitation": "total_precipitation",
    "wind_speed": "wind_speed",
    "wind_from_direction": "wind_from_direction",
    "air_pressure": "surface_pressure",
    "air_pressure_at_sea_level": "mean_sea_level_pressure",
    "relative_humidity": "relative_humidity",
    "fresh_snow": "snowfall",
    "snow_depth": "snow_depth",
    "snow_water_equivalent": "snow_water_equivalent",
}

CDS_INSITU_CANONICAL_TO_RAW: Dict[str, str] = {
    v: k for k, v in CDS_INSITU_RAW_TO_CANONICAL.items()
}


def canonicalize_cds_insitu_variable(var: str) -> str:
    """Map raw CDS in-situ variable identifier to canonical descriptive name."""
    return CDS_INSITU_RAW_TO_CANONICAL.get(var, var)


def cds_insitu_canonical_to_raw(var: str) -> str:
    """Map canonical descriptive name to raw CDS in-situ parameter string."""
    return CDS_INSITU_CANONICAL_TO_RAW.get(var, var)


def _safe_decode_bytes(val: Any) -> Any:
    """Safely decode byte strings or object arrays to utf-8 strings."""
    if isinstance(val, (bytes, np.bytes_)):
        try:
            return val.decode("utf-8").strip()
        except Exception:
            return str(val).strip("b'\" ")
    elif isinstance(val, str):
        return val.strip()
    return val


def parse_cds_insitu_dataframe(
    raw_df: pd.DataFrame,
    harmonize: bool = True,
    filter_quality: bool = False,
) -> pd.DataFrame:
    """
    Parse, un-nest, and pivot raw Copernicus CDS in-situ observation table.

    Addresses CDS In-Situ structural complexities:
    1. Byte-string Decoding: Automatically decodes binary literals (e.g. b'air_temperature').
    2. Nested Variable Un-nesting: Pivots 'observed_variable' and 'observation_value' onto
       standard wide-format multi-variable schema.
    3. Flat Index Pivot: Re-indexes from flat 1D sequence to (time, station_id).
    4. Station Disambiguation: Separates 'primary_station_id' from human-readable 'station_name'.
    5. Elevation Standardization: Extracts elevation from 'height_of_station_above_sea_level'.
    6. Variable Harmonization: Harmonizes raw parameter names to canonical descriptive names.

    Mathematical docstring:
    Input table shape: (N_raw_obs, C_raw) along flat index dimension.
    Output table shape: (N_pivoted_records, C_canonical)
    where:
      Columns contain ['time', 'station_id', 'station_name', 'latitude', 'longitude', 'elevation', ...variables...]
    """
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    # 1. Decode byte-string columns
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(_safe_decode_bytes)

    # 2. Harmonize column name aliases
    col_aliases = {
        "primary_station_id": "station_id",
        "date_time": "time",
        "report_timestamp": "time",
        "height_of_station_above_sea_level": "elevation",
        "station_elevation": "elevation",
    }
    for old_c, new_c in col_aliases.items():
        if old_c in df.columns and new_c not in df.columns:
            df[new_c] = df[old_c]

    # Required core identification columns
    if "station_id" not in df.columns and "primary_station_id" in df.columns:
        df["station_id"] = df["primary_station_id"]
    if "time" not in df.columns:
        for t_candidate in ("time", "date_time", "report_timestamp", "observation_time"):
            if t_candidate in df.columns:
                df["time"] = df[t_candidate]
                break

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])

    # If observed_variable and observation_value are present, un-nest / pivot
    if "observed_variable" in df.columns and "observation_value" in df.columns:
        # Optional QC filtering
        if filter_quality and "quality_flag" in df.columns:
            bad_mask = df["quality_flag"].isin([1, 2, 3])
            df.loc[bad_mask, "observation_value"] = np.nan

        # Map observed_variable to canonical descriptive name if requested
        if harmonize:
            df["observed_variable"] = df["observed_variable"].map(
                lambda v: CDS_INSITU_RAW_TO_CANONICAL.get(v, v)
            )

        # Build station metadata lookup
        meta_cols = [c for c in ["station_id", "station_name", "latitude", "longitude", "elevation"] if c in df.columns]
        station_meta = df[meta_cols].drop_duplicates(subset=["station_id"]).set_index("station_id")

        # Pivot to (time, station_id)
        pivot_idx = ["time", "station_id"] if "station_id" in df.columns else ["time"]
        pivoted = df.pivot_table(
            index=pivot_idx,
            columns="observed_variable",
            values="observation_value",
            aggfunc="first",
        ).reset_index()

        # Re-attach station metadata
        if "station_id" in pivoted.columns and not station_meta.empty:
            for m_col in ["station_name", "latitude", "longitude", "elevation"]:
                if m_col in station_meta.columns and m_col not in pivoted.columns:
                    pivoted[m_col] = pivoted["station_id"].map(station_meta[m_col])

        ordered_first = [c for c in ["time", "station_id", "station_name", "latitude", "longitude", "elevation"] if c in pivoted.columns]
        var_cols = [c for c in pivoted.columns if c not in ordered_first]
        pivoted = pivoted[ordered_first + sorted(var_cols)]
        return pivoted

    # If already wide format:
    if harmonize:
        rename_vars = {k: v for k, v in CDS_INSITU_RAW_TO_CANONICAL.items() if k in df.columns}
        if rename_vars:
            df = df.rename(columns=rename_vars)

    return df


def parse_cds_insitu_dataset(
    source: Union[str, Path, xr.Dataset],
    harmonize: bool = True,
    filter_quality: bool = False,
) -> pd.DataFrame:
    """
    Parse a Copernicus CDS in-situ NetCDF file or xarray.Dataset into a pivoted Parquet-ready DataFrame.

    Parameters
    ----------
    source : Union[str, Path, xr.Dataset]
        Local NetCDF file path or open xarray Dataset.
    harmonize : bool, default True
        If True, translates raw CDS variable names to canonical descriptive names.
    filter_quality : bool, default False
        If True, converts suspect quality flag readings to NaN.

    Returns
    -------
    pd.DataFrame
        CF-compliant pivoted DataFrame.
    """
    if isinstance(source, xr.Dataset):
        df_raw = source.to_dataframe().reset_index()
    else:
        with xr.open_dataset(source) as ds:
            df_raw = ds.to_dataframe().reset_index()
    return parse_cds_insitu_dataframe(df_raw, harmonize=harmonize, filter_quality=filter_quality)


class CDSInSituExtractor(BaseExtractor):
    """
    Automated extraction pipeline for Copernicus CDS in-situ surface land observations.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "./data/cds_insitu",
        variables: Optional[Sequence[str]] = None,
        dataset_name: str = "insitu-observations-surface-land",
        time_aggregation: str = "daily",
        version: str = "3_0_0",
        cds_key: Optional[str] = None,
        cds_url: Optional[str] = None,
        rc_file: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize Copernicus CDS in-situ observation extractor and target cache directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.variables = list(variables if variables is not None else DEFAULT_INSITU_VARIABLES)
        self.dataset_name = dataset_name
        self.time_aggregation = time_aggregation
        self.version = version
        self.cds_key = cds_key
        self.cds_url = cds_url
        self.rc_file = rc_file
        self._client: Optional[Any] = None

    def _get_client(self, cds_key: Optional[str] = None, cds_url: Optional[str] = None) -> Any:
        """Lazily initialize cdsapi.Client with discovered credentials."""
        if self._client is None or cds_key or cds_url:
            try:
                import cdsapi
            except ImportError as exc:
                raise ImportError(
                    "cdsapi is required for CDS In-Situ extraction. Install via 'pip install cdsapi'."
                ) from exc
            creds = CredentialsManager.get_cds_credentials(
                key=cds_key or self.cds_key,
                url=cds_url or self.cds_url,
                rc_file=self.rc_file,
            )
            client_kwargs: Dict[str, Any] = {}
            if creds.get("url"):
                client_kwargs["url"] = creds["url"]
            if creds.get("key"):
                client_kwargs["key"] = creds["key"]
            client = cdsapi.Client(**client_kwargs)
            if not cds_key and not cds_url:
                self._client = client
            return client
        return self._client

    def _build_request_payload(
        self,
        year: int,
        bbox: SpatialBoundingBox,
        variables: Sequence[str],
        data_format: str = "netcdf",
    ) -> Dict[str, Any]:
        """
        Construct CDS API request payload for in-situ observations.
        """
        # Map any canonical variable names to raw CDS parameters
        raw_vars = [cds_insitu_canonical_to_raw(v) for v in variables]
        return {
            "version": self.version,
            "time_aggregation": self.time_aggregation,
            "variable": raw_vars,
            "year": str(year),
            "month": [str(m).zfill(2) for m in range(1, 13)],
            "day": [str(d).zfill(2) for d in range(1, 32)],
            "data_format": data_format,
            "area": bbox.to_area_list(),
        }

    def extract_year(
        self,
        year: int,
        bbox: Optional[SpatialBoundingBox] = None,
        variables: Optional[Sequence[str]] = None,
        output_format: str = "parquet",
        harmonize: bool = True,
        filter_quality: bool = False,
    ) -> Path:
        """
        Download in-situ observations for a single year, un-nest, and save to Parquet.

        Mathematical docstring:
        Output table schema: (N_observations, C_features)
        where columns contain ['time', 'station_id', 'station_name', 'latitude', 'longitude', 'elevation', ...variables...].
        """
        box = bbox if bbox is not None else AFRICA_BBOX
        vars_to_fetch = variables if variables is not None else self.variables
        client = self._get_client()

        target_parquet = self.output_dir / f"insitu_surface_{year}.parquet"
        target_nc = self.output_dir / f"insitu_surface_{year}.nc"

        if output_format == "parquet" and target_parquet.exists():
            logger.info("File already exists locally: %s", target_parquet)
            return target_parquet
        if output_format == "netcdf" and target_nc.exists():
            logger.info("File already exists locally: %s", target_nc)
            return target_nc

        req = self._build_request_payload(year=year, bbox=box, variables=vars_to_fetch, data_format="netcdf")

        tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".nc")
        os.close(tmp_fd)

        try:
            logger.info("Requesting CDS in-situ data for year %d...", year)
            client.retrieve(self.dataset_name, req, tmp_path_str)

            if output_format == "parquet":
                logger.info("Parsing and converting NetCDF in-situ observations to Parquet: %s", target_parquet)
                try:
                    df = parse_cds_insitu_dataset(tmp_path_str, harmonize=harmonize, filter_quality=filter_quality)
                    df.to_parquet(target_parquet, index=False)
                    return target_parquet
                except Exception as conv_err:
                    logger.warning("Un-nesting failed (%s), saving raw NetCDF instead.", conv_err)
                    shutil.move(tmp_path_str, target_nc)
                    return target_nc
            else:
                shutil.move(tmp_path_str, target_nc)
                return target_nc

        finally:
            if os.path.exists(tmp_path_str):
                try:
                    os.remove(tmp_path_str)
                except OSError:
                    pass

    def extract(
        self,
        years: Optional[Union[int, Sequence[int]]] = None,
        start_date: Optional[Union[str, datetime, date, pd.Timestamp]] = None,
        end_date: Optional[Union[str, datetime, date, pd.Timestamp]] = None,
        bbox: Optional[SpatialBoundingBox] = None,
        variables: Optional[Sequence[str]] = None,
        output_format: str = "parquet",
        harmonize: bool = True,
        cds_key: Optional[str] = None,
        cds_url: Optional[str] = None,
    ) -> List[Path]:
        """
        Execute sequential extraction across specified years.
        Supports direct year list or start_date/end_date range.
        """
        if years is not None:
            year_list = [years] if isinstance(years, int) else list(years)
        elif start_date is not None and end_date is not None:
            s_dt, e_dt = parse_date_range(start_date, end_date)
            year_list = list(range(s_dt.year, e_dt.year + 1))
        elif start_date is not None:
            year_list = [parse_date(start_date).year]
        else:
            raise ValueError("Must provide either 'years' or 'start_date' and 'end_date'.")

        results: List[Path] = []
        for y in year_list:
            res_path = self.extract_year(
                year=y,
                bbox=bbox,
                variables=variables,
                output_format=output_format,
                harmonize=harmonize,
            )
            results.append(res_path)
        return results
