"""
MasharikiWeather Unified Data Extraction and API Ingestion Suite.

Exposes production-ready, CF-compliant extraction engines across:
- In-situ Ground Stations: TAHMO, GHCN-Daily, GHCN-Hourly, CDS In-Situ
- Gridded Reanalysis: ARCO-ERA5, Copernicus CDS ERA5, CAMS, MERRA-2, Open-Meteo
- Satellite Remote Sensing: NASA IMERG, NASA POWER, PVGIS, CM SAF SARAH-3, LANDSAF, TROPOMI
- Authentication: Bring Your Own Key (BYOK) Credentials Manager
"""

from __future__ import annotations

import importlib
from typing import Any, List

from .base import (
    AFRICA_BBOX,
    KENYA_BBOX,
    PAN_AFRICA_BBOX,
    BaseExtractor,
    SpatialBoundingBox,
    TemporalRange,
)
from .credentials import (
    CredentialsError,
    CredentialsManager,
    get_cds_api_key,
    get_credentials,
    parse_date,
    parse_date_range,
)

__all__: List[str] = [
    # Base and Credentials
    "BaseExtractor",
    "SpatialBoundingBox",
    "TemporalRange",
    "AFRICA_BBOX",
    "KENYA_BBOX",
    "PAN_AFRICA_BBOX",
    "CredentialsManager",
    "CredentialsError",
    "get_credentials",
    "get_cds_api_key",
    "parse_date",
    "parse_date_range",
    # In-Situ Ground Stations
    "TAHMOExtractor",
    "CDSInSituExtractor",
    "GHCNExtractor",
    "GHCN",
    "GHCNDaily",
    "GHCNHourly",
    "get_stations",
    "get_african_stations",
    "canonicalize_tahmo_variable",
    "tahmo_canonical_to_short",
    "canonicalize_cds_insitu_variable",
    "cds_insitu_canonical_to_raw",
    "parse_cds_insitu_dataframe",
    "parse_cds_insitu_dataset",
    # Gridded Reanalysis
    "ERA5Extractor",
    "generate_n_days_dict",
    "open_era5_grib",
    "open_era5_netcdf",
    "flatten_accum_ds",
    "resample_era5",
    "CopernicusCDSError",
    "check_grib_dependencies",
    "canonicalize_era5_variable",
    "era5_canonical_to_short",
    "harmonize_era5_dataset",
    "resolve_cds_arco_url",
    "ECMWF_CDS_ARCO_URLS",
    "DEFAULT_WAVE_FIELDS",
    "ReanalysisExtractor",
    "CAMS",
    "CAMSWrapper",
    "MERRA2",
    "MERRA2Wrapper",
    "OpenMeteo",
    "OpenMeteoWrapper",
    # Satellite
    "IMERGExtractor",
    "IMERGResult",
    "open_imerg_zarr",
    "canonicalize_imerg_variable",
    "imerg_canonical_to_raw",
    "harmonize_imerg_dataset",
    "IMERG_TRANSITION_DATE",
    "IMERG_FINAL_RUN_NAME",
    "IMERG_LATE_RUN_NAME",
    "SatelliteExtractor",
    "NASAPower",
    "NasaPower",
    "PVGIS",
    "SARAH3",
    "LANDSAF",
    "Landsaf",
    "TROPOMI",
    # Dagster Orchestration
    "defs",
    "materialize_extraction",
    "ParquetStationIOManager",
    "ZarrGriddedIOManager",
    "tahmo_station_observations",
    "era5_gridded_reanalysis",
    "imerg_satellite_precipitation",
    "cds_insitu_observations",
    "tahmo_partitions",
    "era5_partitions",
    "imerg_partitions",
    "cds_insitu_partitions",
    "decode_cds_insitu_dataframe",
    "write_station_metadata_parquets",
]

_LAZY_MODULE_MAPPING = {
    # Dagster
    "defs": ".dagster_pipeline",
    "materialize_extraction": ".dagster_pipeline",
    "ParquetStationIOManager": ".dagster_pipeline",
    "ZarrGriddedIOManager": ".dagster_pipeline",
    "tahmo_station_observations": ".dagster_pipeline",
    "era5_gridded_reanalysis": ".dagster_pipeline",
    "imerg_satellite_precipitation": ".dagster_pipeline",
    "cds_insitu_observations": ".dagster_pipeline",
    "tahmo_partitions": ".dagster_pipeline",
    "era5_partitions": ".dagster_pipeline",
    "imerg_partitions": ".dagster_pipeline",
    "cds_insitu_partitions": ".dagster_pipeline",
    "decode_cds_insitu_dataframe": ".dagster_pipeline",
    "write_station_metadata_parquets": ".dagster_pipeline",
    # In-situ
    "TAHMOExtractor": ".tahmo",
    "canonicalize_tahmo_variable": ".tahmo",
    "tahmo_canonical_to_short": ".tahmo",
    "CDSInSituExtractor": ".cds_insitu",
    "canonicalize_cds_insitu_variable": ".cds_insitu",
    "cds_insitu_canonical_to_raw": ".cds_insitu",
    "parse_cds_insitu_dataframe": ".cds_insitu",
    "parse_cds_insitu_dataset": ".cds_insitu",
    "GHCNExtractor": ".ghcnd",
    "GHCN": ".ghcnd",
    "GHCNDaily": ".ghcnd",
    "GHCNHourly": ".ghcnd",
    "get_stations": ".ghcnd",
    "get_african_stations": ".ghcnd",
    # ERA5
    "ERA5Extractor": ".era5",
    "generate_n_days_dict": ".era5",
    "open_era5_grib": ".era5",
    "open_era5_netcdf": ".era5",
    "flatten_accum_ds": ".era5",
    "resample_era5": ".era5",
    "CopernicusCDSError": ".era5",
    "check_grib_dependencies": ".era5",
    "canonicalize_era5_variable": ".era5",
    "era5_canonical_to_short": ".era5",
    "harmonize_era5_dataset": ".era5",
    "resolve_cds_arco_url": ".era5",
    "ECMWF_CDS_ARCO_URLS": ".era5",
    "DEFAULT_WAVE_FIELDS": ".era5",
    # Reanalysis
    "ReanalysisExtractor": ".reanalysis",
    "CAMS": ".reanalysis",
    "CAMSWrapper": ".reanalysis",
    "MERRA2": ".reanalysis",
    "MERRA2Wrapper": ".reanalysis",
    "OpenMeteo": ".reanalysis",
    "OpenMeteoWrapper": ".reanalysis",
    # Satellite
    "IMERGExtractor": ".imerg",
    "IMERGResult": ".imerg",
    "open_imerg_zarr": ".imerg",
    "canonicalize_imerg_variable": ".imerg",
    "imerg_canonical_to_raw": ".imerg",
    "harmonize_imerg_dataset": ".imerg",
    "IMERG_TRANSITION_DATE": ".imerg",
    "IMERG_FINAL_RUN_NAME": ".imerg",
    "IMERG_LATE_RUN_NAME": ".imerg",
    "SatelliteExtractor": ".satellite",
    "NASAPower": ".satellite",
    "NasaPower": ".satellite",
    "PVGIS": ".satellite",
    "SARAH3": ".satellite",
    "LANDSAF": ".satellite",
    "Landsaf": ".satellite",
    "TROPOMI": ".satellite",
}


def __getattr__(name: str) -> Any:
    """Lazy and resilient dynamic module import handler."""
    if name in _LAZY_MODULE_MAPPING:
        mod_name = _LAZY_MODULE_MAPPING[name]
        try:
            mod = importlib.import_module(mod_name, package=__package__)
            return getattr(mod, name)
        except ImportError as err:
            raise ImportError(
                f"Failed to import '{name}' from '{mod_name}'. "
                f"Verify that optional dependencies are installed. Underlying error: {err}"
            ) from err
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
