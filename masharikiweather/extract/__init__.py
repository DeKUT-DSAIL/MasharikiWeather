"""
MasharikiWeather Extraction Package.

Unified data extraction engines and API wrappers across meteorological,
satellite, reanalysis, and static spatial priors for Africa.
"""

from __future__ import annotations

from . import apis
from .apis import (
    AFRICA_BBOX,
    KENYA_BBOX,
    PAN_AFRICA_BBOX,
    CDSInSituExtractor,
    CredentialsError,
    CredentialsManager,
    ERA5Extractor,
    GHCNExtractor,
    IMERGExtractor,
    ReanalysisExtractor,
    SatelliteExtractor,
    SpatialBoundingBox,
    TAHMOExtractor,
    TemporalRange,
    get_credentials,
    defs,
    materialize_extraction,
    ParquetStationIOManager,
    ZarrGriddedIOManager,
    tahmo_station_observations,
    era5_gridded_reanalysis,
    imerg_satellite_precipitation,
    cds_insitu_observations,
)

__all__ = [
    "apis",
    "CredentialsManager",
    "CredentialsError",
    "get_credentials",
    "SpatialBoundingBox",
    "TemporalRange",
    "AFRICA_BBOX",
    "KENYA_BBOX",
    "PAN_AFRICA_BBOX",
    "ERA5Extractor",
    "CDSInSituExtractor",
    "TAHMOExtractor",
    "IMERGExtractor",
    "GHCNExtractor",
    "ReanalysisExtractor",
    "SatelliteExtractor",
    "defs",
    "materialize_extraction",
    "ParquetStationIOManager",
    "ZarrGriddedIOManager",
    "tahmo_station_observations",
    "era5_gridded_reanalysis",
    "imerg_satellite_precipitation",
    "cds_insitu_observations",
]
