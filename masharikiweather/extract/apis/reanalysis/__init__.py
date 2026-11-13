"""
MasharikiWeather Unified Reanalysis Extraction Package.

Provides unified, production-ready interfaces for:
1. Copernicus Atmosphere Monitoring Service (CAMS) - atmospheric composition, AOD, dust, PM2.5/10.
2. NASA Modern-Era Retrospective analysis for Research and Applications, Version 2 (MERRA-2) - global aerosols and fluxes.
3. Open-Meteo High-Resolution Global Weather & Historical Reanalysis - ECMWF IFS, GFS, ICON, and ERA5.
"""

from __future__ import annotations

from .cams import (
    AQI_CATEGORIES,
    CAMS,
    CAMS_AQ_LEXICON,
    CAMS_PARAMETERS,
    CAMSWrapper,
    DEFAULT_CAMS_AQ_VARS,
    classify_aqi,
    get_cams_data,
    get_cams_metadata,
    list_cams_parameters,
    query_cams,
    query_cams_aerosols,
    query_cams_air_quality,
    query_cams_forecast,
    search_cams_parameters,
)
from .common import (
    DEFAULT_CACHE_DIR,
    REGIONAL_CENTROIDS,
    _BaseDatasetWrapper,
    _normalize_locations,
    _parse_date,
    _safe_request,
)
from .extractor import ReanalysisExtractor
from .merra2 import (
    MERRA2,
    MERRA2_COLLECTIONS,
    MERRA2_LEXICON,
    MERRA2_PARAMETERS,
    MERRA2_VARIABLES,
    MERRA2Wrapper,
    get_merra2_data,
    get_merra2_metadata,
    list_merra2_collections,
    list_merra2_parameters,
    query_merra2,
    search_merra2_parameters,
)
from .open_meteo import (
    DEFAULT_OPENMETEO_VARS,
    OPEN_METEO_MODELS,
    OPENMETEO_LEXICON,
    OPENMETEO_PARAMETERS,
    OpenMeteo,
    OpenMeteoWrapper,
    decode_weather_code,
    get_open_meteo_data,
    get_open_meteo_metadata,
    list_open_meteo_parameters,
    query_open_meteo,
    query_open_meteo_forecast,
    query_open_meteo_soil,
    search_open_meteo_parameters,
    WMO_WEATHER_CODES,
)

__all__ = [
    # CAMS
    "CAMS",
    "CAMSWrapper",
    "get_cams_data",
    "CAMS_PARAMETERS",
    "CAMS_AQ_LEXICON",
    "DEFAULT_CAMS_AQ_VARS",
    "AQI_CATEGORIES",
    "classify_aqi",
    "search_cams_parameters",
    "list_cams_parameters",
    "get_cams_metadata",
    "query_cams",
    "query_cams_aerosols",
    "query_cams_air_quality",
    "query_cams_forecast",
    # MERRA-2
    "MERRA2",
    "MERRA2Wrapper",
    "get_merra2_data",
    "MERRA2_COLLECTIONS",
    "MERRA2_PARAMETERS",
    "MERRA2_VARIABLES",
    "MERRA2_LEXICON",
    "search_merra2_parameters",
    "list_merra2_parameters",
    "get_merra2_metadata",
    "list_merra2_collections",
    "query_merra2",
    # Open-Meteo
    "OpenMeteo",
    "OpenMeteoWrapper",
    "get_open_meteo_data",
    "OPEN_METEO_MODELS",
    "WMO_WEATHER_CODES",
    "decode_weather_code",
    "OPENMETEO_PARAMETERS",
    "OPENMETEO_LEXICON",
    "DEFAULT_OPENMETEO_VARS",
    "search_open_meteo_parameters",
    "list_open_meteo_parameters",
    "get_open_meteo_metadata",
    "query_open_meteo",
    "query_open_meteo_forecast",
    "query_open_meteo_soil",
    # Unified Extractor & Common
    "ReanalysisExtractor",
    "REGIONAL_CENTROIDS",
    "DEFAULT_CACHE_DIR",
    "_BaseDatasetWrapper",
    "_safe_request",
    "_parse_date",
    "_normalize_locations",
]
