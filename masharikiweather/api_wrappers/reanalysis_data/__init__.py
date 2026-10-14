"""
Reanalysis Data API Wrappers for MasharikiWeather
=================================================
Atmospheric reanalysis, global aerosols, and high-resolution numerical weather prediction:
- CAMS (Copernicus Atmosphere Monitoring Service: AOD, dust, PM2.5/PM10, air quality forecast)
- MERRA-2 (NASA GMAO 2D hourly aerosol speciations, SWGDN radiation, surface meteorology)
- Open-Meteo (High-resolution multi-model NWP, historical archive, and soil moisture/temp)
"""

__all__ = []

# --- CAMS ---
try:
    from .cams import (
        CAMS,
        query_cams,
        query_cams_aerosols,
        query_cams_air_quality,
        query_cams_forecast,
        classify_aqi,
        search_cams_parameters,
        list_cams_parameters,
        get_cams_metadata,
    )
    __all__.extend([
        "CAMS",
        "query_cams",
        "query_cams_aerosols",
        "query_cams_air_quality",
        "query_cams_forecast",
        "classify_aqi",
        "search_cams_parameters",
        "list_cams_parameters",
        "get_cams_metadata",
    ])
except ImportError:
    pass

# --- MERRA-2 ---
try:
    from .merra2 import (
        MERRA2,
        query_merra2,
        search_merra2_parameters,
        list_merra2_parameters,
        get_merra2_metadata,
        list_merra2_collections,
    )
    __all__.extend([
        "MERRA2",
        "query_merra2",
        "search_merra2_parameters",
        "list_merra2_parameters",
        "get_merra2_metadata",
        "list_merra2_collections",
    ])
except ImportError:
    pass

# --- Open-Meteo ---
try:
    from .open_meteo import (
        OpenMeteo,
        query_open_meteo,
        query_open_meteo_forecast,
        query_open_meteo_soil,
        decode_weather_code,
        search_open_meteo_parameters,
        list_open_meteo_parameters,
        get_open_meteo_metadata,
    )
    __all__.extend([
        "OpenMeteo",
        "query_open_meteo",
        "query_open_meteo_forecast",
        "query_open_meteo_soil",
        "decode_weather_code",
        "search_open_meteo_parameters",
        "list_open_meteo_parameters",
        "get_open_meteo_metadata",
    ])
except ImportError:
    pass
