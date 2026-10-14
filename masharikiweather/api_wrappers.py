"""
MasharikiWeather API Wrappers Interface
========================================
Unified access point for all weather station, satellite, and reanalysis data wrappers.
"""

from .ghcn_apiwrapper import (
    GHCN,
    GHCNDaily,
    GHCNHourly,
    query_ghcn_daily,
    query_ghcn_hourly,
    get_stations,
    get_african_stations,
    check_station_coverage,
    find_nearest_stations,
)

from .wrappers import (
    # Satellite
    NASAPower,
    query_nasa_power,
    PVGIS,
    query_pvgis,
    simulate_pvgis_pv,
    query_pvgis_tmy,
    SARAH3,
    query_sarah3,
    LANDSAF,
    query_landsaf,
    TROPOMI,
    query_tropomi,
    # Reanalysis
    CAMS,
    query_cams,
    query_cams_aerosols,
    query_cams_air_quality,
    query_cams_forecast,
    classify_aqi,
    MERRA2,
    query_merra2,
    OpenMeteo,
    query_open_meteo,
    query_open_meteo_forecast,
    query_open_meteo_soil,
    # Dispatcher
    query_dataset,
)

__all__ = [
    "GHCN",
    "GHCNDaily",
    "GHCNHourly",
    "query_ghcn_daily",
    "query_ghcn_hourly",
    "get_stations",
    "get_african_stations",
    "check_station_coverage",
    "find_nearest_stations",
    "NASAPower",
    "query_nasa_power",
    "PVGIS",
    "query_pvgis",
    "simulate_pvgis_pv",
    "query_pvgis_tmy",
    "SARAH3",
    "query_sarah3",
    "LANDSAF",
    "query_landsaf",
    "TROPOMI",
    "query_tropomi",
    "CAMS",
    "query_cams",
    "query_cams_aerosols",
    "query_cams_air_quality",
    "query_cams_forecast",
    "classify_aqi",
    "MERRA2",
    "query_merra2",
    "OpenMeteo",
    "query_open_meteo",
    "query_open_meteo_forecast",
    "query_open_meteo_soil",
    "query_dataset",
]
