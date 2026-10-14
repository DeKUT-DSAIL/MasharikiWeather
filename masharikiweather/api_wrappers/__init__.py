"""
MasharikiWeather API Wrappers Interface
========================================
Centralized access layer for:
- Weather Station Data (NOAA GHCN-Daily & GHCN-Hourly)
- Satellite Missions (NASA POWER, PVGIS, SARAH-3, LANDSAF, TROPOMI)
- Atmospheric Reanalyses (CAMS, MERRA-2, Open-Meteo)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

__all__ = []

# --- Weather Station Data ---
try:
    from .weather_station_data import (
        GHCN,
        GHCNDaily,
        GHCNHourly,
        GHCNDailyLexicon,
        GHCNHourlyLexicon,
        DEFAULT_KENYA_BOUNDS,
        AFRICAN_REGIONS,
        CONTINENT_REGIONS,
        GLOBAL_REGIONS,
        haversine_distance,
        align_to_reference_grid,
        find_nearest_stations,
        find_nearest_grid_point,
        to_wide,
        get_stations,
        get_african_stations,
        query_ghcn_daily,
        query_ghcn_hourly,
        check_ghcn_availability,
        check_station_coverage,
        check_variable_activeness,
        get_stations_variable_activeness,
        get_variable_lifecycle,
        calculate_relative_humidity,
        get_station_series,
    )
    __all__.extend([
        "GHCN",
        "GHCNDaily",
        "GHCNHourly",
        "GHCNDailyLexicon",
        "GHCNHourlyLexicon",
        "DEFAULT_KENYA_BOUNDS",
        "AFRICAN_REGIONS",
        "CONTINENT_REGIONS",
        "GLOBAL_REGIONS",
        "haversine_distance",
        "align_to_reference_grid",
        "find_nearest_stations",
        "find_nearest_grid_point",
        "to_wide",
        "get_stations",
        "get_african_stations",
        "query_ghcn_daily",
        "query_ghcn_hourly",
        "check_ghcn_availability",
        "check_station_coverage",
        "check_variable_activeness",
        "get_stations_variable_activeness",
        "get_variable_lifecycle",
        "calculate_relative_humidity",
        "get_station_series",
    ])
except ImportError:
    pass

# --- NASA POWER ---
try:
    from .satellite_data import (
        NASAPower,
        query_nasa_power,
        list_nasa_power_parameters,
        search_nasa_power_parameters,
        get_nasa_power_latest_dates,
        get_nasa_power_parameter_latency,
    )
    __all__.extend([
        "NASAPower",
        "query_nasa_power",
        "list_nasa_power_parameters",
        "search_nasa_power_parameters",
        "get_nasa_power_latest_dates",
        "get_nasa_power_parameter_latency",
    ])
except ImportError:
    pass

# --- PVGIS ---
try:
    from .satellite_data import (
        PVGIS,
        query_pvgis,
        simulate_pvgis_pv,
        query_pvgis_tmy,
        search_pvgis_parameters,
        list_pvgis_databases,
    )
    __all__.extend([
        "PVGIS",
        "query_pvgis",
        "simulate_pvgis_pv",
        "query_pvgis_tmy",
        "search_pvgis_parameters",
        "list_pvgis_databases",
    ])
except ImportError:
    pass

# --- SARAH-3 ---
try:
    from .satellite_data import (
        SARAH3,
        query_sarah3,
        search_sarah3_parameters,
        list_sarah3_parameters,
        get_sarah3_metadata,
    )
    __all__.extend([
        "SARAH3",
        "query_sarah3",
        "search_sarah3_parameters",
        "list_sarah3_parameters",
        "get_sarah3_metadata",
    ])
except ImportError:
    pass

# --- LANDSAF ---
try:
    from .satellite_data import (
        LANDSAF,
        query_landsaf,
        search_landsaf_parameters,
        list_landsaf_parameters,
        get_landsaf_metadata,
        list_landsaf_products,
    )
    __all__.extend([
        "LANDSAF",
        "query_landsaf",
        "search_landsaf_parameters",
        "list_landsaf_parameters",
        "get_landsaf_metadata",
        "list_landsaf_products",
    ])
except ImportError:
    pass

# --- TROPOMI ---
try:
    from .satellite_data import (
        TROPOMI,
        query_tropomi,
        search_tropomi_parameters,
        list_tropomi_parameters,
        get_tropomi_metadata,
    )
    __all__.extend([
        "TROPOMI",
        "query_tropomi",
        "search_tropomi_parameters",
        "list_tropomi_parameters",
        "get_tropomi_metadata",
    ])
except ImportError:
    pass

# --- CAMS ---
try:
    from .reanalysis_data import (
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
    from .reanalysis_data import (
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
    from .reanalysis_data import (
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

# --- Unified Dispatcher ---
def query_dataset(
    source: str,
    *args: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Universal dispatcher querying any supported satellite, reanalysis or weather dataset.
    """
    s = str(source).lower().replace("-", "_").strip()
    if s in ("nasa", "power", "nasa_power"):
        if "query_nasa_power" not in globals():
            raise ImportError("NASA POWER wrapper is not available.")
        return query_nasa_power(*args, **kwargs)
    elif s in ("pvgis", "jrc"):
        if "query_pvgis" not in globals():
            raise ImportError("PVGIS wrapper is not available.")
        return query_pvgis(*args, **kwargs)
    elif s in ("sarah", "sarah3", "cmsaf"):
        if "query_sarah3" not in globals():
            raise ImportError("SARAH-3 wrapper is not available.")
        return query_sarah3(*args, **kwargs)
    elif s in ("cams", "copernicus_atmosphere"):
        if "query_cams" not in globals():
            raise ImportError("CAMS wrapper is not available.")
        return query_cams(*args, **kwargs)
    elif s in ("open_meteo", "meteo", "openmeteo"):
        if "query_open_meteo" not in globals():
            raise ImportError("Open-Meteo wrapper is not available.")
        return query_open_meteo(*args, **kwargs)
    elif s in ("merra2", "merra"):
        if "query_merra2" not in globals():
            raise ImportError("MERRA-2 wrapper is not available.")
        return query_merra2(*args, **kwargs)
    elif s in ("landsaf", "lsa_saf"):
        if "query_landsaf" not in globals():
            raise ImportError("LANDSAF wrapper is not available.")
        return query_landsaf(*args, **kwargs)
    elif s in ("tropomi", "sentinel5p", "s5p"):
        if "query_tropomi" not in globals():
            raise ImportError("TROPOMI wrapper is not available.")
        return query_tropomi(*args, **kwargs)
    elif s in ("ghcn", "ghcnd", "ghcnh", "station", "weather_station"):
        if "query_ghcn_daily" not in globals():
            raise ImportError("GHCN wrapper is not available.")
        return query_ghcn_daily(*args, **kwargs)
    else:
        raise ValueError(
            f"Unknown data source '{source}'. Available: ghcn, nasa_power, pvgis, sarah3, cams, open_meteo, merra2, landsaf, tropomi"
        )

__all__.append("query_dataset")
