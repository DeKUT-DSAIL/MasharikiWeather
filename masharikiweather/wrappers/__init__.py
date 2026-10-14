"""
MasharikiWeather Multi-Source Climate, Satellite & Reanalysis Wrappers
=======================================================================
Centralized access layer for:
- NASA POWER (Hourly & Daily CERES solar radiation and meteorology)
- PVGIS (EU JRC solar database: SARAH-2, SARAH-3, ERA5)
- SARAH-3 (EUMETSAT CM SAF satellite solar irradiance)
- CAMS (Copernicus Atmosphere Monitoring Service: AOD, Dust, McClear, Solar)
- MERRA-2 (NASA GMAO 2D hourly aerosols and radiation reanalysis)
- LANDSAF (EUMETSAT LSA SAF MDSSFTD downward surface shortwave flux)
- TROPOMI (Copernicus Sentinel-5P UV aerosol index and cloud products)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

__all__ = []

# --- NASA POWER ---
try:
    from .nasa_power import (
        NASAPower,
        get_latest_available_dates as get_nasa_power_latest_dates,
        get_parameter_latency as get_nasa_power_parameter_latency,
        list_parameters as list_nasa_power_parameters,
        query_nasa_power,
        search_parameters as search_nasa_power_parameters,
    )
    __all__.extend([
        "NASAPower",
        "query_nasa_power",
        "search_nasa_power_parameters",
        "list_nasa_power_parameters",
        "get_nasa_power_latest_dates",
        "get_nasa_power_parameter_latency",
    ])
except ImportError:
    pass

# --- PVGIS ---
try:
    from .pvgis import (
        PVGIS,
        list_databases as list_pvgis_databases,
        query_pvgis,
        query_pvgis_tmy,
        search_parameters as search_pvgis_parameters,
        simulate_pvgis_pv,
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
    from .sarah3 import (
        SARAH3,
        get_sarah3_metadata,
        list_sarah3_parameters,
        query_sarah3,
        search_sarah3_parameters,
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

# --- CAMS ---
try:
    from .cams import (
        CAMS,
        classify_aqi,
        get_cams_metadata,
        list_cams_parameters,
        query_cams,
        query_cams_aerosols,
        query_cams_air_quality,
        query_cams_forecast,
        search_cams_parameters,
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

# --- Open-Meteo ---
try:
    from .open_meteo import (
        OpenMeteo,
        decode_weather_code,
        get_open_meteo_metadata,
        list_open_meteo_parameters,
        query_open_meteo,
        query_open_meteo_forecast,
        query_open_meteo_soil,
        search_open_meteo_parameters,
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

# --- MERRA-2 ---
try:
    from .merra2 import (
        MERRA2,
        get_merra2_metadata,
        list_merra2_collections,
        list_merra2_parameters,
        query_merra2,
        search_merra2_parameters,
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

# --- LANDSAF ---
try:
    from .landsaf import (
        LANDSAF,
        get_landsaf_metadata,
        list_landsaf_parameters,
        list_landsaf_products,
        query_landsaf,
        search_landsaf_parameters,
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
    from .tropomi import (
        TROPOMI,
        get_tropomi_metadata,
        list_tropomi_parameters,
        query_tropomi,
        search_tropomi_parameters,
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
    else:
        raise ValueError(f"Unknown data source '{source}'. Available: nasa_power, pvgis, sarah3, cams, open_meteo, merra2, landsaf, tropomi")

__all__.append("query_dataset")
