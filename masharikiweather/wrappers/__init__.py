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

from .nasa_power import (
    NASAPower,
    get_latest_available_dates as get_nasa_power_latest_dates,
    get_parameter_latency as get_nasa_power_parameter_latency,
    list_parameters as list_nasa_power_parameters,
    query_nasa_power,
    search_parameters as search_nasa_power_parameters,
)
from .pvgis import (
    PVGIS,
    list_databases as list_pvgis_databases,
    query_pvgis,
    query_pvgis_tmy,
    search_parameters as search_pvgis_parameters,
    simulate_pvgis_pv,
)
from .sarah3 import (
    SARAH3,
    get_sarah3_metadata,
    list_sarah3_parameters,
    query_sarah3,
    search_sarah3_parameters,
)
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
from .merra2 import (
    MERRA2,
    get_merra2_metadata,
    list_merra2_collections,
    list_merra2_parameters,
    query_merra2,
    search_merra2_parameters,
)
from .landsaf import (
    LANDSAF,
    get_landsaf_metadata,
    list_landsaf_parameters,
    list_landsaf_products,
    query_landsaf,
    search_landsaf_parameters,
)
from .tropomi import (
    TROPOMI,
    get_tropomi_metadata,
    list_tropomi_parameters,
    query_tropomi,
    search_tropomi_parameters,
)

__all__ = [
    "NASAPower",
    "query_nasa_power",
    "search_nasa_power_parameters",
    "list_nasa_power_parameters",
    "get_nasa_power_latest_dates",
    "get_nasa_power_parameter_latency",
    "PVGIS",
    "query_pvgis",
    "simulate_pvgis_pv",
    "query_pvgis_tmy",
    "search_pvgis_parameters",
    "list_pvgis_databases",
    "SARAH3",
    "query_sarah3",
    "search_sarah3_parameters",
    "list_sarah3_parameters",
    "get_sarah3_metadata",
    "CAMS",
    "query_cams",
    "query_cams_aerosols",
    "query_cams_air_quality",
    "query_cams_forecast",
    "search_cams_parameters",
    "list_cams_parameters",
    "classify_aqi",
    "get_cams_metadata",
    "OpenMeteo",
    "query_open_meteo",
    "query_open_meteo_forecast",
    "query_open_meteo_soil",
    "search_open_meteo_parameters",
    "list_open_meteo_parameters",
    "decode_weather_code",
    "get_open_meteo_metadata",
    "MERRA2",
    "query_merra2",
    "search_merra2_parameters",
    "list_merra2_parameters",
    "get_merra2_metadata",
    "list_merra2_collections",
    "LANDSAF",
    "query_landsaf",
    "search_landsaf_parameters",
    "list_landsaf_parameters",
    "get_landsaf_metadata",
    "list_landsaf_products",
    "TROPOMI",
    "query_tropomi",
    "search_tropomi_parameters",
    "list_tropomi_parameters",
    "get_tropomi_metadata",
    "query_dataset",
]


def query_dataset(
    source: str,
    locations: Optional[Union[Dict[str, float], Tuple[float, float], List[Any], str]] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    region: Optional[str] = None,
    variables: Optional[List[str]] = None,
    start_date: Optional[Union[str, Any]] = None,
    end_date: Optional[Union[str, Any]] = None,
    format: str = "wide",
    **kwargs,
) -> pd.DataFrame:
    """
    Unified query dispatcher for any satellite, reanalysis, or atmospheric dataset.

    Parameters
    ----------
    source : str
        Target dataset: 'nasa_power', 'pvgis', 'sarah3', 'cams', 'merra2', 'landsaf', 'tropomi'.
    locations : dict, tuple, list, or str, optional
        Target coordinate or named location.
    lat, lon : float, optional
        Direct latitude and longitude coordinates.
    region : str, optional
        Preset region ('kenya', 'east_africa', 'west_africa', 'africa', 'europe', etc.).
    variables : list of str, optional
        Target variables.
    start_date, end_date : str, optional
        Temporal window ('YYYY-MM-DD').
    format : str, default 'wide'
        'wide' (Date x Location matrix) or 'long' (standard spatiotemporal table).
    """
    s_clean = source.strip().lower().replace("-", "_").replace(" ", "_")

    if s_clean in ("nasa_power", "power"):
        return query_nasa_power(
            locations=locations, lat=lat, lon=lon, region=region,
            variables=variables, start_date=start_date, end_date=end_date,
            format=format, **kwargs,
        )
    elif s_clean in ("pvgis", "pvgis_sarah2", "pvgis_sarah3"):
        return query_pvgis(
            locations=locations, lat=lat, lon=lon, region=region,
            variables=variables, start_date=start_date, end_date=end_date,
            format=format, **kwargs,
        )
    elif s_clean in ("sarah3", "sarah_3", "cmsaf"):
        return query_sarah3(
            locations=locations, lat=lat, lon=lon, region=region,
            variables=variables, start_date=start_date, end_date=end_date,
            format=format, **kwargs,
        )
    elif s_clean in ("cams", "cams_aerosols", "cams_aq", "copernicus_atmosphere"):
        return query_cams(
            locations=locations, lat=lat, lon=lon, region=region,
            variables=variables, start_date=start_date, end_date=end_date,
            format=format, **kwargs,
        )
    elif s_clean in ("open_meteo", "openmeteo", "weather", "forecast", "ecmwf_ifs"):
        return query_open_meteo(
            locations=locations, lat=lat, lon=lon, region=region,
            variables=variables, start_date=start_date, end_date=end_date,
            format=format, **kwargs,
        )
    elif s_clean in ("merra2", "merra_2"):
        return query_merra2(
            locations=locations, lat=lat, lon=lon, region=region,
            variables=variables, start_date=start_date, end_date=end_date,
            format=format, **kwargs,
        )
    elif s_clean in ("landsaf", "lsa_saf", "mdssftd"):
        return query_landsaf(
            locations=locations, lat=lat, lon=lon, region=region,
            variables=variables, start_date=start_date, end_date=end_date,
            format=format, **kwargs,
        )
    elif s_clean in ("tropomi", "sentinel5p", "s5p"):
        return query_tropomi(
            locations=locations, lat=lat, lon=lon, region=region,
            variables=variables, start_date=start_date, end_date=end_date,
            format=format, **kwargs,
        )
    else:
        available = ["nasa_power", "pvgis", "sarah3", "cams", "merra2", "landsaf", "tropomi"]
        raise ValueError(f"Unknown dataset source '{source}'. Available wrappers: {available}")
