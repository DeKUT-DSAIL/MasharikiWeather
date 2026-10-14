"""
Satellite Data API Wrappers for MasharikiWeather
================================================
Satellite solar irradiance, geostationary observations, and atmospheric monitoring:
- NASA POWER (CERES SYN1deg satellite solar and meteorological archives)
- PVGIS (EU JRC satellite databases: SARAH-2, SARAH-3, ERA5, PV simulation)
- SARAH-3 (EUMETSAT CM SAF satellite solar irradiance)
- LANDSAF (EUMETSAT LSA SAF MSG SEVIRI 30-min DSSF solar radiation, LST, vegetation)
- TROPOMI (Copernicus Sentinel-5P Level-2 atmospheric trace gases and aerosols)
"""

from .nasa_power import (
    NASAPower,
    query_nasa_power,
    list_parameters as list_nasa_power_parameters,
    search_parameters as search_nasa_power_parameters,
    get_latest_available_dates as get_nasa_power_latest_dates,
    get_parameter_latency as get_nasa_power_parameter_latency,
)
from .pvgis import (
    PVGIS,
    query_pvgis,
    simulate_pvgis_pv,
    query_pvgis_tmy,
    search_parameters as search_pvgis_parameters,
    list_databases as list_pvgis_databases,
)
from .sarah3 import (
    SARAH3,
    query_sarah3,
    search_sarah3_parameters,
    list_sarah3_parameters,
    get_sarah3_metadata,
)
from .landsaf import (
    LANDSAF,
    query_landsaf,
    search_landsaf_parameters,
    list_landsaf_parameters,
    get_landsaf_metadata,
    list_landsaf_products,
)
from .tropomi import (
    TROPOMI,
    query_tropomi,
    search_tropomi_parameters,
    list_tropomi_parameters,
    get_tropomi_metadata,
)

__all__ = [
    "NASAPower",
    "query_nasa_power",
    "list_nasa_power_parameters",
    "search_nasa_power_parameters",
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
]
