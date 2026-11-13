"""
MasharikiWeather Unified Satellite Observation Extraction Package.

Provides unified, production-ready interfaces for:
1. NASA POWER (Solar Irradiance and Surface Meteorology)
2. EU JRC PVGIS (Photovoltaic and Solar Radiation Simulations)
3. EUMETSAT CM SAF SARAH-3 (Surface Solar Radiation Data Set - Heliosat)
4. EUMETSAT LSA SAF / LANDSAF (Land Surface Analysis, DSSF, LST, Albedo)
5. Copernicus Sentinel-5P TROPOMI (Tropospheric Atmospheric Composition & Trace Gases)
"""

from __future__ import annotations

from .common import (
    DEFAULT_CACHE_DIR,
    REGIONAL_CENTROIDS,
    _BaseDatasetWrapper,
    _normalize_locations,
    _parse_date,
    _safe_request,
)
from .extractor import SatelliteExtractor
from .landsaf import (
    LANDSAF,
    LANDSAF_PRODUCTS,
    LANDSAFWrapper,
    Landsaf,
    LandsafWrapper,
    get_landsaf_data,
)
from .nasa_power import (
    NASA_POWER_CATALOG,
    NASA_POWER_PARAMETERS,
    NASAPower,
    NASAPowerWrapper,
    NasaPower,
    NasaPowerWrapper,
    get_nasa_power_data,
)
from .pvgis import (
    PVGIS,
    PVGIS_DATABASES,
    PVGIS_RADIATION_VARS,
    PVGISWrapper,
    get_pvgis_data,
)
from .sarah3 import (
    SARAH3,
    SARAH3_PARAMETERS,
    SARAH3Wrapper,
    get_sarah3_data,
)
from .tropomi import (
    TROPOMI,
    TROPOMI_PRODUCTS,
    TROPOMIWrapper,
    get_tropomi_data,
)

__all__ = [
    'NASAPower',
    'NasaPower',
    'NASAPowerWrapper',
    'NasaPowerWrapper',
    'get_nasa_power_data',
    'NASA_POWER_PARAMETERS',
    'PVGIS',
    'PVGISWrapper',
    'get_pvgis_data',
    'SARAH3',
    'SARAH3Wrapper',
    'get_sarah3_data',
    'SARAH3_PARAMETERS',
    'LANDSAF',
    'Landsaf',
    'LANDSAFWrapper',
    'LandsafWrapper',
    'get_landsaf_data',
    'LANDSAF_PRODUCTS',
    'TROPOMI',
    'TROPOMIWrapper',
    'get_tropomi_data',
    'TROPOMI_PRODUCTS',
    'SatelliteExtractor',
]
