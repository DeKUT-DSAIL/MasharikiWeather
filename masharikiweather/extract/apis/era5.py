"""
ERA5 Atmospheric and Surface Reanalysis Extraction Engine.

Extracts high-resolution atmospheric reanalysis from ECMWF ERA5 with dual-engine architecture:
1. Primary Batch Engine: Cloud-native Google Cloud ARCO ERA5
   (gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3) via lazy xarray.
2. Resilient Fallback Engine: Copernicus Climate Data Store (CDS API) with
   automated temporal chunking (5 to 10-day intervals) and error interception.

All outputs are saved as standardized, CF-compliant Zarr groups on local disk.
"""

from __future__ import annotations

import calendar
import gc
import logging
import os
import shutil
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import (
    AFRICA_BBOX,
    BaseExtractor,
    SpatialBoundingBox,
    TemporalRange,
)
from .credentials import CredentialsError, CredentialsManager, get_cds_api_key, parse_date, parse_date_range
from .resilience import (
    ExtractionEstimate,
    append_to_zarr,
    compute_missing_zarr_windows,
    inspect_zarr_coverage,
    make_progress_bar,
)

logger = logging.getLogger("MasharikiWeather.ERA5")

# Canonical ERA5 URLs across Google Cloud and ECMWF CDS ARCO
DEFAULT_GOOGLE_ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARCO_ERA5_ZARR_URL = DEFAULT_GOOGLE_ARCO_URL

# ECMWF CDS ARCO Zarr Stores Mapping Matrix
# Exact endpoints per CADL specification for Surface (sfc), Waves (wav), and Pressure Levels (pl)
ECMWF_CDS_ARCO_URLS: Dict[str, Dict[str, str]] = {
    "sfc": {
        "geo": "https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr",
        "time": "https://arco.datastores.ecmwf.int/cadl-arco-time-002/arco/reanalysis_era5_single_levels/sfc/timeChunked.zarr",
    },
    "wav": {
        "geo": "https://arco.datastores.ecmwf.int/cadl-arco-geo-003/arco/reanalysis_era5_single_levels/wav/geoChunked.zarr",
        "time": "https://arco.datastores.ecmwf.int/cadl-arco-time-003/arco/reanalysis_era5_single_levels/wav/timeChunked.zarr",
    },
    "pl": {
        "geo": "https://arco.datastores.ecmwf.int/cadl-arco-geo-048/arco/reanalysis_era5_pressure_levels/pl/geoChunked.zarr",
        "time": "https://arco.datastores.ecmwf.int/cadl-arco-time-048/arco/reanalysis_era5_pressure_levels/pl/timeChunked.zarr",
    },
}

DEFAULT_CDS_ARCO_SFC_GEO_URL = ECMWF_CDS_ARCO_URLS["sfc"]["geo"]
DEFAULT_CDS_ARCO_SFC_TIME_URL = ECMWF_CDS_ARCO_URLS["sfc"]["time"]
DEFAULT_CDS_ARCO_WAV_GEO_URL = ECMWF_CDS_ARCO_URLS["wav"]["geo"]
DEFAULT_CDS_ARCO_WAV_TIME_URL = ECMWF_CDS_ARCO_URLS["wav"]["time"]
DEFAULT_CDS_ARCO_PL_GEO_URL = ECMWF_CDS_ARCO_URLS["pl"]["geo"]
DEFAULT_CDS_ARCO_PL_TIME_URL = ECMWF_CDS_ARCO_URLS["pl"]["time"]

DEFAULT_CDS_ARCO_PL_URL = DEFAULT_CDS_ARCO_PL_GEO_URL
DEFAULT_CDS_ARCO_SFC_URL = DEFAULT_CDS_ARCO_SFC_GEO_URL

# Standard ERA5 variables
DEFAULT_SINGLE_LEVEL_FIELDS = [
    # Surface Winds and Temperature
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    # Pressure and Water Column
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_column_water_vapour",
    "total_column_water",
    # Precipitation and Convection
    "total_precipitation",
    "convective_precipitation",
    "large_scale_precipitation",
    "convective_available_potential_energy",
    "convective_inhibition",
    # Radiation and Clouds
    "toa_incident_solar_radiation",
    "surface_solar_radiation_downwards",
    "total_cloud_cover",
    "high_cloud_cover",
    "medium_cloud_cover",
    "low_cloud_cover",
    # Ocean Boundaries
    "sea_surface_temperature",
    "sea_ice_cover",
    # Soil and Land State
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "snow_depth",
    # Invariant and Surface Static
    "land_sea_mask",
    "soil_type",
    "geopotential",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
    # Extended Fluxes
    "potential_evaporation",
    "surface_net_solar_radiation",
    "top_net_thermal_radiation",
    "top_net_solar_radiation",
    "evaporation",
    "snowfall",
    "surface_net_thermal_radiation",
    "surface_thermal_radiation_downwards",
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
]

DEFAULT_PRESSURE_LEVEL_FIELDS = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "relative_humidity",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
]

DEFAULT_PRESSURE_LEVELS = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 775, 800, 850, 925, 1000
]

DEFAULT_WAVE_FIELDS = [
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_direction",
    "mean_wave_period",
]

# ERA5 Single-Level Variables Categorization for GRIB stepType differentiation
ERA5_INSTANTANEOUS_VARIABLES: Set[str] = {
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "100m_u_component_of_wind", "100m_v_component_of_wind",
    "2m_temperature", "2m_dewpoint_temperature",
    "mean_sea_level_pressure", "surface_pressure",
    "total_column_water_vapour", "total_column_water",
    "total_cloud_cover", "high_cloud_cover", "medium_cloud_cover", "low_cloud_cover",
    "sea_surface_temperature", "sea_ice_cover",
    "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
    "soil_temperature_level_1", "soil_temperature_level_2", "snow_depth",
    "land_sea_mask", "soil_type", "geopotential",
    "leaf_area_index_high_vegetation", "leaf_area_index_low_vegetation",
    "convective_available_potential_energy", "convective_inhibition",
}

ERA5_ACCUMULATED_VARIABLES: Set[str] = {
    "total_precipitation", "convective_precipitation", "large_scale_precipitation",
    "toa_incident_solar_radiation", "surface_solar_radiation_downwards",
    "potential_evaporation", "surface_net_solar_radiation",
    "top_net_thermal_radiation", "top_net_solar_radiation",
    "evaporation", "snowfall", "surface_net_thermal_radiation",
    "surface_thermal_radiation_downwards",
    "surface_sensible_heat_flux", "surface_latent_heat_flux",
}

# Bidirectional Cross-Source Harmonization Registry for ERA5
ERA5_CDS_SHORT_TO_CANONICAL: Dict[str, str] = {
    # Surface state variables
    "2t": "2m_temperature",
    "t2m": "2m_temperature",
    "2d": "2m_dewpoint_temperature",
    "d2m": "2m_dewpoint_temperature",
    "10u": "10m_u_component_of_wind",
    "u10": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "100u": "100m_u_component_of_wind",
    "u100": "100m_u_component_of_wind",
    "100v": "100m_v_component_of_wind",
    "v100": "100m_v_component_of_wind",
    "sp": "surface_pressure",
    "msl": "mean_sea_level_pressure",
    "tcc": "total_cloud_cover",
    "tcwv": "total_column_water_vapour",
    "cape": "convective_available_potential_energy",
    "cin": "convective_inhibition",
    "sst": "sea_surface_temperature",
    "siconc": "sea_ice_cover",
    "swvl1": "volumetric_soil_water_layer_1",
    "swvl2": "volumetric_soil_water_layer_2",
    "stl1": "soil_temperature_level_1",
    "stl2": "soil_temperature_level_2",
    "sde": "snow_depth",
    "lsm": "land_sea_mask",
    "slt": "soil_type",
    "z": "geopotential",
    "lai_hv": "leaf_area_index_high_vegetation",
    "lai_lv": "leaf_area_index_low_vegetation",
    "blh": "boundary_layer_height",
    "cbh": "cloud_base_height",
    "fdir": "total_sky_direct_solar_radiation_at_surface",
    "fg10": "10m_wind_gust_since_previous_post_processing",
    "10fg": "10m_wind_gust_since_previous_post_processing",
    "skt": "skin_temperature",
    # Accumulated / Flux variables
    "tp": "total_precipitation",
    "cp": "convective_precipitation",
    "lsp": "large_scale_precipitation",
    "ssrd": "surface_solar_radiation_downwards",
    "tisr": "toa_incident_solar_radiation",
    "strd": "surface_thermal_radiation_downwards",
    "sshf": "surface_sensible_heat_flux",
    "slhf": "surface_latent_heat_flux",
    "pev": "potential_evaporation",
    "ssr": "surface_net_solar_radiation",
    "ttr": "top_net_thermal_radiation",
    "tsr": "top_net_solar_radiation",
    "e": "evaporation",
    "sf": "snowfall",
    "str": "surface_net_thermal_radiation",
    # Pressure level variables
    "q": "specific_humidity",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "r": "relative_humidity",
    "clwc": "specific_cloud_liquid_water_content",
    "ciwc": "specific_cloud_ice_water_content",
    # Wave variables
    "swh": "significant_height_of_combined_wind_waves_and_swell",
    "mwd": "mean_wave_direction",
    "mwp": "mean_wave_period",
}

# Canonical descriptive name to standard CDS short code
ERA5_CANONICAL_TO_CDS_SHORT: Dict[str, str] = {
    "2m_temperature": "2t",
    "2m_dewpoint_temperature": "2d",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "100m_u_component_of_wind": "100u",
    "100m_v_component_of_wind": "100v",
    "total_precipitation": "tp",
    "convective_precipitation": "cp",
    "large_scale_precipitation": "lsp",
    "surface_pressure": "sp",
    "mean_sea_level_pressure": "msl",
    "surface_solar_radiation_downwards": "ssrd",
    "toa_incident_solar_radiation": "tisr",
    "total_cloud_cover": "tcc",
    "total_column_water_vapour": "tcwv",
    "convective_available_potential_energy": "cape",
    "convective_inhibition": "cin",
    "surface_thermal_radiation_downwards": "strd",
    "surface_sensible_heat_flux": "sshf",
    "surface_latent_heat_flux": "slhf",
    "sea_surface_temperature": "sst",
    "boundary_layer_height": "blh",
    "cloud_base_height": "cbh",
    "total_sky_direct_solar_radiation_at_surface": "fdir",
    "10m_wind_gust_since_previous_post_processing": "fg10",
    "skin_temperature": "skt",
    "sea_ice_cover": "siconc",
    "volumetric_soil_water_layer_1": "swvl1",
    "volumetric_soil_water_layer_2": "swvl2",
    "soil_temperature_level_1": "stl1",
    "soil_temperature_level_2": "stl2",
    "snow_depth": "sde",
    "land_sea_mask": "lsm",
    "soil_type": "slt",
    "geopotential": "z",
    "leaf_area_index_high_vegetation": "lai_hv",
    "leaf_area_index_low_vegetation": "lai_lv",
    "potential_evaporation": "pev",
    "surface_net_solar_radiation": "ssr",
    "top_net_thermal_radiation": "ttr",
    "top_net_solar_radiation": "tsr",
    "evaporation": "e",
    "snowfall": "sf",
    "surface_net_thermal_radiation": "str",
    "specific_humidity": "q",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vertical_velocity": "w",
    "relative_humidity": "r",
    "specific_cloud_liquid_water_content": "clwc",
    "specific_cloud_ice_water_content": "ciwc",
    # Wave variables
    "significant_height_of_combined_wind_waves_and_swell": "swh",
    "mean_wave_direction": "mwd",
    "mean_wave_period": "mwp",
}


def canonicalize_era5_variable(var: str) -> str:
    """
    Map an ERA5 variable identifier (short code or canonical name) to canonical descriptive name.

    Parameters
    ----------
    var : str
        Variable identifier (e.g., '2t', 'tp', '2m_temperature').

    Returns
    -------
    str
        Canonical descriptive variable name.
    """
    return ERA5_CDS_SHORT_TO_CANONICAL.get(var, var)


def era5_canonical_to_short(var: str) -> str:
    """
    Map a canonical descriptive variable name to its standard CDS short code.

    Parameters
    ----------
    var : str
        Canonical descriptive name (e.g., '2m_temperature', 'total_precipitation').

    Returns
    -------
    str
        Standard CDS short code (e.g., '2t', 'tp').
    """
    return ERA5_CANONICAL_TO_CDS_SHORT.get(var, var)


def get_era5_variable_candidates(var: str) -> List[str]:
    """
    Generate all possible identifier aliases for an ERA5 variable across canonical,
    legacy GRIB short codes, and CADL ARCO Zarr variable names.

    Mathematical docstring:
        Candidates(V) = {V, canonical(V)} U {k | ERA5_CDS_SHORT_TO_CANONICAL[k] == canonical(V)}
                        U {ERA5_CANONICAL_TO_CDS_SHORT[V]}

    Parameters
    ----------
    var : str
        Variable identifier (e.g., '2m_temperature', '2t', 't2m', 'tp').

    Returns
    -------
    List[str]
        Ordered, deduplicated list of candidate variable identifiers.
    """
    canon = canonicalize_era5_variable(var)
    candidates: List[str] = [var, canon]
    for k, v in ERA5_CDS_SHORT_TO_CANONICAL.items():
        if v == canon:
            candidates.append(k)
    if var in ERA5_CANONICAL_TO_CDS_SHORT:
        candidates.append(ERA5_CANONICAL_TO_CDS_SHORT[var])
    if canon in ERA5_CANONICAL_TO_CDS_SHORT:
        candidates.append(ERA5_CANONICAL_TO_CDS_SHORT[canon])
    return list(dict.fromkeys(candidates))


def harmonize_era5_dataset(ds: xr.Dataset, to_canonical: bool = True) -> xr.Dataset:
    """
    Bidirectionally harmonize variable names in an ERA5 xarray.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with either CDS short codes or canonical variable names.
    to_canonical : bool, default True
        If True, translates CDS short codes to canonical descriptive names.
        If False, translates canonical descriptive names back to CDS short codes.

    Returns
    -------
    xr.Dataset
        Dataset with harmonized variable names.
    """
    mapping = ERA5_CDS_SHORT_TO_CANONICAL if to_canonical else ERA5_CANONICAL_TO_CDS_SHORT
    rename_map = {k: v for k, v in mapping.items() if k in ds.data_vars and k != v}
    if rename_map:
        return ds.rename(rename_map)
    return ds


def _infer_chunking(
    bbox: Optional[SpatialBoundingBox] = None,
    start_date: Optional[Union[str, datetime, date, pd.Timestamp]] = None,
    end_date: Optional[Union[str, datetime, date, pd.Timestamp]] = None,
) -> str:
    """
    Infer optimal ECMWF CDS ARCO chunking ('geo' or 'time') from request shape.

    Heuristic:
    - Point query (lat_min == lat_max, lon_min == lon_max) or small spatial extent
      (delta_lat * delta_lon <= 25.0) -> 'geo' (time-series optimized).
    - Single timestamp (interval <= 3600s) over a wide spatial domain -> 'time' (spatial slice optimized).
    - Short window (<= 24 hours) over continental/global domain (> 100 sq deg) -> 'time'.
    - Multi-day or extended time series -> 'geo'.
    """
    if bbox is not None:
        delta_lat = abs(bbox.lat_max - bbox.lat_min)
        delta_lon = abs(bbox.lon_max - bbox.lon_min)
        spatial_extent = delta_lat * delta_lon
        is_point_or_small = (delta_lat == 0 and delta_lon == 0) or (spatial_extent <= 25.0)
    else:
        spatial_extent = 73.0 * 75.0
        is_point_or_small = False

    if start_date is not None and end_date is not None:
        s_ts = pd.to_datetime(start_date)
        e_ts = pd.to_datetime(end_date)
        duration_seconds = max(0.0, (e_ts - s_ts).total_seconds())
    elif start_date is not None:
        duration_seconds = 0.0
    else:
        duration_seconds = 86400.0 * 5.0

    is_single_time = duration_seconds <= 3600.0

    if is_point_or_small:
        return "geo"
    elif is_single_time:
        return "time"
    elif duration_seconds <= 86400.0 and spatial_extent > 100.0:
        return "time"
    else:
        return "geo"


def resolve_cds_arco_url(
    stream: Optional[str] = None,
    variables: Optional[Sequence[str]] = None,
    levels: Optional[Sequence[int]] = None,
    bbox: Optional[SpatialBoundingBox] = None,
    start_date: Optional[Union[str, datetime, date, pd.Timestamp]] = None,
    end_date: Optional[Union[str, datetime, date, pd.Timestamp]] = None,
    chunking: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Resolve ECMWF CDS ARCO Zarr URL using the query-aware routing matrix.

    The ECMWF CDS ARCO stores vary depending on whether data is accessed by
    a single location over time, or a single time stamp over a spatial area.
    This function routes to the exact target Zarr endpoint based on request shape
    and allows a manual override via `chunking="geo"` or `"time"`.

    Exact Target URLs:
    1. Single Levels (Surface - sfc):
       - Geo:  https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr
       - Time: https://arco.datastores.ecmwf.int/cadl-arco-time-002/arco/reanalysis_era5_single_levels/sfc/timeChunked.zarr
    2. Single Levels (Waves - wav):
       - Geo:  https://arco.datastores.ecmwf.int/cadl-arco-geo-003/arco/reanalysis_era5_single_levels/wav/geoChunked.zarr
       - Time: https://arco.datastores.ecmwf.int/cadl-arco-time-003/arco/reanalysis_era5_single_levels/wav/timeChunked.zarr
    3. Pressure Levels (pl):
       - Geo:  https://arco.datastores.ecmwf.int/cadl-arco-geo-048/arco/reanalysis_era5_pressure_levels/pl/geoChunked.zarr
       - Time: https://arco.datastores.ecmwf.int/cadl-arco-time-048/arco/reanalysis_era5_pressure_levels/pl/timeChunked.zarr

    Mathematical docstring:
        Input: stream, variables, levels, bbox, (start_date, end_date), chunking override.
        Output: Tuple[target_url: str, stream_type: str, chunking_scheme: str]
        where:
            stream_type in ('sfc', 'wav', 'pl')
            chunking_scheme in ('geo', 'time')

    Parameters
    ----------
    stream : Optional[str]
        Target data stream ('sfc', 'wav', 'pl'). Inferred from variables or levels if None.
    variables : Optional[Sequence[str]]
        Requested variable names or short codes.
    levels : Optional[Sequence[int]]
        Requested atmospheric pressure levels in hPa.
    bbox : Optional[SpatialBoundingBox]
        Geographic bounding box.
    start_date : Optional[Union[str, datetime, date, pd.Timestamp]]
        Initial timestamp.
    end_date : Optional[Union[str, datetime, date, pd.Timestamp]]
        Terminal timestamp.
    chunking : Optional[str]
        Manual chunking layout override ('geo' or 'time'). Inferred from query shape if None or 'auto'.

    Returns
    -------
    Tuple[str, str, str]
        (target_url, chosen_stream, chosen_chunking)
    """
    # 1. Determine stream ('sfc', 'wav', 'pl')
    chosen_stream = "sfc"
    if stream is not None:
        s_norm = stream.lower().strip()
        if s_norm in ("sfc", "surface", "single", "single_levels"):
            chosen_stream = "sfc"
        elif s_norm in ("wav", "wave", "waves"):
            chosen_stream = "wav"
        elif s_norm in ("pl", "pressure", "pressure_levels"):
            chosen_stream = "pl"
        else:
            raise ValueError(f"Unknown stream '{stream}'. Must be 'sfc', 'wav', or 'pl'.")
    else:
        if levels is not None and len(levels) > 0:
            chosen_stream = "pl"
        elif variables is not None and variables != "all" and not (isinstance(variables, (list, tuple)) and "all" in variables):
            all_req: Set[str] = set()
            for v in variables:
                all_req.update(get_era5_variable_candidates(v))

            wave_vars_set = {
                "swh", "mwd", "mwp",
                "significant_height_of_combined_wind_waves_and_swell",
                "mean_wave_direction", "mean_wave_period"
            }
            pl_vars_set = set(DEFAULT_PRESSURE_LEVEL_FIELDS) | {"q", "r", "t", "u", "v", "w", "z", "clwc", "ciwc"}

            has_wave = bool(all_req & wave_vars_set)
            has_pl = bool(all_req & pl_vars_set)

            if has_pl:
                chosen_stream = "pl"
            elif has_wave:
                chosen_stream = "wav"
            else:
                chosen_stream = "sfc"

    # 2. Determine chunking ('geo' vs 'time')
    if chunking is not None:
        c_norm = chunking.lower().strip()
        if c_norm in ("geo", "geo_chunked", "geochunked", "location"):
            chosen_chunking = "geo"
        elif c_norm in ("time", "time_chunked", "timechunked", "snapshot"):
            chosen_chunking = "time"
        elif c_norm in ("auto", "default"):
            chosen_chunking = _infer_chunking(bbox=bbox, start_date=start_date, end_date=end_date)
        else:
            raise ValueError(f"Unknown chunking '{chunking}'. Must be 'geo', 'time', or 'auto'.")
    else:
        chosen_chunking = _infer_chunking(bbox=bbox, start_date=start_date, end_date=end_date)

    target_url = ECMWF_CDS_ARCO_URLS[chosen_stream][chosen_chunking]
    return target_url, chosen_stream, chosen_chunking


def check_grib_dependencies() -> bool:
    """
    Verify ecCodes and cfgrib availability without directing users to Conda.

    Returns
    -------
    bool
        True if ecCodes and cfgrib are available, False otherwise.
    """
    try:
        import cfgrib  # noqa: F401
        import eccodes  # noqa: F401
        return True
    except (ImportError, RuntimeError) as err:
        logger.warning(
            "ecCodes / cfgrib C-bindings are not available in the current environment: %s. "
            "GRIB-based ERA5 CDS decoding will not function. "
            "Please build and run your environment using the MasharikiWeather Dockerfile "
            "(docker build -t masharikiweather .), which includes precompiled ecCodes C-libraries.",
            err,
        )
        return False


class CopernicusCDSError(Exception):
    """Raised when Copernicus CDS API encounters request limits or failures."""
    pass


def generate_n_days_dict(
    start_date: Union[str, datetime, date, pd.Timestamp],
    end_date: Union[str, datetime, date, pd.Timestamp],
    n_days: int = 5,
) -> Dict[int, Dict[Tuple[str, str], List[str]]]:
    """
    Partition a temporal window into n-day chunks per year and month.
    Accurately handles month-start/end, year-start/end, and leap years (Feb 28 vs Feb 29).

    Parameters
    ----------
    start_date : Union[str, datetime, date, pd.Timestamp]
        Start of period.
    end_date : Union[str, datetime, date, pd.Timestamp]
        End of period.
    n_days : int, default 5
        Number of days per API query chunk to satisfy Copernicus request quotas.

    Returns
    -------
    Dict[int, Dict[Tuple[str, str], List[str]]]
        Dictionary mapping year -> {(week_start_str, week_end_str): [list of zero-padded day strings]}.
    """
    start, end = parse_date_range(start_date, end_date)
    current = start.floor("D")
    end_floor = end.floor("D")
    week_dict: Dict[int, Dict[Tuple[str, str], List[str]]] = {}

    while current <= end_floor:
        year = current.year
        month = current.month
        last_day_of_month = calendar.monthrange(year, month)[1]
        month_end = pd.Timestamp(datetime(year, month, last_day_of_month))

        chunk_end = min(current + timedelta(days=n_days - 1), month_end, end_floor)

        week_start_str = current.strftime("%d_%m")
        week_end_str = chunk_end.strftime("%d_%m")

        num_days = (chunk_end - current).days + 1
        days = [(current + timedelta(days=i)).day for i in range(num_days)]
        days_str = [str(d).zfill(2) for d in days]

        if year not in week_dict:
            week_dict[year] = {}

        week_dict[year][(week_start_str, week_end_str)] = days_str
        current = chunk_end + timedelta(days=1)

    return week_dict


def flatten_accum_ds(ds: xr.Dataset) -> xr.Dataset:
    """
    Flatten accumulated fields along valid_time coordinate, dropping step/number dimensions.
    Adheres only to the proven architecture in AI4weather/src/pipeline/ingestion/era5/process_grib.py.

    Mathematical docstring:
        Input tensor shape: (T_ref, Step, Lat, Lon) or (T_ref, Lat, Lon)
        Output tensor shape: (T_valid, Lat, Lon)
        where:
            valid_time = time + step (for forecast accumulations)
            T_valid = unique valid times across temporal interval.
    """
    if "step" in ds.dims:
        if "valid_time" not in ds.coords and "time" in ds.coords:
            ds = ds.assign_coords(valid_time=ds.time + ds.step)
        stacked = ds.stack(stack_time=("time", "step"))
        if "valid_time" in stacked.coords:
            stacked = stacked.assign_coords(valid_time=stacked["valid_time"])
            ds_flat = stacked.swap_dims({"stack_time": "valid_time"})
            ds_flat = ds_flat.drop_vars(["time", "step", "number", "stack_time"], errors="ignore")
            ds_flat = ds_flat.sortby("valid_time")
            _, unique_idx = np.unique(ds_flat["valid_time"], return_index=True)
            ds_flat = ds_flat.isel(valid_time=unique_idx)
            return ds_flat
    return ds


def resample_era5(
    ds: xr.Dataset,
    freq: str,
) -> xr.Dataset:
    """
    Resample an ERA5 Dataset to a coarser temporal cadence (e.g., '3h', '6h', '1D').

    Only preserves physical conservation laws and thermodynamic dimensionality:
    - Accumulated flux variables (total precipitation, solar radiation, heat fluxes)
      are aggregated only by SUM (conserving mass and energy integrals over the interval).
    - Instantaneous state variables (temperature, dewpoint, pressure, wind components, CAPE, CIN)
      are aggregated only by MEAN (preserving intensive state averages).
    - Static invariant variables (land-sea mask, soil type, geopotential) are preserved via FIRST.
    - If freq is '1h', 'h', or 'native', the dataset is returned unaltered to preserve native cadence.

    Mathematical docstring:
        Input tensor shape: (T_in, [Level], Lat, Lon)
        Output tensor shape: (T_out, [Level], Lat, Lon)
        where:
            T_out = ceil(T_in / step_ratio)
            For flux variables: V_out[t] = sum(V_in[t_i])
            For state variables: V_out[t] = mean(V_in[t_i])

    Parameters
    ----------
    ds : xr.Dataset
        Input ERA5 dataset at native 1-hourly temporal resolution.
    freq : str
        Target resampling frequency string (e.g., '3h', '6h', '1D').

    Returns
    -------
    xr.Dataset
        Resampled CF-compliant Dataset with conserved flux sums and state means.
    """
    if freq.lower() in ("1h", "h", "native"):
        return ds

    time_dim = "time" if "time" in ds.dims else ("valid_time" if "valid_time" in ds.dims else None)
    if time_dim is None:
        return ds

    accum_vars: List[str] = []
    state_vars: List[str] = []
    static_vars: List[str] = []

    for v in ds.data_vars:
        v_str = str(v)
        v_lower = v_str.lower()
        if time_dim not in ds[v].dims:
            static_vars.append(v_str)
        elif v_str in ERA5_ACCUMULATED_VARIABLES or v_lower in ("tp", "cp", "lsp", "ssrd", "tisr", "strd", "sshf", "slhf"):
            accum_vars.append(v_str)
        elif any(kw in v_lower for kw in ("precip", "solar_radiation", "thermal_radiation", "heat_flux", "evaporation", "snowfall")) and not any(kw in v_lower for kw in ("cape", "cin", "convective_available", "convective_inhibition")):
            accum_vars.append(v_str)
        else:
            state_vars.append(v_str)

    resampled_parts: List[xr.Dataset] = []
    if accum_vars:
        resampled_parts.append(ds[accum_vars].resample({time_dim: freq}).sum(skipna=False))
    if state_vars:
        resampled_parts.append(ds[state_vars].resample({time_dim: freq}).mean(skipna=False))
    if static_vars:
        resampled_parts.append(ds[static_vars])

    if not resampled_parts:
        return ds

    merged = xr.merge(resampled_parts, compat="override")
    return merged


def _open_single_era5_grib(path: Path, **kwargs: Any) -> xr.Dataset:
    """
    Open a single ECMWF ERA5 GRIB file, separating instantaneous and accumulated fields.
    Adheres to AI4weather process_grib.py by flattening accumulations and preserving 1-hourly resolution.
    """
    p_str = str(path)
    sub_ds: List[xr.Dataset] = []

    # 1. Attempt opening instantaneous fields (stepType='instant')
    ds_inst: Optional[xr.Dataset] = None
    try:
        ds_inst = xr.open_dataset(
            p_str,
            engine="cfgrib",
            decode_timedelta=True,
            backend_kwargs={"filter_by_keys": {"stepType": "instant"}},
            **kwargs,
        )
        if len(ds_inst.data_vars) > 0:
            sub_ds.append(ds_inst)
    except Exception:
        pass

    # 2. Attempt opening accumulated fields (stepType='accum')
    ds_accum: Optional[xr.Dataset] = None
    try:
        ds_accum = xr.open_dataset(
            p_str,
            engine="cfgrib",
            decode_timedelta=True,
            backend_kwargs={"filter_by_keys": {"stepType": "accum"}},
            **kwargs,
        )
        if len(ds_accum.data_vars) > 0:
            if "time" in ds_accum.coords and "step" in ds_accum.coords and "valid_time" not in ds_accum.coords:
                ds_accum = ds_accum.assign_coords(valid_time=ds_accum.time + ds_accum.step)
            ds_accum = flatten_accum_ds(ds_accum)
            sub_ds.append(ds_accum)
    except Exception:
        pass

    # 3. Fallback: plain open_dataset (for uniform files, e.g. pressure levels)
    if not sub_ds:
        try:
            ds_plain = xr.open_dataset(p_str, engine="cfgrib", decode_timedelta=True, **kwargs)
            sub_ds.append(ds_plain)
        except Exception as exc:
            raise RuntimeError(f"Failed opening GRIB file {path} with cfgrib: {exc}") from exc

    # Align on common valid_time or time if both instant and accum present
    if ds_inst is not None and ds_accum is not None and len(ds_inst.data_vars) > 0 and len(ds_accum.data_vars) > 0:
        inst_time_coord = "valid_time" if "valid_time" in ds_inst.coords else "time"
        accum_time_coord = "valid_time" if "valid_time" in ds_accum.coords else "time"
        if inst_time_coord in ds_inst.coords and accum_time_coord in ds_accum.coords:
            inst_times = ds_inst[inst_time_coord].values
            accum_times = ds_accum[accum_time_coord].values
            common = np.intersect1d(inst_times, accum_times)
            if len(common) > 0:
                inst_sel = ds_inst.sel({inst_time_coord: common})
                accum_sel = ds_accum.sel({accum_time_coord: common})
                merged = xr.merge([inst_sel, accum_sel], compat="override")
            else:
                merged = xr.merge(sub_ds, compat="override")
        else:
            merged = xr.merge(sub_ds, compat="override")
    else:
        merged = sub_ds[0] if len(sub_ds) == 1 else xr.merge(sub_ds, compat="override")

    # Coordinate standardization: latitude -> lat, longitude -> lon
    rename_map: Dict[str, str] = {}
    if "latitude" in merged.coords and "lat" not in merged.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in merged.coords and "lon" not in merged.coords:
        rename_map["longitude"] = "lon"
    if "valid_time" in merged.coords and "time" not in merged.dims:
        rename_map["valid_time"] = "time"

    if rename_map:
        merged = merged.rename(rename_map)

    # Clean redundant coordinate dimensions
    drop_candidates = ["surface", "number", "step", "valid_time"]
    actual_drops = [v for v in drop_candidates if v in merged.variables and v not in merged.dims]
    if actual_drops:
        merged = merged.drop_vars(actual_drops, errors="ignore")

    return merged


def open_era5_grib(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    harmonize: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Open ECMWF ERA5 GRIB file(s) with separate handling for instantaneous and accumulated fields.

    In ECMWF ERA5 single-levels, instantaneous variables (e.g., 2t, 10u, sp) have
    stepType='instant', while accumulated variables (e.g., tp, ssrd, sshf) have
    stepType='accum'. Opening mixed GRIB files with cfgrib without stepType filtering
    causes DatasetBuildError.

    This function attempts opening both stepType subsets separately ('instant' and 'accum')
    and merges them into a unified CF-compliant xarray.Dataset.
    Native physical units are preserved per project specifications (no unit conversion applied).

    Mathematical docstring:
    Input: single GRIB file path or sequence of file paths.
    Output: xr.Dataset with dimensions (time, [level], lat, lon) and combined data_vars.

    Parameters
    ----------
    paths : Union[str, Path, Sequence[Union[str, Path]]]
        Path or sequence of paths to GRIB files.
    harmonize : bool, default False
        If True, harmonizes CDS short names (e.g. '2t', 'tp') to canonical descriptive names.
    **kwargs : Any
        Additional keyword arguments forwarded to xr.open_dataset.

    Returns
    -------
    xr.Dataset
        Unified CF-compliant Dataset containing both instantaneous and accumulated fields.
    """
    try:
        import cfgrib  # noqa: F401
        import eccodes  # noqa: F401
    except (ImportError, RuntimeError) as exc:
        raise ImportError(
            f"ecCodes / cfgrib C-bindings are not available in the current environment: {exc}. "
            "GRIB-based ERA5 CDS decoding will not function. "
            "Please build and run your environment using the MasharikiWeather Dockerfile "
            "(docker build -t masharikiweather .), which includes precompiled ecCodes C-libraries."
        ) from exc

    file_list = [Path(p) for p in paths] if isinstance(paths, (list, tuple)) else [Path(paths)]
    if not file_list:
        raise ValueError("No GRIB files provided to open_era5_grib.")

    datasets = [_open_single_era5_grib(f, **kwargs) for f in file_list]
    if len(datasets) == 1:
        combined = datasets[0]
    else:
        if "time" in datasets[0].dims:
            combined = xr.concat(datasets, dim="time", coords="minimal", compat="override")
            combined = combined.sortby("time")
        else:
            combined = xr.merge(datasets, compat="override")

    if harmonize:
        combined = harmonize_era5_dataset(combined, to_canonical=True)

    return combined


def _open_single_era5_netcdf(path: Path, **kwargs: Any) -> xr.Dataset:
    """Open a single NetCDF file downloaded from CDS and standardize coordinates."""
    p_str = str(path)
    ds: Optional[xr.Dataset] = None
    for eng in ["netcdf4", "h5netcdf", None]:
        try:
            ds = xr.open_dataset(p_str, engine=eng, **kwargs) if eng else xr.open_dataset(p_str, **kwargs)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(f"Failed opening NetCDF file {path} with available engines.")

    rename_map: Dict[str, str] = {}
    if "latitude" in ds.coords and "lat" not in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename_map["longitude"] = "lon"
    if "valid_time" in ds.coords and "time" not in ds.dims:
        rename_map["valid_time"] = "time"
    if rename_map:
        ds = ds.rename(rename_map)

    # Clean redundant coordinate variables
    drop_candidates = ["surface", "number", "valid_time"]
    actual_drops = [v for v in drop_candidates if v in ds.variables and v not in ds.dims]
    if actual_drops:
        ds = ds.drop_vars(actual_drops, errors="ignore")

    return ds


def open_era5_netcdf(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    harmonize: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Open ECMWF ERA5 NetCDF file(s) bypassing cfgrib and eccodes C-dependencies.

    Parameters
    ----------
    paths : Union[str, Path, Sequence[Union[str, Path]]]
        Path or sequence of paths to NetCDF (.nc) files.
    harmonize : bool, default False
        If True, harmonizes short names (e.g. '2t', 'tp') to canonical descriptive names.
    **kwargs : Any
        Additional keyword arguments forwarded to xr.open_dataset.

    Returns
    -------
    xr.Dataset
        Unified CF-compliant Dataset containing variables from the NetCDF files.
    """
    file_list = [Path(p) for p in paths] if isinstance(paths, (list, tuple)) else [Path(paths)]
    if not file_list:
        raise ValueError("No NetCDF files provided to open_era5_netcdf.")

    datasets = [_open_single_era5_netcdf(f, **kwargs) for f in file_list]
    if len(datasets) == 1:
        combined = datasets[0]
    else:
        sample_vars = set(datasets[0].data_vars)
        same_vars = all(set(d.data_vars) == sample_vars for d in datasets)
        if same_vars and "time" in datasets[0].dims:
            combined = xr.concat(datasets, dim="time", coords="minimal", compat="override")
            combined = combined.sortby("time")
        else:
            grouped: Dict[Tuple[str, ...], List[xr.Dataset]] = {}
            for d in datasets:
                key = tuple(sorted(str(v) for v in d.data_vars))
                grouped.setdefault(key, []).append(d)
            merged_groups = []
            for k, group_ds in grouped.items():
                if len(group_ds) == 1:
                    merged_groups.append(group_ds[0])
                elif "time" in group_ds[0].dims:
                    c = xr.concat(group_ds, dim="time", coords="minimal", compat="override").sortby("time")
                    merged_groups.append(c)
                else:
                    merged_groups.append(group_ds[0])
            combined = xr.merge(merged_groups, compat="override")

    if harmonize:
        combined = harmonize_era5_dataset(combined, to_canonical=True)

    return combined


class ERA5Extractor(BaseExtractor):
    """
    Multi-Backend High-Performance ERA5 Extraction Pipeline.

    Supports three extraction backends:
    1. Legacy CDS API (`use_legacy=True`, default):
       Extracts directly from ECMWF Copernicus Climate Data Store API.
       Supports NetCDF4 format (`data_format="netcdf4"`, default) to bypass cfgrib/eccodes
       C-dependencies, or GRIB format (`data_format="grib"`).
       Applies adaptive temporal chunking starting at 5 days, automatically catching request
       size limit errors and timeouts, reducing chunk size (4 days, 3 days, etc.) on-the-fly.
    2. CDS ARCO (`use_cds_arco=True`):
       Lightweight access to ECMWF CDS analysis-ready cloud Zarr stores using obstore
       retry/resilience mechanisms and Bearer token authentication via CDSAPI_KEY or ~/.cdsapirc.
    3. Google ARCO (`use_google_arco=True`):
       Accesses the public Google Cloud Public Dataset ERA5 ARCO Zarr store.
    """

    extractor_name: str = "era5"
    network_name: str = "era5"

    def __init__(
        self,
        output_dir: Union[str, Path] = "./mashariki-data/gridded/era5",
        single_fields: Optional[Sequence[str]] = None,
        pressure_fields: Optional[Sequence[str]] = None,
        wave_fields: Optional[Sequence[str]] = None,
        pressure_levels: Optional[Sequence[int]] = None,
        use_legacy: Optional[bool] = None,
        use_cds_arco: Optional[bool] = None,
        use_google_arco: Optional[bool] = None,
        data_format: str = "grib",
        arco_url: str = DEFAULT_GOOGLE_ARCO_URL,
        cds_arco_url: Optional[str] = None,
        chunking: Optional[str] = None,
        stream: Optional[str] = None,
        cds_key: Optional[str] = None,
        cds_url: Optional[str] = None,
        cds_rc_file: Optional[Union[str, Path]] = None,
        prefer_arco: Optional[bool] = None,
        force_cds: Optional[bool] = None,
    ) -> None:
        """Initialize ERA5 extractor with multi-backend architecture."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.single_fields = list(single_fields if single_fields is not None else DEFAULT_SINGLE_LEVEL_FIELDS)
        self.pressure_fields = list(pressure_fields if pressure_fields is not None else DEFAULT_PRESSURE_LEVEL_FIELDS)
        self.wave_fields = list(wave_fields if wave_fields is not None else DEFAULT_WAVE_FIELDS)
        self.pressure_levels = list(pressure_levels if pressure_levels is not None else DEFAULT_PRESSURE_LEVELS)

        self._user_specified_legacy = use_legacy is not None
        self.use_legacy = use_legacy if use_legacy is not None else True
        self.use_cds_arco = bool(use_cds_arco) if use_cds_arco is not None else False
        self.use_google_arco = bool(use_google_arco) if use_google_arco is not None else False
        self.data_format = data_format

        self.prefer_arco = prefer_arco if prefer_arco is not None else False
        self.force_cds = force_cds if force_cds is not None else False

        self.arco_url = arco_url
        self.cds_arco_url = cds_arco_url
        self.chunking = chunking
        self.stream = stream
        self.cds_key = cds_key
        self.cds_url = cds_url
        self.cds_rc_file = cds_rc_file
        self._arco_ds: Optional[xr.Dataset] = None

    def check_google_arco_availability(
        self,
        start_date: Union[str, datetime, date, pd.Timestamp],
        end_date: Union[str, datetime, date, pd.Timestamp],
        variables: Optional[Sequence[str]] = None,
        bbox: Optional[SpatialBoundingBox] = None,
    ) -> bool:
        """
        Check whether requested temporal interval and variables are available in Google ARCO ERA5.

        Verifies:
        1. Requested dates fall within valid ERA5 operational coverage (1940 to present).
        2. Remote Google Cloud ARCO-ERA5 Zarr store is accessible without network/permission errors.
        3. Store temporal horizon covers the requested start and end timestamps.
        4. If variables are specified, candidate identifiers for each requested variable exist in the dataset.

        Mathematical docstring:
            Input: start_date, end_date, variables: Sequence[V], bbox: SpatialBoundingBox
            Output: bool (True if all candidates exist in ds.data_vars and [t_start, t_end] in [t_min, t_max])

        Parameters
        ----------
        start_date : Union[str, datetime, date, pd.Timestamp]
            Initial observation timestamp.
        end_date : Union[str, datetime, date, pd.Timestamp]
            Terminal observation timestamp.
        variables : Optional[Sequence[str]]
            Variables to extract.
        bbox : Optional[SpatialBoundingBox]
            Spatial bounding box (unused for global store availability).

        Returns
        -------
        bool
            True if Google ARCO ERA5 covers the request; False otherwise.
        """
        try:
            s_dt, e_dt = parse_date_range(start_date, end_date)
            now_dt = pd.Timestamp.now(tz=e_dt.tz if e_dt.tzinfo else None)
            if s_dt < pd.Timestamp("1940-01-01") or e_dt > now_dt:
                return False

            ds = self._get_arco_dataset()
            if ds is None:
                return False

            time_dim = "time" if "time" in ds.coords or "time" in ds.dims else None
            if time_dim is not None:
                time_coord = ds[time_dim]
                t_min = pd.to_datetime(time_coord[0].values)
                t_max = pd.to_datetime(time_coord[-1].values)

                s_naive = s_dt.tz_localize(None) if s_dt.tzinfo else s_dt
                e_naive = e_dt.tz_localize(None) if e_dt.tzinfo else e_dt
                t_min_naive = t_min.tz_localize(None) if t_min.tzinfo else t_min
                t_max_naive = t_max.tz_localize(None) if t_max.tzinfo else t_max

                if s_naive < t_min_naive or e_naive > t_max_naive:
                    logger.info(
                        "Google ARCO ERA5 temporal bounds [%s, %s] do not cover requested interval [%s, %s].",
                        t_min_naive, t_max_naive, s_naive, e_naive,
                    )
                    return False

            if variables is not None and variables != "all" and not (isinstance(variables, (list, tuple)) and "all" in variables):
                for v in variables:
                    cands = get_era5_variable_candidates(v)
                    if not any(c in ds.data_vars for c in cands):
                        logger.info(
                            "Variable '%s' (candidates: %s) is not present in Google ARCO ERA5 catalog.",
                            v, cands,
                        )
                        return False

            return True
        except Exception as exc:
            logger.info("Google ARCO ERA5 availability check failed: %s", exc)
            return False

    def _get_arco_dataset(self) -> xr.Dataset:
        """Lazily initialize anonymous Google Cloud ARCO-ERA5 connection."""
        if self._arco_ds is None:
            logger.info("Opening remote Google ARCO ERA5 Zarr store: %s", self.arco_url)
            self._arco_ds = xr.open_zarr(
                self.arco_url,
                chunks=None,
                storage_options={"token": "anon"},
            )
        return self._arco_ds

    def extract_arco_slice(
        self,
        bbox: SpatialBoundingBox,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        variables: Optional[Sequence[str]] = None,
        levels: Optional[Sequence[int]] = None,
    ) -> xr.Dataset:
        """
        Extract spatiotemporal tensor slice from Google Cloud ARCO ERA5 directly.

        Mathematical docstring:
        Input spatial interval: [lat_min, lat_max], [lon_min, lon_max]
        Output tensor shape: (T, [Level], Lat, Lon)
        where:
          T = floor((t_end - t_start) / 1 hour) + 1
          Lat = floor((lat_max - lat_min) / 0.25) + 1
          Lon = floor((lon_max - lon_min) / 0.25) + 1
        """
        ds = self._get_arco_dataset()
        if variables is None or variables == "all" or (isinstance(variables, (list, tuple)) and "all" in variables):
            requested_vars = list(set(self.single_fields + self.pressure_fields))
            valid_vars = [v for v in requested_vars if v in ds.data_vars]
        else:
            matched: List[str] = []
            for v in variables:
                for cand in get_era5_variable_candidates(v):
                    if cand in ds.data_vars:
                        matched.append(cand)
                        break
            valid_vars = list(dict.fromkeys(matched))
        if not valid_vars:
            raise ValueError(f"None of the requested variables {variables} exist in ARCO catalog.")

        missing_vars = [v for v in (variables or []) if not any(c in valid_vars for c in get_era5_variable_candidates(v))]
        if missing_vars:
            logger.warning("Variables missing from ARCO catalog and skipped: %s", missing_vars)

        start_str = pd.to_datetime(start_date).strftime("%Y-%m-%dT%H:00:00")
        end_str = pd.to_datetime(end_date).strftime("%Y-%m-%dT%H:00:00")

        # ARCO latitude descends from 90 to -90
        lat_max, lon_min, lat_min, lon_max = bbox.to_area_list()

        sliced = ds[valid_vars].sel(
            time=slice(start_str, end_str),
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max),
        )

        target_levels = list(levels) if levels is not None else self.pressure_levels
        if "level" in sliced.dims:
            valid_lvls = [lvl for lvl in target_levels if lvl in sliced.level.values]
            if valid_lvls:
                sliced = sliced.sel(level=valid_lvls)

        rename_map = {}
        if "latitude" in sliced.coords:
            rename_map["latitude"] = "lat"
        if "longitude" in sliced.coords:
            rename_map["longitude"] = "lon"
        if rename_map:
            sliced = sliced.rename(rename_map)

        dim_order = ["time"]
        if "level" in sliced.dims:
            dim_order.append("level")
        if "lat" in sliced.dims and "lon" in sliced.dims:
            dim_order.extend(["lat", "lon"])
        sliced = sliced.transpose(*dim_order)

        sliced = harmonize_era5_dataset(sliced, to_canonical=True)
        return sliced

    def extract_google_arco_slice(
        self,
        bbox: SpatialBoundingBox,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        variables: Optional[Sequence[str]] = None,
        levels: Optional[Sequence[int]] = None,
    ) -> xr.Dataset:
        """Extract spatiotemporal tensor slice from Google Cloud ARCO-ERA5 Zarr store."""
        return self.extract_arco_slice(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            levels=levels,
        )

    def _extract_single_cds_arco_store(
        self,
        target_url: str,
        bbox: SpatialBoundingBox,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        variables: Optional[Sequence[str]] = None,
        levels: Optional[Sequence[int]] = None,
        cds_key: Optional[str] = None,
        harmonize: bool = True,
    ) -> xr.Dataset:
        """
        Open and slice a single ECMWF CDS ARCO Zarr store via obstore.

        Mathematical docstring:
            Input spatial interval: [bbox.lat_min, bbox.lat_max], [bbox.lon_min, bbox.lon_max]
            Input temporal interval: [start_date, end_date]
            Output tensor shape: (T, [Level], Lat, Lon)
        """
        try:
            from obstore.store import HTTPStore
        except ImportError as exc:
            raise ImportError(
                "obstore is required for CDS ARCO Zarr loading. "
                "Please install it with: pip install obstore>=0.10.0 or pip install 'masharikiweather[arco]'"
            ) from exc
        from zarr.storage import ObjectStore

        cdsapi_key = get_cds_api_key(key=cds_key or self.cds_key, rc_file=self.cds_rc_file)

        logger.info("Opening remote CDS ARCO Zarr store via obstore: %s", target_url)
        http_store = HTTPStore(target_url, client_options={"default_headers": {"Authorization": f"Bearer {cdsapi_key}"}})
        store = ObjectStore(http_store, read_only=True)
        ds = xr.open_zarr(store)

        if variables is None or variables == "all" or (isinstance(variables, (list, tuple)) and "all" in variables):
            requested_vars = list(ds.data_vars)
        else:
            matched: List[str] = []
            for v in variables:
                cands = get_era5_variable_candidates(v)
                for cand in cands:
                    if cand in ds.data_vars:
                        matched.append(cand)
                        break
            matched = list(dict.fromkeys(matched))
            if not matched:
                for v in variables:
                    for cand in get_era5_variable_candidates(v):
                        if cand in ds.data_vars:
                            matched.append(cand)
                matched = list(dict.fromkeys(matched))
            requested_vars = matched if matched else list(ds.data_vars)

        start_str = pd.to_datetime(start_date).strftime("%Y-%m-%dT%H:00:00")
        end_str = pd.to_datetime(end_date).strftime("%Y-%m-%dT%H:00:00")

        time_dim = "valid_time" if "valid_time" in ds.dims else ("time" if "time" in ds.dims else None)
        lat_dim = "latitude" if "latitude" in ds.dims else ("lat" if "lat" in ds.dims else None)
        lon_dim = "longitude" if "longitude" in ds.dims else ("lon" if "lon" in ds.dims else None)

        sel_dict: Dict[str, Any] = {}
        if time_dim:
            sel_dict[time_dim] = slice(start_str, end_str)

        if lat_dim:
            lat_vals = ds[lat_dim].values
            is_descending = len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]
            if is_descending:
                sel_dict[lat_dim] = slice(bbox.lat_max, bbox.lat_min)
            else:
                sel_dict[lat_dim] = slice(bbox.lat_min, bbox.lat_max)

        if lon_dim:
            sel_dict[lon_dim] = slice(bbox.lon_min, bbox.lon_max)

        lvl_dim = None
        for candidate in ["pressureLevel", "pressure_level", "level"]:
            if candidate in ds.dims or candidate in ds.coords:
                lvl_dim = candidate
                break

        if lvl_dim:
            target_levels = list(levels) if levels is not None else self.pressure_levels
            valid_lvls = [lvl for lvl in target_levels if lvl in ds[lvl_dim].values]
            if valid_lvls:
                sel_dict[lvl_dim] = valid_lvls

        sliced = ds[requested_vars].sel(sel_dict)

        rename_map: Dict[str, str] = {}
        if lat_dim and lat_dim != "lat":
            rename_map[lat_dim] = "lat"
        if lon_dim and lon_dim != "lon":
            rename_map[lon_dim] = "lon"
        if time_dim and time_dim != "time":
            rename_map[time_dim] = "time"
        if lvl_dim and lvl_dim != "level":
            rename_map[lvl_dim] = "level"
        if rename_map:
            sliced = sliced.rename(rename_map)

        if harmonize:
            sliced = harmonize_era5_dataset(sliced, to_canonical=True)

        dim_order = ["time"]
        if "level" in sliced.dims:
            dim_order.append("level")
        if "lat" in sliced.dims and "lon" in sliced.dims:
            dim_order.extend(["lat", "lon"])
        present = [d for d in dim_order if d in sliced.dims]
        extra = [d for d in sliced.dims if d not in present]
        sliced = sliced.transpose(*(present + extra))

        return sliced

    def extract_cds_arco_slice(
        self,
        bbox: SpatialBoundingBox,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        variables: Optional[Sequence[str]] = None,
        levels: Optional[Sequence[int]] = None,
        stream: Optional[str] = None,
        chunking: Optional[str] = None,
        url: Optional[str] = None,
        cds_key: Optional[str] = None,
        harmonize: bool = True,
    ) -> xr.Dataset:
        """
        Extract spatiotemporal tensor slice from Copernicus CDS ARCO Zarr store using obstore
        and the Query-Aware Routing Matrix.

        Parameters
        ----------
        bbox : SpatialBoundingBox
            Target spatial boundary.
        start_date : Union[str, datetime, pd.Timestamp]
            Initial date.
        end_date : Union[str, datetime, pd.Timestamp]
            Terminal date.
        variables : Optional[Sequence[str]]
            Variables to extract.
        levels : Optional[Sequence[int]]
            Pressure levels in hPa.
        stream : Optional[str]
            Target ARCO stream ('sfc', 'wav', or 'pl'). Inferred from variables/levels if None.
        chunking : Optional[str]
            Chunking layout override ('geo' or 'time'). Inferred from query shape if None.
        url : Optional[str]
            Direct target Zarr store URL override.
        cds_key : Optional[str]
            Copernicus CDS API key.
        harmonize : bool, default True
            Whether to translate CDS short codes to canonical variable names.

        Returns
        -------
        xr.Dataset
            CF-compliant sliced Dataset.
        """
        if url is not None:
            return self._extract_single_cds_arco_store(
                target_url=url,
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                variables=variables,
                levels=levels,
                cds_key=cds_key,
                harmonize=harmonize,
            )

        user_stream = stream or self.stream
        user_chunking = chunking or self.chunking

        # When stream is not direct, detect if variables span multiple streams
        if user_stream is None and variables is not None and variables != "all" and not (isinstance(variables, (list, tuple)) and "all" in variables):
            wave_vars_set = {
                "swh", "mwd", "mwp",
                "significant_height_of_combined_wind_waves_and_swell",
                "mean_wave_direction", "mean_wave_period"
            }
            pl_vars_set = set(DEFAULT_PRESSURE_LEVEL_FIELDS) | {"q", "r", "t", "u", "v", "w", "z", "clwc", "ciwc"}

            req_sfc = []
            req_wav = []
            req_pl = []
            for v in variables:
                cands = set(get_era5_variable_candidates(v))
                if cands & wave_vars_set:
                    req_wav.append(v)
                elif cands & pl_vars_set:
                    req_pl.append(v)
                else:
                    req_sfc.append(v)

            streams_needed = []
            if req_sfc:
                streams_needed.append(("sfc", req_sfc))
            if req_wav:
                streams_needed.append(("wav", req_wav))
            if req_pl or (levels is not None and len(levels) > 0):
                streams_needed.append(("pl", req_pl or self.pressure_fields))

            if len(streams_needed) > 1:
                logger.info("Multi-stream CDS ARCO query requested: %s", [s[0] for s in streams_needed])
                sub_datasets: List[xr.Dataset] = []
                for st_name, st_vars in streams_needed:
                    target_url, _, _ = resolve_cds_arco_url(
                        stream=st_name,
                        variables=st_vars,
                        levels=levels,
                        bbox=bbox,
                        start_date=start_date,
                        end_date=end_date,
                        chunking=user_chunking,
                    )
                    sub_ds = self._extract_single_cds_arco_store(
                        target_url=target_url,
                        bbox=bbox,
                        start_date=start_date,
                        end_date=end_date,
                        variables=st_vars,
                        levels=levels,
                        cds_key=cds_key,
                        harmonize=harmonize,
                    )
                    sub_datasets.append(sub_ds)
                return xr.merge(sub_datasets, compat="override")

        target_url, _, _ = resolve_cds_arco_url(
            stream=user_stream,
            variables=variables,
            levels=levels,
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            chunking=user_chunking,
        )
        return self._extract_single_cds_arco_store(
            target_url=target_url,
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            levels=levels,
            cds_key=cds_key,
            harmonize=harmonize,
        )

    def extract_cds_slice(
        self,
        bbox: SpatialBoundingBox,
        start_date: Union[str, datetime, date, pd.Timestamp],
        end_date: Union[str, datetime, date, pd.Timestamp],
        variables: Optional[Sequence[str]] = None,
        data_levels: str = "both",
        n_days_per_chunk: int = 5,
        time_interval: int = 1,
        data_format: str = "grib",
        cds_key: Optional[str] = None,
        cds_url: Optional[str] = None,
    ) -> List[Path]:
        """
        Extract ERA5 data from Copernicus CDS API using adaptive temporal chunking.

        CRITICAL ADAPTIVE CHUNKING RULE:
        Starts with a maximum 5-day interval. If the CDS API throws an error
        indicating the request size is too large or times out, catches the error,
        logs 'Request size too large, reducing chunk size to X days...', and
        automatically retries with a smaller chunk size (4 days, 3 days, etc.).

        Parameters
        ----------
        bbox : SpatialBoundingBox
            Geographic bounding box.
        start_date : Union[str, datetime, date, pd.Timestamp]
            Initial date.
        end_date : Union[str, datetime, date, pd.Timestamp]
            Terminal date.
        variables : Optional[Sequence[str]]
            Variables to extract.
        data_levels : str, default 'both'
            'single', 'pressure', or 'both'.
        n_days_per_chunk : int, default 5
            Starting number of days per query chunk (capped at 5 days).
        time_interval : int, default 1
            Cadence step in hours.
        data_format : str, default 'netcdf4'
            'netcdf4' (bypasses cfgrib C-dependency) or 'grib'.
        cds_key : Optional[str]
            Copernicus CDS API key.
        cds_url : Optional[str]
            Copernicus CDS API URL.

        Returns
        -------
        List[Path]
            Paths to downloaded NetCDF or GRIB files.
        """
        try:
            import cdsapi
        except ImportError as exc:
            raise ImportError("cdsapi library is required for CDS retrieval. Install via 'pip install cdsapi'.") from exc

        creds = CredentialsManager.get_cds_credentials(
            key=cds_key or self.cds_key,
            url=cds_url or self.cds_url,
            rc_file=self.cds_rc_file,
        )
        client_kwargs: Dict[str, Any] = {}
        if creds.get("url"):
            client_kwargs["url"] = creds["url"]
        if creds.get("key"):
            client_kwargs["key"] = creds["key"]
        client = cdsapi.Client(**client_kwargs)

        area_str = bbox.to_area_str_list()

        if variables is not None and variables != "all" and not (isinstance(variables, (list, tuple)) and "all" in variables):
            mapped_vars = [canonicalize_era5_variable(v) for v in variables]
            if data_levels == "pressure":
                pressure_vars = list(mapped_vars)
                single_vars = []
            elif data_levels == "single":
                single_vars = list(mapped_vars)
                pressure_vars = []
            else:
                pressure_vars = [v for v in mapped_vars if v in self.pressure_fields]
                single_vars = [v for v in mapped_vars if v not in self.pressure_fields]
        else:
            single_vars = list(self.single_fields)
            pressure_vars = list(self.pressure_fields)

        fmt = data_format.lower().strip()
        is_netcdf = fmt in ("netcdf4", "netcdf", "nc")
        data_fmt = "netcdf" if is_netcdf else "grib"
        ext = ".nc" if is_netcdf else ".grib"

        # Adaptive chunking: start with maximum 5-day interval
        current_chunk_size = min(max(1, int(n_days_per_chunk)), 5)
        start_ts, end_ts = parse_date_range(start_date, end_date)
        curr = start_ts.floor("D")
        end_floor = end_ts.floor("D")
        downloaded_paths: List[Path] = []

        while curr <= end_floor:
            chunk_success = False
            while not chunk_success:
                year = curr.year
                month = curr.month
                last_day = calendar.monthrange(year, month)[1]
                month_end = pd.Timestamp(datetime(year, month, last_day))
                chunk_end = min(curr + timedelta(days=current_chunk_size - 1), month_end, end_floor)

                w_start = curr.strftime("%d_%m")
                w_end = chunk_end.strftime("%d_%m")
                num_days = (chunk_end - curr).days + 1
                days = [(curr + timedelta(days=i)).day for i in range(num_days)]
                days_str = [str(d).zfill(2) for d in days]
                month_str = str(month).zfill(2)
                times = [f"{str(h).zfill(2)}:00" for h in range(0, 24, time_interval)]
                prefix = f"era5_cds_{w_start}_to_{w_end}_{year}"

                chunk_files: List[Path] = []
                try:
                    if data_levels in ("single", "both") and single_vars:
                        out_path = self.output_dir / f"{prefix}_single{ext}"
                        if out_path.exists():
                            logger.info("File already exists locally: %s", out_path)
                            chunk_files.append(out_path)
                        else:
                            request_payload = {
                                "product_type": ["reanalysis"],
                                "variable": single_vars,
                                "year": [str(year)],
                                "month": [month_str],
                                "day": days_str,
                                "time": times,
                                "area": area_str,
                                "grid": ["0.25", "0.25"],
                                "data_format": data_fmt,
                                "download_format": "unarchived",
                            }
                            self._execute_cds_request(client, "reanalysis-era5-single-levels", request_payload, out_path)
                            chunk_files.append(out_path)

                    if data_levels in ("pressure", "both") and pressure_vars:
                        out_path = self.output_dir / f"{prefix}_pressure{ext}"
                        if out_path.exists():
                            logger.info("File already exists locally: %s", out_path)
                            chunk_files.append(out_path)
                        else:
                            request_payload = {
                                "product_type": ["reanalysis"],
                                "variable": pressure_vars,
                                "pressure_level": [str(p) for p in self.pressure_levels],
                                "year": [str(year)],
                                "month": [month_str],
                                "day": days_str,
                                "time": times,
                                "area": area_str,
                                "grid": ["0.25", "0.25"],
                                "data_format": data_fmt,
                                "download_format": "unarchived",
                            }
                            self._execute_cds_request(client, "reanalysis-era5-pressure-levels", request_payload, out_path)
                            chunk_files.append(out_path)

                    downloaded_paths.extend(chunk_files)
                    chunk_success = True
                    curr = chunk_end + timedelta(days=1)

                except Exception as exc:
                    msg = str(exc).lower()
                    is_size_or_timeout = any(
                        term in msg for term in [
                            "request is too large", "too large", "limit", "cost limit",
                            "timeout", "timed out", "time out", "gateway", "504", "408", "quota", "exceeded"
                        ]
                    )
                    if is_size_or_timeout and current_chunk_size > 1:
                        new_size = current_chunk_size - 1
                        logger.info("Request size too large, reducing chunk size to %d days...", new_size)
                        current_chunk_size = new_size
                        continue
                    raise

        return downloaded_paths

    def _execute_cds_request(
        self,
        client: Any,
        dataset: str,
        request: Dict[str, Any],
        target_path: Path,
    ) -> None:
        """Execute a single CDS API request with error interception."""
        logger.info("Submitting CDS API request to dataset '%s' -> %s", dataset, target_path.name)
        tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".tmp")
        os.close(tmp_fd)
        try:
            client.retrieve(dataset, request, tmp_path_str)
            shutil.move(tmp_path_str, target_path)
            logger.info("Downloaded: %s", target_path)
        except Exception as exc:
            msg = str(exc)
            msg_lower = msg.lower()
            if any(t in msg_lower for t in ["request is too large", "cost limit", "limit", "too large", "timeout", "timed out", "quota", "exceeded"]):
                raise CopernicusCDSError(
                    f"Copernicus CDS limit exceeded: {msg}. "
                    "Advice: Reduce requested spatial bounds or reduce n_days_per_chunk (e.g. 2 or 1 days), "
                    "or query using Google Cloud ARCO ERA5."
                ) from exc
            raise CopernicusCDSError(f"CDS API request failed for {target_path.name}: {msg}") from exc
        finally:
            if os.path.exists(tmp_path_str):
                try:
                    os.remove(tmp_path_str)
                except OSError:
                    pass

    def extract(
        self,
        start_date: Union[str, datetime, date, pd.Timestamp],
        end_date: Union[str, datetime, date, pd.Timestamp],
        bbox: Optional[SpatialBoundingBox] = None,
        variables: Optional[Sequence[str]] = None,
        levels: Optional[Sequence[int]] = None,
        save_zarr: bool = True,
        zarr_name: Optional[str] = None,
        use_legacy: Optional[bool] = None,
        use_cds_arco: Optional[bool] = None,
        use_google_arco: Optional[bool] = None,
        data_format: Optional[str] = None,
        chunk_days: int = 5,
        chunking: Optional[str] = None,
        stream: Optional[str] = None,
        cds_key: Optional[str] = None,
        cds_url: Optional[str] = None,
        cds_arco_url: Optional[str] = None,
        force_cds: Optional[bool] = None,
        prefer_arco: Optional[bool] = None,
        show_progress: bool = True,
        resume: bool = True,
        overwrite: bool = False,
        keep_cache: bool = False,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> Union[xr.Dataset, Path, ExtractionEstimate]:
        """
        Execute ERA5 extraction across legacy CDS API, CDS ARCO, or Google ARCO backends.

        Parameters
        ----------
        start_date : Union[str, datetime, date, pd.Timestamp]
            Initial date.
        end_date : Union[str, datetime, date, pd.Timestamp]
            Terminal date.
        bbox : Optional[SpatialBoundingBox]
            Spatial bounding box (defaults to AFRICA_BBOX).
        variables : Optional[Sequence[str]]
            Variables to extract.
        levels : Optional[Sequence[int]]
            Pressure levels in hPa.
        save_zarr : bool, default True
            If True, writes consolidated Zarr store to disk and returns Path.
            If False, returns xr.Dataset.
        zarr_name : Optional[str]
            Optional custom name for output Zarr store.
        use_legacy : Optional[bool]
            If True, extracts via standard CDS API with adaptive chunking and NetCDF4 format.
            Defaults to True unless an ARCO backend is specified.
        use_cds_arco : Optional[bool]
            If True, extracts via Copernicus CDS analysis-ready Zarr stores using obstore.
        use_google_arco : Optional[bool]
            If True, extracts via Google Cloud Public Dataset ARCO-ERA5 Zarr store.
        data_format : Optional[str]
            Format for legacy CDS API downloads ('netcdf4' or 'grib'). Defaults to 'netcdf4'.
        chunk_days : int, default 5
            Starting temporal chunk window size in days for legacy CDS API queries.
        chunking : Optional[str]
            Manual chunking layout override for CDS ARCO ('geo' or 'time').
        stream : Optional[str]
            Manual stream override for CDS ARCO ('sfc', 'wav', or 'pl').
        cds_key : Optional[str]
            Copernicus CDS API key.
        cds_url : Optional[str]
            Copernicus CDS API endpoint URL.
        cds_arco_url : Optional[str]
            Remote CDS ARCO Zarr store URL.
        force_cds : Optional[bool]
            Backward compatibility flag forcing legacy CDS extraction.
        prefer_arco : Optional[bool]
            Backward compatibility flag prioritizing ARCO extraction.
        show_progress : bool, default True
            If True, displays a progress bar during extraction.
        resume : bool, default True
            If True, resumes from manifest-backed cache checkpoints when available.
        overwrite : bool, default False
            If True, ignores cached checkpoints and re-fetches all data.
        keep_cache : bool, default False
            If True, retains cache directory after successful extraction.
        dry_run : bool, default False
            If True, returns an ExtractionEstimate without making remote API calls.

        Returns
        -------
        Union[xr.Dataset, Path, ExtractionEstimate]
            Extracted Dataset, local Zarr store Path, or ExtractionEstimate if `dry_run=True`.
        """
        box = bbox if bbox is not None else AFRICA_BBOX
        t_range = self.validate_temporal_range(start_date, end_date)
        if variables is not None:
            self.validate_variables(variables, strict=True)
        self.validate_temporal_bounds(t_range, strict=False)
        s_dt, e_dt = parse_date_range(start_date, end_date)
        s_str = s_dt.strftime("%Y%m%d")
        e_str = e_dt.strftime("%Y%m%d")

        # Backend determination
        # Direct user selection in extract() or direct user selection in __init__()
        if force_cds is True or (force_cds is None and self.force_cds):
            chosen = "legacy"
        elif use_google_arco is True or prefer_arco is True:
            chosen = "google_arco"
        elif use_cds_arco is True:
            chosen = "cds_arco"
        elif use_legacy is True:
            chosen = "legacy"
        elif use_legacy is None and self._user_specified_legacy and self.use_legacy:
            chosen = "legacy"
        elif use_cds_arco is None and self.use_cds_arco:
            chosen = "cds_arco"
        elif use_google_arco is None and (self.use_google_arco or self.prefer_arco):
            chosen = "google_arco"
        else:
            chosen = "cascade"

        if dry_run:
            use_arco_flag = chosen in ("google_arco", "cds_arco", "cascade")
            return self.estimate_extraction(
                start_date=start_date,
                end_date=end_date,
                bbox=box,
                variables=variables,
                levels=levels,
                chunk_days=chunk_days,
                use_arco=use_arco_flag,
            )

        if save_zarr and zarr_name is None and s_dt.year != e_dt.year:
            last_zarr: Optional[Path] = None
            for yr in range(s_dt.year, e_dt.year + 1):
                y_start = max(s_dt, pd.Timestamp(f"{yr}-01-01"))
                y_end = min(e_dt, pd.Timestamp(f"{yr}-12-31 23:59:59"))
                last_zarr = self.extract(
                    start_date=y_start,
                    end_date=y_end,
                    bbox=box,
                    variables=variables,
                    levels=levels,
                    save_zarr=True,
                    zarr_name=f"{yr}.zarr",
                    use_legacy=use_legacy,
                    use_cds_arco=use_cds_arco,
                    use_google_arco=use_google_arco,
                    data_format=data_format,
                    force_cds=force_cds,
                    prefer_arco=prefer_arco,
                    chunk_days=chunk_days,
                    chunking=chunking,
                    stream=stream,
                    cds_key=cds_key,
                    cds_url=cds_url,
                    cds_arco_url=cds_arco_url,
                    show_progress=show_progress,
                    resume=resume,
                    overwrite=overwrite,
                    keep_cache=keep_cache,
                    dry_run=False,
                )
            return last_zarr if last_zarr is not None else self.output_dir / f"{s_dt.year}.zarr"

        effective_zarr_name = zarr_name or f"{s_dt.year}.zarr"
        append_existing_zarr = False
        if save_zarr and resume and not overwrite:
            existing_zarr_path = self.output_dir / effective_zarr_name
            cov = inspect_zarr_coverage(existing_zarr_path)
            if cov is not None:
                ex_min, ex_max, ex_vars, ex_bbox = cov
                vars_ok = True
                if variables is not None and variables != "all" and not (isinstance(variables, (list, tuple)) and "all" in variables):
                    for req_v in variables:
                        cands = get_era5_variable_candidates(req_v)
                        if not any(c in ex_vars for c in cands):
                            vars_ok = False
                            break
                if hasattr(box, "lat_min"):
                    b_lat_min, b_lon_min, b_lat_max, b_lon_max = box.lat_min, box.lon_min, box.lat_max, box.lon_max
                else:
                    b_lat_min, b_lon_min, b_lat_max, b_lon_max = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                bbox_ok = (
                    ex_bbox is None
                    or (
                        ex_bbox[0] <= b_lat_min + 0.3
                        and ex_bbox[1] <= b_lon_min + 0.3
                        and ex_bbox[2] >= b_lat_max - 0.3
                        and ex_bbox[3] >= b_lon_max - 0.3
                    )
                )
                if vars_ok and bbox_ok:
                    deltas, needs_rebuild = compute_missing_zarr_windows(s_dt, e_dt, ex_min, ex_max)
                    if not needs_rebuild and not deltas:
                        logger.info("Yearly ERA5 Zarr already covers requested period: %s", existing_zarr_path)
                        return existing_zarr_path
                    if not needs_rebuild and deltas:
                        start_date, end_date = deltas[0]
                        s_dt, e_dt = parse_date_range(start_date, end_date)
                        s_str = s_dt.strftime("%Y%m%d")
                        e_str = e_dt.strftime("%Y%m%d")
                        append_existing_zarr = True

        n_days_total = max(1, (e_dt - s_dt).days + 1)
        is_multi_chunk = n_days_total > max(1, int(chunk_days))
        use_cache = bool(save_zarr or is_multi_chunk)
        cache = (
            self.create_cache(
                self.output_dir,
                request_params={
                    "start_date": s_str,
                    "end_date": e_str,
                    "bbox": box,
                    "variables": list(variables) if variables is not None else None,
                    "levels": list(levels) if levels is not None else None,
                    "chosen": chosen,
                    "save_zarr": save_zarr,
                    "zarr_name": zarr_name,
                    "chunk_days": chunk_days,
                },
                resume=resume,
                overwrite=overwrite,
            )
            if use_cache
            else None
        )

        if cache is not None:
            if save_zarr and cache.is_done("zarr_output"):
                marker_path = cache.get_path("zarr_output")
                if marker_path is not None and marker_path.is_file():
                    target_zarr = Path(marker_path.read_text(encoding="utf-8").strip())
                    if target_zarr.exists():
                        cache.record_hit("zarr_output")
                        if not keep_cache:
                            cache.cleanup(keep_on_failure=True)
                        return target_zarr
            elif not save_zarr and cache.is_done("dataset"):
                cache.record_hit("dataset")
                ds_cached = cache.load_dataset("dataset")
                if ds_cached is not None and not keep_cache:
                    ds_cached = ds_cached.load()
                    cache.cleanup(keep_on_failure=True)
                return ds_cached

        progress_source = "era5_arco" if chosen in ("google_arco", "cds_arco", "cascade") else "era5"
        total_steps = 2 if save_zarr else 1

        def _finalize_zarr(ds_slice: xr.Dataset, default_name: str, pbar: Any) -> Path:
            pbar.set_postfix(phase="writing Zarr")
            name = effective_zarr_name
            zarr_path = self.output_dir / name
            target_chunks = {"time": min(48, max(1, ds_slice.sizes.get("time", 1))), "lat": -1, "lon": -1}
            if "level" in ds_slice.dims:
                target_chunks["level"] = ds_slice.sizes["level"]
            chunked = ds_slice.chunk(target_chunks)
            for var in chunked.variables:
                chunked[var].encoding.clear()
            if append_existing_zarr and zarr_path.exists():
                append_to_zarr(chunked, zarr_path, append_dim="time", chunks=target_chunks)
            else:
                chunked.to_zarr(zarr_path, mode="w", zarr_format=3)
            ds_slice.close()
            chunked.close()
            del ds_slice, chunked
            gc.collect()
            if cache is not None:
                cache.save_bytes("zarr_output", str(zarr_path).encode("utf-8"), suffix=".txt")
                if not keep_cache:
                    cache.cleanup(keep_on_failure=True)
            pbar.update(1)
            return zarr_path

        def _stream_legacy_files_to_zarr(
            downloaded_files: Sequence[Union[str, Path]],
            fmt_name: str,
            default_name: str,
            pbar: Any,
        ) -> Path:
            pbar.set_postfix(phase="writing Zarr")
            name = effective_zarr_name
            zarr_path = self.output_dir / name
            is_nc_files = bool(downloaded_files) and all(str(p).lower().endswith((".nc", ".nc4")) for p in downloaded_files)
            opener = open_era5_netcdf if (fmt_name in ("netcdf4", "netcdf", "nc") or is_nc_files) else open_era5_grib

            if len(downloaded_files) > 1 or (append_existing_zarr and zarr_path.exists()):
                if not append_existing_zarr and zarr_path.exists():
                    shutil.rmtree(zarr_path, ignore_errors=True)
                for chunk_file in downloaded_files:
                    ds_chunk = opener([chunk_file], harmonize=True)
                    for var in ds_chunk.variables:
                        ds_chunk[var].encoding.clear()
                    target_chunks = {"time": min(48, max(1, ds_chunk.sizes.get("time", 1))), "lat": -1, "lon": -1}
                    if "level" in ds_chunk.dims:
                        target_chunks["level"] = ds_chunk.sizes["level"]
                    append_to_zarr(ds_chunk, zarr_path, append_dim="time", chunks=target_chunks)
                    ds_chunk.close()
                    del ds_chunk
                    gc.collect()
            else:
                combined_ds = opener(downloaded_files, harmonize=True)
                combined_ds.to_zarr(zarr_path, mode="w", zarr_format=3)
                combined_ds.close()
                del combined_ds
                gc.collect()

            if cache is not None:
                cache.save_bytes("zarr_output", str(zarr_path).encode("utf-8"), suffix=".txt")
                if not keep_cache:
                    cache.cleanup(keep_on_failure=True)
            pbar.update(1)
            return zarr_path

        def _finalize_dataset(ds_slice: xr.Dataset) -> xr.Dataset:
            if cache is not None:
                cache.save_dataset("dataset", ds_slice, fmt="netcdf")
                if not keep_cache:
                    cache.cleanup(keep_on_failure=True)
            return ds_slice

        with make_progress_bar(
            total=total_steps,
            source=progress_source,
            unit="chunk",
            show_progress=show_progress,
        ) as pbar:
            pbar.set_postfix(phase="queued")

            # 0. Priority Cascade: Google ARCO -> CDS Legacy -> CDS ARCO
            if chosen == "cascade":
                logger.info("Executing ERA5 extraction via Priority Cascade (Google ARCO -> CDS Legacy -> CDS ARCO)...")
                arco_available = False
                try:
                    arco_available = self.check_google_arco_availability(
                        start_date=start_date,
                        end_date=end_date,
                        variables=variables,
                        bbox=box,
                    )
                except Exception as check_err:
                    logger.info("Google ARCO availability check encountered error: %s", check_err)
                    arco_available = False

                if arco_available:
                    try:
                        logger.info("Priority cascade [1/3]: Extracting from Google Cloud ARCO...")
                        pbar.set_postfix(phase="downloading", backend="google_arco")
                        ds_slice = self.extract_google_arco_slice(
                            bbox=box,
                            start_date=start_date,
                            end_date=end_date,
                            variables=variables,
                            levels=levels,
                        )
                        pbar.update(1)
                        if save_zarr:
                            return _finalize_zarr(ds_slice, f"era5_google_arco_{s_str}_{e_str}.zarr", pbar)
                        return _finalize_dataset(ds_slice)
                    except Exception as arco_err:
                        logger.warning(
                            "Google ARCO extraction failed despite availability check: %s. Falling back to CDS Legacy...",
                            arco_err,
                        )
                else:
                    logger.info(
                        "Google ARCO ERA5 is unavailable or lagging for requested range [%s, %s]. Falling back to CDS Legacy...",
                        s_str, e_str,
                    )

                # Fallback 1: CDS Legacy API
                cds_legacy_err_msg = ""
                try:
                    fmt = (data_format or self.data_format or "grib").lower().strip()
                    logger.info("Priority cascade [2/3]: Attempting CDS Legacy API extraction (format: %s)...", fmt)
                    pbar.set_postfix(phase="downloading", backend="cds_legacy")
                    downloaded = self.extract_cds_slice(
                        bbox=box,
                        start_date=start_date,
                        end_date=end_date,
                        variables=variables,
                        n_days_per_chunk=chunk_days,
                        data_format=fmt,
                        cds_key=cds_key or self.cds_key,
                        cds_url=cds_url or self.cds_url,
                    )
                    if save_zarr:
                        pbar.update(1)
                        return _stream_legacy_files_to_zarr(downloaded, fmt, f"era5_legacy_{s_str}_{e_str}.zarr", pbar)

                    is_nc = bool(downloaded) and all(str(p).lower().endswith((".nc", ".nc4")) for p in downloaded)
                    if fmt in ("netcdf4", "netcdf", "nc") or is_nc:
                        combined = open_era5_netcdf(downloaded, harmonize=True)
                    else:
                        combined = open_era5_grib(downloaded, harmonize=True)
                    pbar.update(1)
                    return _finalize_dataset(combined)
                except Exception as legacy_err:
                    cds_legacy_err_msg = str(legacy_err)
                    logger.warning(
                        "CDS Legacy API extraction failed: %s. Falling back to CDS ARCO...",
                        legacy_err,
                    )

                # Fallback 2: CDS ARCO via obstore
                try:
                    logger.info("Priority cascade [3/3]: Attempting CDS ARCO Zarr extraction...")
                    pbar.set_postfix(phase="downloading", backend="cds_arco")
                    ds_slice = self.extract_cds_arco_slice(
                        bbox=box,
                        start_date=start_date,
                        end_date=end_date,
                        variables=variables,
                        levels=levels,
                        stream=stream or self.stream,
                        chunking=chunking or self.chunking,
                        url=cds_arco_url or self.cds_arco_url,
                        cds_key=cds_key or self.cds_key,
                    )
                    pbar.update(1)
                    if save_zarr:
                        return _finalize_zarr(ds_slice, f"era5_cds_arco_{s_str}_{e_str}.zarr", pbar)
                    return _finalize_dataset(ds_slice)
                except Exception as arco_err:
                    if cache is not None:
                        cache.mark_failed("cascade", arco_err)
                    logger.error(
                        "All ERA5 backends in priority cascade failed. Google ARCO: unavailable; CDS Legacy: %s; CDS ARCO: %s",
                        cds_legacy_err_msg, arco_err,
                    )
                    raise RuntimeError(
                        f"All ERA5 extraction backends failed in priority cascade. "
                        f"Google ARCO: unavailable or failed; "
                        f"CDS Legacy: {cds_legacy_err_msg}; "
                        f"CDS ARCO: {arco_err}"
                    ) from arco_err

            # 1. CDS ARCO Backend
            if chosen == "cds_arco":
                logger.info("Executing ERA5 extraction via CDS ARCO Zarr backend...")
                pbar.set_postfix(phase="downloading", backend="cds_arco")
                ds_slice = self.extract_cds_arco_slice(
                    bbox=box,
                    start_date=start_date,
                    end_date=end_date,
                    variables=variables,
                    levels=levels,
                    stream=stream or self.stream,
                    chunking=chunking or self.chunking,
                    url=cds_arco_url or self.cds_arco_url,
                    cds_key=cds_key or self.cds_key,
                )
                pbar.update(1)
                if save_zarr:
                    return _finalize_zarr(ds_slice, f"era5_cds_arco_{s_str}_{e_str}.zarr", pbar)
                return _finalize_dataset(ds_slice)

            # 2. Google ARCO Backend
            if chosen == "google_arco":
                logger.info("Executing ERA5 extraction via Google Cloud ARCO backend...")
                pbar.set_postfix(phase="downloading", backend="google_arco")
                ds_slice = self.extract_google_arco_slice(
                    bbox=box,
                    start_date=start_date,
                    end_date=end_date,
                    variables=variables,
                    levels=levels,
                )
                pbar.update(1)
                if save_zarr:
                    return _finalize_zarr(ds_slice, f"era5_google_arco_{s_str}_{e_str}.zarr", pbar)
                return _finalize_dataset(ds_slice)

            # 3. Legacy CDS API Backend
            fmt = (data_format or self.data_format or "grib").lower().strip()
            logger.info("Executing ERA5 extraction via legacy CDS API backend (format: %s)...", fmt)
            pbar.set_postfix(phase="downloading", backend="cds_legacy")
            downloaded = self.extract_cds_slice(
                bbox=box,
                start_date=start_date,
                end_date=end_date,
                variables=variables,
                n_days_per_chunk=chunk_days,
                data_format=fmt,
                cds_key=cds_key or self.cds_key,
                cds_url=cds_url or self.cds_url,
            )

            try:
                if save_zarr:
                    pbar.update(1)
                    return _stream_legacy_files_to_zarr(downloaded, fmt, f"era5_legacy_{s_str}_{e_str}.zarr", pbar)

                is_nc = bool(downloaded) and all(str(p).lower().endswith((".nc", ".nc4")) for p in downloaded)
                if fmt in ("netcdf4", "netcdf", "nc") or is_nc:
                    combined = open_era5_netcdf(downloaded, harmonize=True)
                else:
                    combined = open_era5_grib(downloaded, harmonize=True)
                pbar.update(1)
                return _finalize_dataset(combined)
            except Exception as err:
                if cache is not None:
                    cache.mark_failed("legacy", err)
                raise RuntimeError(
                    f"Failed to decode legacy ERA5 files downloaded to {self.output_dir}: {err}"
                ) from err

