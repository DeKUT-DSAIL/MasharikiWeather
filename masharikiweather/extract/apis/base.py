"""
Base Extraction Interfaces, Spatial/Temporal Validators, and CRS Compliance Engines.

Defines standardized data classes, abstract extraction contracts, high-throughput
vectorized validators, and CF-1.8+ / WGS 84 (EPSG:4326) CRS enforcement utilities
for MasharikiWeather.
"""

from __future__ import annotations

import calendar
import difflib
import logging
import os
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    pyproj = None
    HAS_PYPROJ = False

try:
    import rioxarray
    HAS_RIOXARRAY = True
except ImportError:
    rioxarray = None
    HAS_RIOXARRAY = False

logger = logging.getLogger("MasharikiWeather.Extract.Base")


# =====================================================================
# 1. Validation Exception Hierarchy
# =====================================================================

class ExtractionValidationError(ValueError):
    """Base exception for all extraction and ingestion validation failures."""
    pass


class VariableNotAvailableError(ExtractionValidationError):
    """Raised when a requested meteorological variable is unavailable in the extractor."""
    pass


class StationNotFoundError(ExtractionValidationError):
    """Raised when requested station identifiers are invalid or cross-network mismatched."""
    pass


class TemporalCoverageError(ExtractionValidationError):
    """Raised when a requested temporal range falls outside operational or station bounds."""
    pass


class SpatialCoverageError(ExtractionValidationError):
    """Raised when requested spatial bounds fall outside dataset geographic domains."""
    pass


class CRSComplianceError(ExtractionValidationError):
    """Raised when a dataset violates CF grid_mapping conventions or EPSG:4326 standards."""
    pass


# =====================================================================
# 2. Validation Result Containers
# =====================================================================

@dataclass(frozen=True)
class VariableValidationResult:
    """
    Container capturing the outcome of variable availability validation.

    Mathematical docstring:
    Input variables: Sequence of length V_req.
    Output:
      valid_variables: List of length V_valid <= V_req.
      invalid_variables: List of length V_invalid = V_req - V_valid.
    """

    is_valid: bool
    requested_variables: List[str]
    valid_variables: List[str]
    invalid_variables: List[str]
    suggestions: Dict[str, List[str]] = field(default_factory=dict)
    cross_extractor_alternatives: Dict[str, List[str]] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class StationValidationResult:
    """
    Container capturing the outcome of station identifier validation.

    Mathematical docstring:
    Input stations: Sequence of length N_req.
    Output:
      valid_stations: List of length N_valid <= N_req.
      invalid_stations: List of length N_invalid = N_req - N_valid.
    """

    is_valid: bool
    requested_stations: List[str]
    valid_stations: List[str]
    invalid_stations: List[str]
    mismatched_network_stations: Dict[str, str] = field(default_factory=dict)
    suggestions: Dict[str, List[str]] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TemporalValidationResult:
    """
    Container capturing the outcome of temporal coverage validation.

    Mathematical docstring:
    Input temporal range: [T_start, T_end].
    Station catalog: (N_stations, C_metadata).
    Output:
      coverage_fractions: Dict mapping station_id to float in [0.0, 1.0].
      overlap_seconds: Integer seconds of active temporal overlap.
    """

    is_valid: bool
    requested_range: TemporalRange
    active_stations: List[str]
    inactive_stations: List[str]
    pre_commission_stations: List[str] = field(default_factory=list)
    post_decommission_stations: List[str] = field(default_factory=list)
    coverage_fractions: Dict[str, float] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CRSValidationResult:
    """
    Container capturing the outcome of CRS compliance validation.

    Attributes:
    is_compliant: True if CF grid_mapping and coordinate bounds obey target CRS.
    epsg_code: Detected or verified EPSG integer code (e.g. 4326).
    crs_name: Geographic CRS description string.
    spatial_dims: Tuple of (latitude_dim, longitude_dim).
    lat_bounds: Detected min/max latitude tuple.
    lon_bounds: Detected min/max longitude tuple.
    has_grid_mapping: True if CF grid_mapping variable and attributes are present.
    issues: List of non-compliance issues encountered.
    """

    is_compliant: bool
    epsg_code: Optional[int]
    crs_name: Optional[str]
    spatial_dims: Tuple[Optional[str], Optional[str]]
    lat_bounds: Optional[Tuple[float, float]] = None
    lon_bounds: Optional[Tuple[float, float]] = None
    has_grid_mapping: bool = False
    issues: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Allow CRSValidationResult to be evaluated directly as a boolean in boolean contexts."""
        return self.is_compliant


# =====================================================================
# 3. Spatial and Temporal Containers
# =====================================================================

@dataclass(frozen=True)
class SpatialBoundingBox:
    """
    Standard spatial bounding box container with coordinate sanity enforcement.

    Parameters
    ----------
    lat_min : float
        Southernmost latitude in degrees north [-90.0, 90.0].
    lat_max : float
        Northernmost latitude in degrees north [-90.0, 90.0].
    lon_min : float
        Westernmost longitude in degrees east [-180.0, 180.0].
    lon_max : float
        Easternmost longitude in degrees east [-180.0, 180.0].
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.lat_min <= 90.0):
            raise ValueError(f"lat_min out of bounds [-90, 90]: {self.lat_min}")
        if not (-90.0 <= self.lat_max <= 90.0):
            raise ValueError(f"lat_max out of bounds [-90, 90]: {self.lat_max}")
        if self.lat_min > self.lat_max:
            raise ValueError(f"lat_min ({self.lat_min}) cannot exceed lat_max ({self.lat_max})")

        if not (-180.0 <= self.lon_min <= 180.0):
            raise ValueError(f"lon_min out of bounds [-180, 180]: {self.lon_min}")
        if not (-180.0 <= self.lon_max <= 180.0):
            raise ValueError(f"lon_max out of bounds [-180, 180]: {self.lon_max}")
        if self.lon_min > self.lon_max:
            raise ValueError(f"lon_min ({self.lon_min}) cannot exceed lon_max ({self.lon_max})")

    @classmethod
    def from_tuple(cls, bounds: Sequence[float], format_order: str = "min_max") -> SpatialBoundingBox:
        """
        Construct bounding box from numeric sequence.

        Supported formats:
        - 'min_max': [lat_min, lat_max, lon_min, lon_max]
        - 'north_west_south_east': [lat_max, lon_min, lat_min, lon_max] (CDS/ARCO order)
        - 'west_south_east_north': [lon_min, lat_min, lon_max, lat_max] (GIS/Earthdata order)
        """
        if len(bounds) != 4:
            raise ValueError(f"Expected 4 coordinate values, received {len(bounds)}")

        if format_order == "min_max":
            return cls(lat_min=float(bounds[0]), lat_max=float(bounds[1]),
                       lon_min=float(bounds[2]), lon_max=float(bounds[3]))
        elif format_order == "north_west_south_east":
            return cls(lat_max=float(bounds[0]), lon_min=float(bounds[1]),
                       lat_min=float(bounds[2]), lon_max=float(bounds[3]))
        elif format_order == "west_south_east_north":
            return cls(lon_min=float(bounds[0]), lat_min=float(bounds[1]),
                       lon_max=float(bounds[2]), lat_max=float(bounds[3]))
        else:
            raise ValueError(f"Unknown format_order: {format_order}")

    def to_area_list(self) -> List[float]:
        """
        Return coordinates in CDS / ARCO format: [North, West, South, East].
        """
        return [self.lat_max, self.lon_min, self.lat_min, self.lon_max]

    def to_area_str_list(self) -> List[str]:
        """
        Return coordinates in CDS string format: ['North', 'West', 'South', 'East'].
        """
        return [str(self.lat_max), str(self.lon_min), str(self.lat_min), str(self.lon_max)]

    def to_bbox_tuple(self) -> Tuple[float, float, float, float]:
        """
        Return coordinates in GIS / Earthdata format: (West, South, East, North).
        """
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)

    def to_dict(self) -> Dict[str, float]:
        """
        Return dictionary representation.
        """
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }

    def contains(self, lat: float, lon: float) -> bool:
        """
        Check whether given point resides within bounding box.
        """
        return (self.lat_min <= lat <= self.lat_max) and (self.lon_min <= lon <= self.lon_max)


def synthetic_fallback_allowed() -> bool:
    """
    Return True only when the environment variable MW_ALLOW_SYNTHETIC_FALLBACK is set
    to a truthy value. Climatological stand-in values are never produced otherwise.
    """
    return os.environ.get("MW_ALLOW_SYNTHETIC_FALLBACK", "").strip().lower() in ("1", "true", "yes")


# Canonical preset regions for Africa and continent
AFRICA_BBOX = SpatialBoundingBox(lat_min=-35.0, lat_max=38.0, lon_min=-20.0, lon_max=55.0)
EAST_AFRICA_BBOX = SpatialBoundingBox(lat_min=-12.0, lat_max=15.0, lon_min=28.0, lon_max=52.0)
KENYA_BBOX = SpatialBoundingBox(lat_min=-4.7, lat_max=5.5, lon_min=33.8, lon_max=41.9)
PAN_AFRICA_BBOX = AFRICA_BBOX


@dataclass(frozen=True)
class TemporalRange:
    """
    Standard temporal range container with validation and chunking utilities.

    Parameters
    ----------
    start : pd.Timestamp
        Start timestamp (inclusive).
    end : pd.Timestamp
        End timestamp (inclusive).
    """

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if not isinstance(self.start, pd.Timestamp):
            s = pd.to_datetime(self.start)
            if s.tzinfo is not None:
                s = s.tz_localize(None)
            object.__setattr__(self, "start", s)
        if not isinstance(self.end, pd.Timestamp):
            e = pd.to_datetime(self.end)
            if e.tzinfo is not None:
                e = e.tz_localize(None)
            object.__setattr__(self, "end", e)
        if self.start > self.end:
            raise ValueError(f"start date ({self.start}) cannot exceed end date ({self.end})")

    @classmethod
    def from_dates(
        cls,
        start_date: Union[str, datetime, date, pd.Timestamp],
        end_date: Union[str, datetime, date, pd.Timestamp],
    ) -> TemporalRange:
        """
        Construct TemporalRange from strings or datetime objects.
        """
        s = pd.to_datetime(start_date).tz_localize(None) if pd.to_datetime(start_date).tzinfo is not None else pd.to_datetime(start_date)
        e = pd.to_datetime(end_date).tz_localize(None) if pd.to_datetime(end_date).tzinfo is not None else pd.to_datetime(end_date)
        return cls(start=s, end=e)

    def to_iso_tuple(self) -> Tuple[str, str]:
        """
        Return ISO format strings (YYYY-MM-DDTHH:MM:SS).
        """
        return (self.start.isoformat(), self.end.isoformat())

    def to_date_str_tuple(self) -> Tuple[str, str]:
        """
        Return date strings (YYYY-MM-DD).
        """
        return (self.start.strftime("%Y-%m-%d"), self.end.strftime("%Y-%m-%d"))

    def years(self) -> List[int]:
        """
        Return unique years covering range.
        """
        return list(range(self.start.year, self.end.year + 1))

    def split_by_month(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Split temporal range into monthly sub-windows [month_start, month_end].
        Properly handles month boundaries, year boundaries, and leap years (Feb 28 vs Feb 29).
        """
        windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        cur = self.start
        while cur <= self.end:
            year = cur.year
            month = cur.month
            _, num_days = calendar.monthrange(year, month)
            month_end = min(pd.Timestamp(datetime(year, month, num_days, 23, 59, 59)), self.end)
            windows.append((cur, month_end))
            next_start = pd.Timestamp(datetime(year, month, num_days)) + timedelta(days=1)
            cur = next_start
        return windows

    def split_by_days(self, n_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Split temporal range into discrete n-day blocks.
        """
        if n_days < 1:
            raise ValueError(f"n_days must be >= 1, received {n_days}")
        windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        cur = self.start
        delta = timedelta(days=n_days)
        while cur <= self.end:
            chunk_end = min(cur + delta - timedelta(seconds=1), self.end)
            windows.append((cur, chunk_end))
            cur = chunk_end + timedelta(seconds=1)
        return windows


# =====================================================================
# 4. Global Registries & Catalogs
# =====================================================================

WGS84_WKT: str = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
)

WGS84_CF_ATTRIBUTES: Dict[str, Any] = {
    "grid_mapping_name": "latitude_longitude",
    "epsg_code": "EPSG:4326",
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.314245179,
    "inverse_flattening": 298.257223563,
    "reference_ellipsoid_name": "WGS 84",
    "prime_meridian_name": "Greenwich",
    "longitude_of_prime_meridian": 0.0,
    "geographic_crs_name": "WGS 84",
    "horizontal_datum_name": "World Geodetic System 1984",
    "spatial_ref": WGS84_WKT,
    "crs_wkt": WGS84_WKT,
}

GLOBAL_EXTRACTOR_VARIABLE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "imerg": {
        "description": "NASA GPM IMERG Half-Hourly & Daily Satellite Precipitation",
        "category": "satellite_precipitation",
        "canonical_variables": [
            "total_precipitation",
            "uncalibrated_precipitation",
            "precipitation_error",
            "high_quality_precipitation",
            "probability_liquid_precipitation",
        ],
        "aliases": {
            "precipitationcal": "total_precipitation",
            "precipitationuncal": "uncalibrated_precipitation",
            "randomerror": "precipitation_error",
            "hqprecipitation": "high_quality_precipitation",
            "probabilityliquidprecipitation": "probability_liquid_precipitation",
            "precipitation": "total_precipitation",
            "precip": "total_precipitation",
            "prcp": "total_precipitation",
            "pr": "total_precipitation",
            "tp": "total_precipitation",
        },
    },
    "era5": {
        "description": "ECMWF ERA5 Reanalysis (Single Levels & Pressure Levels)",
        "category": "reanalysis",
        "canonical_variables": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
            "total_precipitation",
            "surface_pressure",
            "mean_sea_level_pressure",
            "relative_humidity",
            "total_cloud_cover",
            "high_cloud_cover",
            "medium_cloud_cover",
            "low_cloud_cover",
            "surface_solar_radiation_downwards",
            "toa_incident_solar_radiation",
            "convective_precipitation",
            "large_scale_precipitation",
            "convective_available_potential_energy",
            "convective_inhibition",
            "total_column_water_vapour",
            "volumetric_soil_water_layer_1",
        ],
        "aliases": {
            "t2m": "2m_temperature",
            "2t": "2m_temperature",
            "temperature": "2m_temperature",
            "temp": "2m_temperature",
            "d2m": "2m_dewpoint_temperature",
            "2d": "2m_dewpoint_temperature",
            "dewpoint": "2m_dewpoint_temperature",
            "u10": "10m_u_component_of_wind",
            "v10": "10m_v_component_of_wind",
            "tp": "total_precipitation",
            "sp": "surface_pressure",
            "msl": "mean_sea_level_pressure",
            "mslp": "mean_sea_level_pressure",
            "rh": "relative_humidity",
            "tcc": "total_cloud_cover",
            "ssrd": "surface_solar_radiation_downwards",
            "cape": "convective_available_potential_energy",
            "cin": "convective_inhibition",
        },
    },
    "tahmo": {
        "description": "Trans-African Hydro-Meteorological Observatory (TAHMO) Ground Stations",
        "category": "in_situ",
        "canonical_variables": [
            "2m_temperature",
            "relative_humidity",
            "total_precipitation",
            "wind_speed",
            "wind_from_direction",
            "surface_solar_radiation_downwards",
            "surface_pressure",
        ],
        "aliases": {
            "te": "2m_temperature",
            "temp": "2m_temperature",
            "temperature": "2m_temperature",
            "rh": "relative_humidity",
            "pr": "total_precipitation",
            "precip": "total_precipitation",
            "ws": "wind_speed",
            "wd": "wind_from_direction",
            "sr": "surface_solar_radiation_downwards",
            "rad": "surface_solar_radiation_downwards",
            "ap": "surface_pressure",
            "pres": "surface_pressure",
        },
    },
    "ghcnd": {
        "description": "NOAA Global Historical Climatology Network (GHCN-Daily & GHCN-Hourly)",
        "category": "in_situ",
        "canonical_variables": [
            "2m_temperature",
            "2m_temperature_maximum",
            "2m_temperature_minimum",
            "total_precipitation",
            "snowfall",
            "snow_depth",
        ],
        "aliases": {
            "t2m_max": "2m_temperature_maximum",
            "t2m_min": "2m_temperature_minimum",
            "t2m_mean": "2m_temperature",
            "tmax": "2m_temperature_maximum",
            "tmin": "2m_temperature_minimum",
            "tavg": "2m_temperature",
            "temperature": "2m_temperature",
            "precipitation": "total_precipitation",
            "prcp": "total_precipitation",
            "snow": "snowfall",
            "snwd": "snow_depth",
        },
    },
    "cds_insitu": {
        "description": "Copernicus Climate Data Store In-Situ Surface Observations",
        "category": "in_situ",
        "canonical_variables": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "total_precipitation",
            "surface_pressure",
            "mean_sea_level_pressure",
            "relative_humidity",
            "wind_speed",
            "wind_from_direction",
            "snowfall",
            "snow_depth",
        ],
        "aliases": {
            "air_temperature": "2m_temperature",
            "dew_point_temperature": "2m_dewpoint_temperature",
            "precipitation_amount": "total_precipitation",
            "surface_air_pressure": "surface_pressure",
            "air_pressure_at_mean_sea_level": "mean_sea_level_pressure",
            "temperature": "2m_temperature",
            "dewpoint": "2m_dewpoint_temperature",
            "precipitation": "total_precipitation",
            "pressure": "surface_pressure",
        },
    },
}

# Construct inverted index mapping variable names/aliases to provider extractors
CANONICAL_VARIABLE_EXTRACTOR_MAP: Dict[str, List[str]] = {}
for _ext_name, _ext_info in GLOBAL_EXTRACTOR_VARIABLE_REGISTRY.items():
    for _can_var in _ext_info["canonical_variables"]:
        CANONICAL_VARIABLE_EXTRACTOR_MAP.setdefault(_can_var, []).append(_ext_name)
    for _alias, _can_var in _ext_info.get("aliases", {}).items():
        if _alias not in CANONICAL_VARIABLE_EXTRACTOR_MAP:
            CANONICAL_VARIABLE_EXTRACTOR_MAP[_alias] = []
        if _ext_name not in CANONICAL_VARIABLE_EXTRACTOR_MAP[_alias]:
            CANONICAL_VARIABLE_EXTRACTOR_MAP[_alias].append(_ext_name)

# Operational temporal bounds for known extractors
EXTRACTOR_OPERATIONAL_BOUNDS: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {
    "era5": (pd.Timestamp("1940-01-01"), pd.Timestamp.now() - pd.Timedelta(days=5)),
    "imerg": (pd.Timestamp("2000-06-01"), pd.Timestamp.now()),
    "tahmo": (pd.Timestamp("2014-01-01"), pd.Timestamp.now()),
    "ghcnd": (pd.Timestamp("1880-01-01"), pd.Timestamp.now()),
    "cds_insitu": (pd.Timestamp("1900-01-01"), pd.Timestamp.now()),
}

# Station network pattern heuristics
STATION_NETWORK_PATTERNS: Dict[str, Dict[str, Any]] = {
    "tahmo": {
        "regex": re.compile(r"^TA\d{4,6}$", re.IGNORECASE),
        "description": "TAHMO weather station (format: TA#####)",
        "extractor": "tahmo",
        "sample": "TA00185",
    },
    "ghcn": {
        "regex": re.compile(r"^[A-Z]{2}[A-Z0-9]{9}$"),
        "description": "NOAA GHCN 11-character station identifier",
        "extractor": "ghcnd",
        "sample": "KE000063740",
    },
    "cds_insitu": {
        "regex": re.compile(r"^(?:\d{5}|\d+-\d+-\d+-\w+)$"),
        "description": "WMO 5-digit index or WIGOS station identifier",
        "extractor": "cds_insitu",
        "sample": "63740",
    },
    "icao_kmd": {
        "regex": re.compile(r"^[A-Z]{4}$"),
        "description": "ICAO airport code / Kenya Meteorological Department synoptic station",
        "extractor": "ghcnd",
        "sample": "KEMU",
    },
}


# =====================================================================
# 5. CRS Compliance and Spatial Standardization
# =====================================================================

def find_spatial_dimensions(
    ds: Union[xr.Dataset, xr.DataArray]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Identify latitude and longitude dimension or coordinate names in an xarray object.

    Mathematical docstring:
    Input ds: Dataset or DataArray with arbitrary coordinate labels.
    Output: (lat_name, lon_name) strings if identified, else (None, None).
    """
    lat_candidates = ("latitude", "lat", "y", "LAT", "LATITUDE", "Y")
    lon_candidates = ("longitude", "lon", "long", "x", "LON", "LONGITUDE", "X")

    lat_dim: Optional[str] = None
    lon_dim: Optional[str] = None

    for candidate in lat_candidates:
        if candidate in ds.dims or candidate in ds.coords:
            lat_dim = candidate
            break

    for candidate in lon_candidates:
        if candidate in ds.dims or candidate in ds.coords:
            lon_dim = candidate
            break

    return lat_dim, lon_dim


def enforce_wgs84_crs(
    ds: xr.Dataset,
    standardize_names: bool = True,
    wrap_lon_to_180: bool = True,
    grid_mapping_name: str = "spatial_ref",
) -> xr.Dataset:
    """
    Enforce WGS 84 (EPSG:4326) CRS compliance in an xarray.Dataset for gridded products.

    Operations:
    1. Detects spatial dimensions and coordinates (latitude, longitude).
    2. Standardizes dimension names to ('latitude', 'longitude') if standardize_names=True.
    3. Normalizes longitude coordinates from [0, 360) to [-180, 180] degrees east if wrap_lon_to_180=True,
       preserving Dask chunking lazily and maintaining monotonic ordering.
    4. Enforces CF-compliant standard attributes (standard_name, units, axis) on coordinates.
    5. Injects CF grid_mapping container variable ('spatial_ref') with complete OGC WKT parameters.
    6. Attaches 'grid_mapping' attribute to all multi-dimensional spatial data variables.
    7. Employs rioxarray write_crs if rioxarray is installed; falls back to pyproj or native CF attributes.

    Mathematical docstring:
    Input ds: Dataset of shape (..., N_lat, N_lon) with coordinates in degrees.
    Output ds: Dataset of shape (..., N_lat, N_lon) with standard WGS 84 CF grid_mapping.
    Complexity: O(N_lat + N_lon) metadata and coordinate alignment; lazy dask evaluation preserved.

    Parameters
    ----------
    ds : xr.Dataset
        Input gridded weather dataset (ERA5, IMERG, etc.).
    standardize_names : bool, default True
        Whether to rename lat/lon dimensions to 'latitude' and 'longitude'.
    wrap_lon_to_180 : bool, default True
        Whether to wrap longitudes > 180 to [-180, 180] degrees east.
    grid_mapping_name : str, default 'spatial_ref'
        Name of the CF coordinate reference variable to inject.

    Returns
    -------
    xr.Dataset
        CRS-compliant, immutable xarray Dataset.
    """
    ds_out = ds.copy(deep=False)
    lat_dim, lon_dim = find_spatial_dimensions(ds_out)

    if lat_dim is None or lon_dim is None:
        logger.warning("Could not identify spatial dimensions for CRS enforcement; leaving dataset unchanged.")
        return ds_out

    # 1. Standardize coordinate names if requested
    rename_dict: Dict[str, str] = {}
    if standardize_names:
        if lat_dim != "latitude":
            rename_dict[lat_dim] = "latitude"
        if lon_dim != "longitude":
            rename_dict[lon_dim] = "longitude"
        if rename_dict:
            ds_out = ds_out.rename(rename_dict)
            lat_dim = "latitude"
            lon_dim = "longitude"

    # 2. Normalize longitude from [0, 360) to [-180, 180]
    lon_vals = ds_out[lon_dim].values
    if wrap_lon_to_180 and np.any(lon_vals > 180.0):
        wrapped = xr.where(ds_out[lon_dim] > 180.0, ds_out[lon_dim] - 360.0, ds_out[lon_dim])
        ds_out = ds_out.assign_coords({lon_dim: wrapped})
        # Re-sort monotonically if wrapping disrupted monotonicity
        if not bool(np.all(np.diff(ds_out[lon_dim].values) >= 0)):
            ds_out = ds_out.sortby(lon_dim)

    # 3. Coordinate sanity verification
    min_lat = float(ds_out[lat_dim].min().values)
    max_lat = float(ds_out[lat_dim].max().values)
    min_lon = float(ds_out[lon_dim].min().values)
    max_lon = float(ds_out[lon_dim].max().values)

    if not (-90.001 <= min_lat <= 90.001 and -90.001 <= max_lat <= 90.001):
        raise CRSComplianceError(f"Latitude out of physical range [-90, 90]: [{min_lat}, {max_lat}]")
    if not (-180.001 <= min_lon <= 180.001 and -180.001 <= max_lon <= 180.001):
        raise CRSComplianceError(f"Longitude out of standard range [-180, 180]: [{min_lon}, {max_lon}]")

    # 4. Enforce CF coordinate attributes
    ds_out[lat_dim].attrs.update({
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
        "axis": "Y",
    })
    ds_out[lon_dim].attrs.update({
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
    })

    # 5. Build CF grid_mapping attributes (Multi-tier: pyproj -> static dictionary)
    cf_attrs: Dict[str, Any] = dict(WGS84_CF_ATTRIBUTES)
    if HAS_PYPROJ:
        try:
            proj_crs = pyproj.CRS.from_epsg(4326)
            cf_attrs = proj_crs.to_cf()
            cf_attrs["epsg_code"] = "EPSG:4326"
            cf_attrs["spatial_ref"] = proj_crs.to_wkt()
            cf_attrs["crs_wkt"] = proj_crs.to_wkt()
        except Exception as exc:
            logger.debug("pyproj CRS generation fallback: %s", exc)

    # 6. Inject CF grid mapping scalar container
    ds_out[grid_mapping_name] = xr.DataArray(np.int32(0), attrs=cf_attrs)

    # 7. Tag all multi-dimensional spatial data variables with grid_mapping
    for var_name, da in ds_out.data_vars.items():
        if var_name != grid_mapping_name:
            if lat_dim in da.dims and lon_dim in da.dims:
                da.attrs["grid_mapping"] = grid_mapping_name

    # 8. rioxarray integration if available
    if HAS_RIOXARRAY:
        try:
            ds_out = ds_out.rio.write_crs("EPSG:4326", inplace=False)
            ds_out = ds_out.rio.set_spatial_dims(x_dim=lon_dim, y_dim=lat_dim, inplace=False)
        except Exception as exc:
            logger.debug("rioxarray write_crs skipped: %s", exc)

    return ds_out


def validate_crs_compliance(
    ds: Union[xr.Dataset, xr.DataArray],
    target_epsg: int = 4326,
    strict: bool = False,
) -> CRSValidationResult:
    """
    Validate that an xarray Dataset conforms to WGS 84 (EPSG:4326) CF conventions.

    Mathematical docstring:
    Input ds: Dataset or DataArray with spatial dimensions.
    Output: CRSValidationResult capturing compliance status, coordinate bounds, and diagnostic issues.

    Parameters
    ----------
    ds : Union[xr.Dataset, xr.DataArray]
        Dataset or DataArray to validate.
    target_epsg : int, default 4326
        Target EPSG coordinate reference code.
    strict : bool, default False
        If True, raises CRSComplianceError on non-compliance.

    Returns
    -------
    CRSValidationResult
        Structured validation outcome.
    """
    issues: List[str] = []
    lat_dim, lon_dim = find_spatial_dimensions(ds)

    if lat_dim is None or lon_dim is None:
        issues.append("Missing recognized spatial dimensions (latitude/lat/y and longitude/lon/x).")
        result = CRSValidationResult(
            is_compliant=False,
            epsg_code=None,
            crs_name=None,
            spatial_dims=(lat_dim, lon_dim),
            issues=issues,
        )
        if strict:
            raise CRSComplianceError(f"CRS non-compliant: {issues}")
        return result

    lat_vals = ds[lat_dim].values
    lon_vals = ds[lon_dim].values
    lat_bounds = (float(np.min(lat_vals)), float(np.max(lat_vals)))
    lon_bounds = (float(np.min(lon_vals)), float(np.max(lon_vals)))

    if not (-90.001 <= lat_bounds[0] and lat_bounds[1] <= 90.001):
        issues.append(f"Latitude bounds [{lat_bounds[0]}, {lat_bounds[1]}] exceed physical range [-90, 90].")
    if not (-180.001 <= lon_bounds[0] and lon_bounds[1] <= 180.001):
        issues.append(f"Longitude bounds [{lon_bounds[0]}, {lon_bounds[1]}] unnormalized; exceeds [-180, 180].")

    # Check CF coordinate attributes
    lat_units = str(ds[lat_dim].attrs.get("units", "")).lower()
    lon_units = str(ds[lon_dim].attrs.get("units", "")).lower()
    if lat_units not in ("degrees_north", "degree_north", "degree_n", "degrees_n"):
        issues.append(f"Latitude units '{lat_units}' non-standard; expected 'degrees_north'.")
    if lon_units not in ("degrees_east", "degree_east", "degree_e", "degrees_e"):
        issues.append(f"Longitude units '{lon_units}' non-standard; expected 'degrees_east'.")

    # Check CF grid_mapping container
    has_grid_mapping = False
    epsg_detected: Optional[int] = None
    crs_name: Optional[str] = None

    if isinstance(ds, xr.Dataset):
        for candidate_var in ("spatial_ref", "crs", "grid_mapping"):
            if candidate_var in ds:
                gm_attrs = ds[candidate_var].attrs
                if gm_attrs.get("grid_mapping_name") == "latitude_longitude" or "EPSG:4326" in str(gm_attrs.get("epsg_code", "")):
                    has_grid_mapping = True
                    epsg_detected = 4326
                    crs_name = gm_attrs.get("geographic_crs_name", "WGS 84")
                    break

        if not has_grid_mapping:
            issues.append("Missing CF grid_mapping container variable ('spatial_ref' or 'crs').")

    is_compliant = len(issues) == 0
    result = CRSValidationResult(
        is_compliant=is_compliant,
        epsg_code=epsg_detected if is_compliant else None,
        crs_name=crs_name,
        spatial_dims=(lat_dim, lon_dim),
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        has_grid_mapping=has_grid_mapping,
        issues=issues,
    )

    if strict and not is_compliant:
        raise CRSComplianceError(f"Dataset fails WGS 84 CRS compliance: {'; '.join(issues)}")

    return result


# =====================================================================
# 6. Centralized Safe Validation Interface
# =====================================================================

def detect_station_network_heuristic(station_id: str) -> Optional[str]:
    """
    Detect the likely observing network of a station identifier using pattern heuristics.

    Supported network patterns:
    - 'tahmo': TA##### (e.g., TA00185)
    - 'ghcn': 11 alphanumeric characters starting with 2-letter country code (e.g. KE000063740)
    - 'cds_insitu': 5-digit WMO index (e.g. 63740) or standard WIGOS string
    - 'icao_kmd': 4-letter alphabetic ICAO/KMD code (e.g., KEMU, HKMO, HKEL, HKJK)
    """
    clean_id = str(station_id).strip()
    for net_name, info in STATION_NETWORK_PATTERNS.items():
        if info["regex"].match(clean_id):
            return net_name
    return None


def suggest_extractor_for_variable(variable_name: str) -> List[str]:
    """
    Identify which MasharikiWeather extractors support a given meteorological variable.

    Mathematical docstring:
    Input variable_name: String query.
    Output: List of extractor names supporting the variable or its aliases.
    """
    clean = str(variable_name).strip().lower().replace(" ", "_")
    if clean in CANONICAL_VARIABLE_EXTRACTOR_MAP:
        return list(CANONICAL_VARIABLE_EXTRACTOR_MAP[clean])

    # Check fuzzy matches across all known variables
    all_known = list(CANONICAL_VARIABLE_EXTRACTOR_MAP.keys())
    matches = difflib.get_close_matches(clean, all_known, n=1, cutoff=0.6)
    if matches:
        return list(CANONICAL_VARIABLE_EXTRACTOR_MAP[matches[0]])

    return []


def validate_variables(
    requested_variables: Sequence[str],
    available_variables: Optional[Sequence[str]] = None,
    extractor_name: Optional[str] = None,
    strict: bool = True,
) -> VariableValidationResult:
    """
    Validate requested meteorological variables against extractor capabilities.

    Provides:
    1. Fast vectorized and set-membership availability checks.
    2. Fuzzy matching and typo suggestions within the extractor's valid variable pool.
    3. Cross-extractor suggestions when asking for a variable provided by a different source
       (e.g., requesting '2m_temperature' on IMERG suggests ERA5, TAHMO, GHCN, CDS-InSitu).

    Mathematical docstring:
    Input requested_variables: Sequence of length V_req.
    Available variables: Sequence of length V_avail.
    Complexity: O(V_req * log(V_avail)) string normalization and fuzzy comparison.

    Parameters
    ----------
    requested_variables : Sequence[str]
        List of variable names requested by caller.
    available_variables : Optional[Sequence[str]]
        Explicit set of valid variables for this extractor. If None, resolved from registry.
    extractor_name : Optional[str]
        Identifier of the calling extractor ('imerg', 'era5', 'tahmo', etc.).
    strict : bool, default True
        If True, raises VariableNotAvailableError when any requested variable is unavailable.
        If False, issues UserWarning and returns sanitized results.

    Returns
    -------
    VariableValidationResult
        Detailed validation container.
    """
    ext_key = str(extractor_name or "").lower().strip()
    registry_entry = GLOBAL_EXTRACTOR_VARIABLE_REGISTRY.get(ext_key, {})

    if available_variables is not None:
        avail_pool = set(available_variables)
    elif registry_entry:
        avail_pool = set(registry_entry["canonical_variables"]).union(registry_entry.get("aliases", {}).keys())
    else:
        avail_pool = set()

    clean_requested = [str(v).strip() for v in requested_variables]
    valid_vars: List[str] = []
    invalid_vars: List[str] = []
    suggestions: Dict[str, List[str]] = {}
    alternatives: Dict[str, List[str]] = {}
    diagnostics: List[str] = []

    for var in clean_requested:
        var_norm = var.lower().replace(" ", "_")
        if var in avail_pool or var_norm in avail_pool:
            valid_vars.append(var)
        else:
            invalid_vars.append(var)
            # 1. Fuzzy match within same extractor
            close = difflib.get_close_matches(var_norm, list(avail_pool), n=3, cutoff=0.5) if avail_pool else []
            if close:
                suggestions[var] = close

            # 2. Cross-extractor suggestions
            ext_alts = suggest_extractor_for_variable(var_norm)
            filtered_alts = [e for e in ext_alts if e != ext_key]
            if filtered_alts:
                alternatives[var] = filtered_alts

            # Construct diagnostic message
            msg_parts = [f"Variable '{var}' is not provided by {extractor_name or 'extractor'}."]
            if close:
                msg_parts.append(f"Did you mean: {close}?")
            if filtered_alts:
                msg_parts.append(
                    f"'{var}' is available in alternative extractors: {filtered_alts}."
                )
            if ext_key == "imerg" and not close:
                msg_parts.append(
                    "Note: IMERG is a dedicated satellite precipitation mission and does not provide temperature, pressure, or wind."
                )
            diagnostics.append(" ".join(msg_parts))

    is_valid = len(invalid_vars) == 0
    result = VariableValidationResult(
        is_valid=is_valid,
        requested_variables=clean_requested,
        valid_variables=valid_vars,
        invalid_variables=invalid_vars,
        suggestions=suggestions,
        cross_extractor_alternatives=alternatives,
        diagnostics=diagnostics,
    )

    if not is_valid:
        combined_diag = " | ".join(diagnostics)
        if strict:
            raise VariableNotAvailableError(
                f"Variable validation failed for {extractor_name or 'extractor'}: {combined_diag}"
            )
        else:
            logger.warning("Variable validation warning: %s", combined_diag)
            warnings.warn(combined_diag, UserWarning, stacklevel=2)

    return result


def validate_station_identifiers(
    stations: Sequence[str],
    valid_catalog: Optional[Union[Sequence[str], pd.DataFrame, pd.Index, Set[str]]] = None,
    extractor_network: str = "tahmo",
    strict: bool = True,
) -> StationValidationResult:
    """
    Validate station identifiers against known networks and catch cross-network mismatches.

    Detects:
    1. Cross-network syntax mismatches (e.g. querying KEMU on TAHMO, or TA00185 on GHCN).
    2. Missing stations from catalog with fuzzy matching suggestions.
    3. Vectorized station catalog set-membership validation.

    Mathematical docstring:
    Input stations: Sequence of length N_req.
    Output: StationValidationResult with categorized valid, invalid, and mismatched stations.
    Complexity: O(N_req) pattern checking and catalog membership lookup.

    Parameters
    ----------
    stations : Sequence[str]
        Station identifiers requested by caller.
    valid_catalog : Optional[Union[Sequence[str], pd.DataFrame, pd.Index, Set[str]]]
        Known valid station inventory. If DataFrame, searches 'station_id' or index.
    extractor_network : str, default 'tahmo'
        Expected observing network ('tahmo', 'ghcn', 'ghcnd', 'cds_insitu').
    strict : bool, default True
        If True, raises StationNotFoundError on invalid stations or cross-network mismatches.

    Returns
    -------
    StationValidationResult
        Structured station validation outcome.
    """
    clean_stations = [str(s).strip() for s in stations if str(s).strip()]
    if not clean_stations:
        return StationValidationResult(
            is_valid=True,
            requested_stations=[],
            valid_stations=[],
            invalid_stations=[],
        )

    # Build catalog set if provided
    catalog_set: Optional[Set[str]] = None
    if valid_catalog is not None:
        if isinstance(valid_catalog, pd.DataFrame):
            if "station_id" in valid_catalog.columns:
                catalog_set = set(valid_catalog["station_id"].astype(str).str.strip())
            else:
                catalog_set = set(valid_catalog.index.astype(str).str.strip())
        elif isinstance(valid_catalog, (pd.Index, set)):
            catalog_set = set(str(s).strip() for s in valid_catalog)
        elif isinstance(valid_catalog, Sequence):
            catalog_set = set(str(s).strip() for s in valid_catalog)

    net_key = str(extractor_network).lower().strip()
    if net_key in ("ghcnd", "ghcnh", "ghcn_daily", "ghcn_hourly"):
        net_key = "ghcn"

    valid_stations: List[str] = []
    invalid_stations: List[str] = []
    mismatches: Dict[str, str] = {}
    suggestions: Dict[str, List[str]] = {}
    diagnostics: List[str] = []

    for st in clean_stations:
        in_catalog = catalog_set is not None and st in catalog_set
        detected_net = detect_station_network_heuristic(st)

        # Cross-network mismatch diagnostic checks
        if detected_net is not None and detected_net != net_key:
            mismatches[st] = detected_net
            invalid_stations.append(st)

            if net_key == "tahmo" and detected_net == "icao_kmd":
                diag = (
                    f"Station '{st}' is not a valid TAHMO station identifier. "
                    f"'{st}' appears to be a Kenya Meteorological Department / ICAO airport code (e.g. Moi Air Base). "
                    "TAHMO stations use identifiers starting with 'TA' followed by digits (e.g., 'TA00185'). "
                    "For Kenya national stations, query GHCNExtractor or CDSInSituExtractor."
                )
            elif net_key == "ghcn" and detected_net == "tahmo":
                diag = (
                    f"Station '{st}' begins with 'TA', which corresponds to a TAHMO weather station. "
                    "GHCN uses 11-character NOAA identifiers (e.g., 'KE000063740'). "
                    "Did you mean to use TAHMOExtractor?"
                )
            elif net_key == "cds_insitu" and detected_net == "tahmo":
                diag = (
                    f"Station '{st}' is a TAHMO identifier. CDS In-Situ utilizes WMO 5-digit station indices "
                    "or WIGOS identifiers. Consider using TAHMOExtractor."
                )
            else:
                diag = (
                    f"Station '{st}' matches format for '{detected_net}', but was requested on '{net_key}'."
                )
            diagnostics.append(diag)
            continue

        if catalog_set is not None:
            if in_catalog:
                valid_stations.append(st)
            else:
                invalid_stations.append(st)
                close = difflib.get_close_matches(st, list(catalog_set), n=3, cutoff=0.6)
                if close:
                    suggestions[st] = close
                    diag = f"Station '{st}' not found in {net_key} catalog. Did you mean: {close}?"
                else:
                    diag = f"Station '{st}' not found in active {net_key} station inventory."
                diagnostics.append(diag)
        else:
            # When catalog is absent, rely on pattern conformity
            if detected_net == net_key or detected_net is None:
                valid_stations.append(st)
            else:
                invalid_stations.append(st)

    is_valid = len(invalid_stations) == 0
    result = StationValidationResult(
        is_valid=is_valid,
        requested_stations=clean_stations,
        valid_stations=valid_stations,
        invalid_stations=invalid_stations,
        mismatched_network_stations=mismatches,
        suggestions=suggestions,
        diagnostics=diagnostics,
    )

    if not is_valid:
        combined_diag = " | ".join(diagnostics)
        if strict:
            raise StationNotFoundError(
                f"Station validation failed for {net_key}: {combined_diag}"
            )
        else:
            logger.warning("Station validation warning: %s", combined_diag)
            warnings.warn(combined_diag, UserWarning, stacklevel=2)

    return result


def validate_temporal_coverage(
    temporal_range: TemporalRange,
    station_catalog: Optional[pd.DataFrame] = None,
    extractor_bounds: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    requested_stations: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> TemporalValidationResult:
    """
    High-throughput vectorized temporal coverage validation across station inventories.

    Vectorized evaluations:
    1. Pre-commission check: station active interval begins after query end date.
    2. Post-decommission check: station active interval ends before query start date.
    3. Active coverage fraction: exact temporal intersection ratio in [0.0, 1.0].
    4. Operational bounds checking for remote APIs.

    Mathematical docstring:
    Input query: [T_start, T_end].
    Station catalog: (N_stations, C_metadata) with start_date and end_date columns.
    Complexity: O(N_stations) vectorized Pandas / NumPy datetime operations.

    Parameters
    ----------
    temporal_range : TemporalRange
        Requested start and end timestamps.
    station_catalog : Optional[pd.DataFrame]
        Catalog containing station_id, start_date, and end_date.
    extractor_bounds : Optional[Tuple[pd.Timestamp, pd.Timestamp]]
        API operational limits (earliest_available, latest_available).
    requested_stations : Optional[Sequence[str]]
        Subset of station identifiers to validate. If None, validates all catalog stations.
    strict : bool, default False
        If True, raises TemporalCoverageError if requested range is out of operational bounds
        or if requested stations have 0.0 coverage.

    Returns
    -------
    TemporalValidationResult
        Structured temporal validation outcome.
    """
    q_start = temporal_range.start
    q_end = temporal_range.end
    diagnostics: List[str] = []

    # 1. Extractor global operational limits check
    if extractor_bounds is not None:
        op_start, op_end = extractor_bounds
        if q_start < op_start:
            diagnostics.append(
                f"Requested start date {q_start.strftime('%Y-%m-%d')} precedes extractor earliest operational date {op_start.strftime('%Y-%m-%d')}."
            )
        if q_end > op_end:
            diagnostics.append(
                f"Requested end date {q_end.strftime('%Y-%m-%d')} exceeds extractor latest operational date {op_end.strftime('%Y-%m-%d')}."
            )

    active_stations: List[str] = []
    inactive_stations: List[str] = []
    pre_comm_stations: List[str] = []
    post_decomm_stations: List[str] = []
    coverage_fractions: Dict[str, float] = {}

    # 2. Vectorized station catalog check
    if station_catalog is not None and not station_catalog.empty:
        df = station_catalog.copy()
        id_col = "station_id" if "station_id" in df.columns else (
            "station" if "station" in df.columns else df.columns[0]
        )

        if requested_stations is not None:
            clean_req = set(str(s).strip() for s in requested_stations)
            df = df[df[id_col].astype(str).str.strip().isin(clean_req)]

        if not df.empty and "start_date" in df.columns and "end_date" in df.columns:
            st_ids = df[id_col].astype(str).str.strip().values
            st_start = pd.to_datetime(df["start_date"], errors="coerce")
            st_end = pd.to_datetime(df["end_date"], errors="coerce").fillna(pd.Timestamp.max)

            # Vectorized masks
            pre_comm_mask = (st_start > q_end).values
            post_decomm_mask = (st_end < q_start).values
            active_mask = ((st_start <= q_end) & (st_end >= q_start)).values

            pre_comm_stations = list(st_ids[pre_comm_mask])
            post_decomm_stations = list(st_ids[post_decomm_mask])
            inactive_stations = pre_comm_stations + post_decomm_stations
            active_stations = list(st_ids[active_mask])

            # Vectorized overlap duration and coverage fraction
            total_query_sec = max((q_end - q_start).total_seconds(), 1.0)
            overlap_start = st_start.clip(lower=q_start)
            overlap_end = st_end.clip(upper=q_end)
            overlap_sec = (overlap_end - overlap_start).dt.total_seconds().values
            fractions = np.clip(overlap_sec / total_query_sec, 0.0, 1.0)

            for sid, frac, is_act in zip(st_ids, fractions, active_mask):
                if is_act:
                    coverage_fractions[sid] = float(np.round(frac, 4))
                else:
                    coverage_fractions[sid] = 0.0

            if pre_comm_stations:
                diagnostics.append(
                    f"{len(pre_comm_stations)} station(s) commissioned after query window: {pre_comm_stations[:5]}"
                )
            if post_decomm_stations:
                diagnostics.append(
                    f"{len(post_decomm_stations)} station(s) decommissioned before query window: {post_decomm_stations[:5]}"
                )

    is_valid = len(diagnostics) == 0 and (len(inactive_stations) == 0 or len(active_stations) > 0)
    result = TemporalValidationResult(
        is_valid=is_valid,
        requested_range=temporal_range,
        active_stations=active_stations,
        inactive_stations=inactive_stations,
        pre_commission_stations=pre_comm_stations,
        post_decommission_stations=post_decomm_stations,
        coverage_fractions=coverage_fractions,
        diagnostics=diagnostics,
    )

    if strict and not is_valid:
        raise TemporalCoverageError(
            f"Temporal coverage validation failed: {'; '.join(diagnostics)}"
        )

    return result


# =====================================================================
# 7. Base Extractor Class
# =====================================================================

class BaseExtractor(ABC):
    """
    Abstract Base Class for all MasharikiWeather extraction engines.

    Provides standardized spatial bounding box, temporal range, variable availability,
    station network, and WGS 84 (EPSG:4326) CRS enforcement interfaces.
    """

    extractor_name: str = "base"
    network_name: str = ""
    supported_variables: Optional[Set[str]] = None
    operational_bounds: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

    @abstractmethod
    def extract(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute remote or cloud extraction and return dataset or local storage paths.
        """
        pass

    def validate_spatial_bounds(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> SpatialBoundingBox:
        """
        Validate and construct standard SpatialBoundingBox.
        """
        return SpatialBoundingBox(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

    def validate_temporal_range(
        self,
        start_date: Union[str, datetime, date, pd.Timestamp],
        end_date: Union[str, datetime, date, pd.Timestamp],
    ) -> TemporalRange:
        """
        Validate and construct standard TemporalRange.
        """
        return TemporalRange.from_dates(start_date=start_date, end_date=end_date)

    def validate_variables(
        self,
        requested_variables: Sequence[str],
        strict: bool = True,
    ) -> VariableValidationResult:
        """
        Validate variable availability against this extractor's catalog.
        """
        avail = getattr(self, "supported_variables", None)
        if not isinstance(avail, (set, list, tuple)) or len(avail) == 0:
            avail = None
        return validate_variables(
            requested_variables=requested_variables,
            available_variables=avail,
            extractor_name=self.extractor_name,
            strict=strict,
        )

    def validate_stations(
        self,
        requested_stations: Sequence[str],
        station_catalog: Optional[Union[Sequence[str], pd.DataFrame, pd.Index, Set[str]]] = None,
        strict: bool = True,
    ) -> StationValidationResult:
        """
        Validate station identifiers and diagnose cross-network mismatches.
        """
        return validate_station_identifiers(
            stations=requested_stations,
            valid_catalog=station_catalog,
            extractor_network=self.network_name or self.extractor_name,
            strict=strict,
        )

    def validate_temporal_bounds(
        self,
        temporal_range: Union[TemporalRange, Tuple[Any, Any]],
        station_catalog: Optional[pd.DataFrame] = None,
        requested_stations: Optional[Sequence[str]] = None,
        strict: bool = False,
    ) -> TemporalValidationResult:
        """
        Validate temporal range against operational limits and station active windows.
        """
        if not isinstance(temporal_range, TemporalRange):
            t_range = TemporalRange.from_dates(temporal_range[0], temporal_range[1])
        else:
            t_range = temporal_range

        bounds = self.operational_bounds or EXTRACTOR_OPERATIONAL_BOUNDS.get(self.extractor_name.lower())
        return validate_temporal_coverage(
            temporal_range=t_range,
            station_catalog=station_catalog,
            extractor_bounds=bounds,
            requested_stations=requested_stations,
            strict=strict,
        )

    def enforce_crs(
        self,
        ds: xr.Dataset,
        standardize_names: bool = True,
        wrap_lon_to_180: bool = True,
    ) -> xr.Dataset:
        """
        Enforce WGS 84 (EPSG:4326) CRS compliance and CF-1.8 grid_mapping on dataset.
        """
        return enforce_wgs84_crs(
            ds=ds,
            standardize_names=standardize_names,
            wrap_lon_to_180=wrap_lon_to_180,
        )

    def validate_crs(
        self,
        ds: Union[xr.Dataset, xr.DataArray],
        strict: bool = False,
    ) -> CRSValidationResult:
        """
        Check that dataset complies with WGS 84 (EPSG:4326) CF grid_mapping conventions.
        """
        return validate_crs_compliance(ds=ds, strict=strict)
