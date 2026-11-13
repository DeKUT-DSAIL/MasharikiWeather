import difflib
import logging
import os
import shutil
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import dask
from dask.diagnostics import ProgressBar
import fsspec
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr

from .extract.apis.base import AFRICA_BBOX, SpatialBoundingBox
from .rechunker import DEFAULT_TIME_CHUNKS, get_zarr_encoding, rechunk_dataset

logger = logging.getLogger(__name__)

__all__ = [
    "MasharikiWeatherDataset",
    "ObservationDataFrame",
    "StationList",
    "ParameterList",
    "Windows",
    "sliding_window_view",
    "df_add_missing_columns",
    "extract_stations",
    "extract_gridded",
    "pull_station_data",
    "pull_gridded_data",
    "pull_weather_data",
]

# Canonical meteorological variable lexicon and bidirectional cross-source harmonization
CANONICAL_VARIABLE_METADATA: Dict[str, Dict[str, Any]] = {
    "2m_temperature": {
        "standard_name": "air_temperature",
        "units": "degC",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Near-surface dry air temperature at 2 meters",
        "aliases": ["temperature", "temperature_2m", "temp", "te", "2t", "air_temperature"],
    },
    "2m_dewpoint_temperature": {
        "standard_name": "dew_point_temperature",
        "units": "degC",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Near-surface dewpoint temperature at 2 meters",
        "aliases": ["dewpoint", "dewpoint_temperature", "2d", "dew_point_temperature"],
    },
    "10m_u_component_of_wind": {
        "standard_name": "eastward_wind",
        "units": "m s**-1",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Zonal 10m wind velocity",
        "aliases": ["u10", "10u"],
    },
    "10m_v_component_of_wind": {
        "standard_name": "northward_wind",
        "units": "m s**-1",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Meridional 10m wind velocity",
        "aliases": ["v10", "10v"],
    },
    "wind_speed": {
        "standard_name": "wind_speed",
        "units": "m s**-1",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Scalar horizontal wind speed",
        "aliases": ["ws", "wind_velocity"],
    },
    "wind_from_direction": {
        "standard_name": "wind_from_direction",
        "units": "degrees",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Meteorological wind origin direction (0-360)",
        "aliases": ["wd", "wind_direction"],
    },
    "surface_pressure": {
        "standard_name": "surface_air_pressure",
        "units": "Pa",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Barometric surface pressure",
        "aliases": ["sp", "ap", "pres", "pressure", "air_pressure"],
    },
    "mean_sea_level_pressure": {
        "standard_name": "air_pressure_at_mean_sea_level",
        "units": "Pa",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Barometric mean sea level pressure",
        "aliases": ["msl", "air_pressure_at_sea_level"],
    },
    "relative_humidity": {
        "standard_name": "relative_humidity",
        "units": "%",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Near-surface relative humidity",
        "aliases": ["rh"],
    },
    "total_precipitation": {
        "standard_name": "precipitation_amount",
        "units": "mm",
        "physical_type": "PRECIPITATION",
        "resampling_method": "sum",
        "description": "Cumulative total precipitation",
        "aliases": ["precipitation", "precip", "pr", "tp", "precipitationcal", "precipitation_amount", "accumulated_precipitation"],
    },
    "convective_precipitation": {
        "standard_name": "convective_precipitation_amount",
        "units": "mm",
        "physical_type": "PRECIPITATION",
        "resampling_method": "sum",
        "description": "Convective precipitation amount",
        "aliases": ["cp"],
    },
    "large_scale_precipitation": {
        "standard_name": "large_scale_precipitation_amount",
        "units": "mm",
        "physical_type": "PRECIPITATION",
        "resampling_method": "sum",
        "description": "Stratiform large scale precipitation amount",
        "aliases": ["lsp"],
    },
    "surface_solar_radiation_downwards": {
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "units": "J m**-2",
        "physical_type": "PHYSICAL",
        "resampling_method": "sum",
        "description": "Downwelling surface solar radiation shortwave flux",
        "aliases": ["solar_radiation", "sr", "rad", "ssrd"],
    },
    "total_cloud_cover": {
        "standard_name": "cloud_area_fraction",
        "units": "1",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Total fractional cloud cover",
        "aliases": ["tcc"],
    },
    "convective_available_potential_energy": {
        "standard_name": "atmosphere_convective_available_potential_energy",
        "units": "J kg**-1",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Convective available potential energy",
        "aliases": ["cape"],
    },
    "convective_inhibition": {
        "standard_name": "atmosphere_convective_inhibition",
        "units": "J kg**-1",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Convective inhibition",
        "aliases": ["cin"],
    },
}

ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canon_var, meta_info in CANONICAL_VARIABLE_METADATA.items():
    ALIAS_TO_CANONICAL[canon_var.lower()] = canon_var
    for a in meta_info["aliases"]:
        ALIAS_TO_CANONICAL[a.lower()] = canon_var


def canonicalize_variable_name(name: str) -> str:
    """
    Map legacy, source-specific, or sensor-qualified variable name to canonical form.

    Mathematical docstring:
        Canonical mapping: str -> str.
        Examples:
            'temperature' -> '2m_temperature'
            'rh_S001402' -> 'relative_humidity_S001402'
            'tp' -> 'total_precipitation'
    """
    low = name.lower().strip()
    if low in ALIAS_TO_CANONICAL:
        return ALIAS_TO_CANONICAL[low]
    if "_" in name:
        parts = name.split("_")
        prefix = parts[0].lower().strip()
        suffix = "_".join(parts[1:])
        if prefix in ALIAS_TO_CANONICAL:
            return f"{ALIAS_TO_CANONICAL[prefix]}_{suffix}"
    return name


def _normalize_source_name(source: str) -> str:
    """
    Map observation network source aliases to standard canonical keys.

    Args:
        source: User-supplied or catalog source identifier.

    Returns:
        Standard source string ('tahmo', 'ghcn', 'cds_insitu').
    """
    s = str(source).lower().strip()
    if s in ("ghcnd", "ghcn_daily", "ghcn_hourly", "ghcn"):
        return "ghcn"
    if s in ("cds_insitu", "cds", "insitu"):
        return "cds_insitu"
    if s == "tahmo":
        return "tahmo"
    return s


def _tag_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Attach the originating observation source name to a long-format frame.

    Args:
        df: Long-format observations with columns including time and station.
        source: Name of the observation source (e.g. 'tahmo', 'ghcn', 'cds_insitu').

    Returns:
        Copy of df with an additional string column '__source__'.
    """
    out = df.copy()
    out["__source__"] = _normalize_source_name(source)
    return out



def _normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize time/timestamp column name in flat observation DataFrames and ensure datetime dtype.

    Mathematical docstring:
        Input: DataFrame with N_records rows and columns containing a temporal indicator.
        Output: DataFrame with guaranteed 'time' column of datetime64[ns, UTC] or tz-aware/naive datetime.
    """
    if "time" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "time"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "time"})
        elif "level_0" in df.columns:
            df = df.rename(columns={"level_0": "time"})
        elif df.index.name in ["time", "timestamp", "datetime"]:
            df = df.reset_index().rename(columns={df.index.name: "time"})
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(names=["time"])
        else:
            for col in list(df.columns):
                if col not in ["station", "station_id"]:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        df = df.rename(columns={col: "time"})
                        break
                    except Exception:
                        pass
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    return df


class StationList(list):
    """
    Specialized list container for active station identifiers.
    Preserves Python list semantics while exposing a .size property and .tolist()
    for array and pandas Index compatibility.
    """

    @property
    def size(self) -> int:
        """Return the total number of station identifiers in the list."""
        return len(self)

    def tolist(self) -> List[str]:
        """Convert StationList instance to a standard Python list of strings."""
        return list(self)


class ParameterList(list):
    """
    Specialized list container for active weather parameter identifiers.
    Preserves Python list semantics while exposing a .size property and .tolist()
    for array and pandas Index compatibility.
    """

    @property
    def size(self) -> int:
        """Return the total number of weather parameters in the list."""
        return len(self)

    def tolist(self) -> List[str]:
        """Convert ParameterList instance to a standard Python list of strings."""
        return list(self)


class ObservationDataFrame(pd.DataFrame):
    """
    PeakWeather-inspired DataFrame container for multi-station observation matrices.

    Features 2-level MultiIndex columns: Level 0 = station_id, Level 1 = parameter.
    Supports multi-key tuple slicing:
        ax.plot(df.index, df['TA00001', 'temperature'], label='TA00001 Temp')
    chained single-key slicing:
        df['TA00001']['temperature']
    and cross-station variable extraction:
        df['temperature']
    while transparently resolving canonical variable names, aliases, and collocated sensors.
    """

    @property
    def _constructor(self) -> type:
        """Internal constructor ensuring DataFrame slicing retains ObservationDataFrame type."""
        return ObservationDataFrame

    def _resolve_param_candidates(self, param: str) -> List[str]:
        p_low = str(param).lower().strip()
        candidates = [param, p_low]
        if p_low in ALIAS_TO_CANONICAL:
            canon = ALIAS_TO_CANONICAL[p_low]
            candidates.append(canon)
            meta = CANONICAL_VARIABLE_METADATA.get(canon, {})
            candidates.extend(meta.get("aliases", []))
        return list(dict.fromkeys(candidates))

    def __getitem__(self, key: Any) -> Any:
        """
        Index multi-station observations via 2-level tuple, station string, or parameter string.

        Mathematical docstring:
        - Tuple index: (station_id, parameter) -> pd.Series of shape (T_timestamps,)
        - Station index: station_id -> ObservationDataFrame of shape (T_timestamps, C_parameters)
        - Parameter index: parameter -> ObservationDataFrame of shape (T_timestamps, N_stations)
        """
        try:
            return super().__getitem__(key)
        except (KeyError, TypeError):
            pass

        # 1. Tuple key indexing on 2-level MultiIndex: (station_id, parameter)
        if isinstance(key, tuple) and len(key) == 2 and isinstance(self.columns, pd.MultiIndex):
            st, param = key
            candidates = self._resolve_param_candidates(param)
            # Exact candidate match
            for c in candidates:
                if (st, c) in self.columns:
                    return super().__getitem__((st, c))
            # Collocated sensor prefix match (e.g. rh_S001402)
            matching_cols = [
                col for col in self.columns
                if col[0] == st and any(col[1].startswith(c + "_") for c in candidates)
            ]
            if len(matching_cols) == 1:
                return super().__getitem__(matching_cols[0])
            elif len(matching_cols) > 1:
                return super().__getitem__(matching_cols)

        # 2. String key indexing
        elif isinstance(key, str):
            if isinstance(self.columns, pd.MultiIndex):
                # Check if key is a station code in Level 0
                if key in self.columns.get_level_values(0):
                    sub_df = super().__getitem__(key)
                    return ObservationDataFrame(sub_df) if isinstance(sub_df, pd.DataFrame) else sub_df

                # Check if key is a parameter across all stations in Level 1
                candidates = self._resolve_param_candidates(key)
                for c in candidates:
                    if c in self.columns.get_level_values(1):
                        return ObservationDataFrame(self.xs(c, axis=1, level=1))
            else:
                # Single-level columns (e.g. df['TA00001'])
                candidates = self._resolve_param_candidates(key)
                for c in candidates:
                    if c in self.columns:
                        return super().__getitem__(c)
                matching_cols = [
                    col for col in self.columns
                    if any(str(col).startswith(c + "_") for c in candidates)
                ]
                if len(matching_cols) == 1:
                    return super().__getitem__(matching_cols[0])
                elif len(matching_cols) > 1:
                    return super().__getitem__(matching_cols)

        return super().__getitem__(key)

FrameArray = Union[pd.DataFrame, np.ndarray]

@dataclass(init=False)
class Windows:
    """
    A dynamic multi-modal container for sliding window views across N-Grids and Stations natively.
    """
    # 1. Point Stations (Eager Temporal Vectors)
    station_x: Union[pd.DataFrame, np.ndarray, xr.Dataset]
    station_mask_x: Union[pd.DataFrame, np.ndarray, xr.Dataset]
    station_y: Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]]
    station_mask_y: Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]]
    
    index_x: Optional[Union[pd.DatetimeIndex, np.ndarray]]
    index_y: Optional[Union[pd.DatetimeIndex, np.ndarray]]

    # 2. Dynamic N-Grids Mapping (Keys derived dynamically from gridded_url)
    grids_x: Dict[str, Union[np.ndarray, xr.Dataset]]
    grids_y: Dict[str, Union[np.ndarray, xr.Dataset]]
    grids_features: Dict[str, Sequence[str]]
    grids_lats: Dict[str, np.ndarray]
    grids_lons: Dict[str, np.ndarray]

    # 3. Static Priors (Mapped at native resolutions per-layer)
    station_static: Optional[np.ndarray]
    station_static_categorical: Optional[np.ndarray]
    native_topography: Optional[np.ndarray]
    native_topography_categorical: Optional[np.ndarray]
    native_topography_lats: Optional[np.ndarray]
    native_topography_lons: Optional[np.ndarray]

    station_coords: Optional[np.ndarray]  # (N_stations, 2) lat/lon for edge construction
    station_mask_2d: Optional[np.ndarray] # Optional physical boolean grid constraints
    
    # 4. Target Routing Engine
    target_node: Optional[str] # E.g., 'tahmo', 'era5', or 'imerg' to define the loss output layer.

    # 5. Temporal Engine & TSL Delay Math
    delay: int
    valid_batch_indices: Optional[np.ndarray]

    # Mathematical Coordinates
    station_labels: Optional[Sequence[str]]
    feature_labels: Optional[Sequence[str]]
    static_feature_labels: Optional[Sequence[str]]
    static_categorical_labels: Optional[Sequence[str]]
    grid_lats: Optional[np.ndarray]
    grid_lons: Optional[np.ndarray]

    # Scientific Metadata
    variable_types: Dict[str, Literal["PHYSICAL", "PRECIPITATION", "STATIC"]]

    # 6. Optional Mass Conservation Targets (CHIRPS/TAMSAT)
    constraint_targets: Dict[str, np.ndarray]
    temporal_masks: Dict[str, np.ndarray]
    constraint_time_index: Optional[np.ndarray]
    constraint_resolution: Optional[str]

    def __init__(
        self,
        station_x: Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]] = None,
        station_mask_x: Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]] = None,
        station_y: Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]] = None,
        station_mask_y: Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]] = None,
        index_x: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
        index_y: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
        grids_x: Optional[Dict[str, Union[np.ndarray, xr.Dataset]]] = None,
        grids_y: Optional[Dict[str, Union[np.ndarray, xr.Dataset]]] = None,
        grids_features: Optional[Dict[str, Sequence[str]]] = None,
        grids_lats: Optional[Dict[str, np.ndarray]] = None,
        grids_lons: Optional[Dict[str, np.ndarray]] = None,
        station_static: Optional[np.ndarray] = None,
        station_static_categorical: Optional[np.ndarray] = None,
        native_topography: Optional[np.ndarray] = None,
        native_topography_categorical: Optional[np.ndarray] = None,
        native_topography_lats: Optional[np.ndarray] = None,
        native_topography_lons: Optional[np.ndarray] = None,
        station_coords: Optional[np.ndarray] = None,
        station_mask_2d: Optional[np.ndarray] = None,
        target_node: Optional[str] = None,
        delay: int = 0,
        valid_batch_indices: Optional[np.ndarray] = None,
        station_labels: Optional[Sequence[str]] = None,
        feature_labels: Optional[Sequence[str]] = None,
        static_feature_labels: Optional[Sequence[str]] = None,
        static_categorical_labels: Optional[Sequence[str]] = None,
        grid_lats: Optional[np.ndarray] = None,
        grid_lons: Optional[np.ndarray] = None,
        variable_types: Optional[Dict[str, Literal["PHYSICAL", "PRECIPITATION", "STATIC"]]] = None,
        constraint_targets: Optional[Dict[str, np.ndarray]] = None,
        temporal_masks: Optional[Dict[str, np.ndarray]] = None,
        constraint_time_index: Optional[np.ndarray] = None,
        constraint_resolution: Optional[str] = None,
        x: Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]] = None,
        mask_x: Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]] = None,
        y: Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]] = None,
        mask_y: Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]] = None,
        static_stations: Optional[np.ndarray] = None,
        static_stations_categorical: Optional[np.ndarray] = None,
        **kwargs: Any,
    ):
        self.station_x = station_x if station_x is not None else (x if x is not None else np.array([]))
        self.station_mask_x = station_mask_x if station_mask_x is not None else (mask_x if mask_x is not None else np.array([]))
        self.station_y = station_y if station_y is not None else y
        self.station_mask_y = station_mask_y if station_mask_y is not None else mask_y
        self.index_x = index_x
        self.index_y = index_y
        self.grids_x = grids_x if grids_x is not None else {}
        self.grids_y = grids_y if grids_y is not None else {}
        self.grids_features = grids_features if grids_features is not None else {}
        self.grids_lats = grids_lats if grids_lats is not None else {}
        self.grids_lons = grids_lons if grids_lons is not None else {}
        self.station_static = station_static if station_static is not None else static_stations
        self.station_static_categorical = (
            station_static_categorical if station_static_categorical is not None else static_stations_categorical
        )
        self.native_topography = native_topography
        self.native_topography_categorical = native_topography_categorical
        self.native_topography_lats = native_topography_lats
        self.native_topography_lons = native_topography_lons
        self.station_coords = station_coords
        self.station_mask_2d = station_mask_2d
        self.target_node = target_node
        self.delay = delay
        self.valid_batch_indices = valid_batch_indices
        self.station_labels = station_labels
        self.feature_labels = feature_labels
        self.static_feature_labels = static_feature_labels
        self.static_categorical_labels = static_categorical_labels
        self.grid_lats = grid_lats
        self.grid_lons = grid_lons
        self.variable_types = variable_types if variable_types is not None else {}
        self.constraint_targets = constraint_targets if constraint_targets is not None else {}
        self.temporal_masks = temporal_masks if temporal_masks is not None else {}
        self.constraint_time_index = constraint_time_index
        self.constraint_resolution = constraint_resolution
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def x(self) -> Union[pd.DataFrame, np.ndarray, xr.Dataset]:
        """
        Historical station input tensor.
        Contract shape: (B, T_in, N_stations, C_features) or pd.DataFrame.
        """
        return self.station_x

    @x.setter
    def x(self, value: Union[pd.DataFrame, np.ndarray, xr.Dataset]) -> None:
        self.station_x = value

    @property
    def y(self) -> Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]]:
        """
        Target station output tensor.
        Contract shape: (B, T_out, N_stations, C_targets) or pd.DataFrame or None.
        """
        return self.station_y

    @y.setter
    def y(self, value: Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]]) -> None:
        self.station_y = value

    @property
    def mask_x(self) -> Union[pd.DataFrame, np.ndarray, xr.Dataset]:
        """
        Historical station observation binary mask tensor.
        Contract shape: (B, T_in, N_stations, C_features) or pd.DataFrame.
        """
        return self.station_mask_x

    @mask_x.setter
    def mask_x(self, value: Union[pd.DataFrame, np.ndarray, xr.Dataset]) -> None:
        self.station_mask_x = value

    @property
    def mask_y(self) -> Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]]:
        """
        Target station observation binary mask tensor.
        Contract shape: (B, T_out, N_stations, C_targets) or pd.DataFrame or None.
        """
        return self.station_mask_y

    @mask_y.setter
    def mask_y(self, value: Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]]) -> None:
        self.station_mask_y = value

    @property
    def static_stations(self) -> Optional[np.ndarray]:
        """
        Station continuous static terrain and geographical priors tensor.
        Contract shape: (N_stations, C_static_cont) or None.
        """
        return self.station_static

    @static_stations.setter
    def static_stations(self, value: Optional[np.ndarray]) -> None:
        self.station_static = value

    @property
    def static_stations_categorical(self) -> Optional[np.ndarray]:
        """
        Station categorical static terrain priors tensor (e.g. soil/land cover classes).
        Contract shape: (N_stations, C_static_cat) or None.
        """
        return self.station_static_categorical

    @static_stations_categorical.setter
    def static_stations_categorical(self, value: Optional[np.ndarray]) -> None:
        self.station_static_categorical = value

    @property
    def gridded_x(self) -> Optional[Union[np.ndarray, xr.Dataset, Dict[str, Union[np.ndarray, xr.Dataset]]]]:
        """
        Historical gridded reanalysis/satellite tensor or dataset mapping.
        Contract shape: (B, T_in, H_lat, W_lon, C_channels) or xr.Dataset or Dict[str, xr.Dataset].
        """
        if self.grids_x:
            if len(self.grids_x) == 1:
                return next(iter(self.grids_x.values()))
            return self.grids_x
        return None

    @gridded_x.setter
    def gridded_x(self, value: Any) -> None:
        if isinstance(value, dict):
            self.grids_x = value
        elif value is not None:
            self.grids_x = {"grid": value}

    @property
    def gridded_y(self) -> Optional[Union[np.ndarray, xr.Dataset, Dict[str, Union[np.ndarray, xr.Dataset]]]]:
        """
        Target gridded reanalysis/satellite tensor or dataset mapping.
        Contract shape: (B, T_out, H_lat, W_lon, C_channels) or xr.Dataset or Dict[str, xr.Dataset].
        """
        if self.grids_y:
            if len(self.grids_y) == 1:
                return next(iter(self.grids_y.values()))
            return self.grids_y
        return None

    @gridded_y.setter
    def gridded_y(self, value: Any) -> None:
        if isinstance(value, dict):
            self.grids_y = value
        elif value is not None:
            self.grids_y = {"grid": value}

    @property
    def gridded_features(self) -> Sequence[str]:
        if self.grids_features:
            for k in self.grids_features:
                return self.grids_features[k]
        return self.feature_labels or []

class GriddedResult(tuple):
    """
    Unified dual-interface container returning (extracted_dfs, mask_dfs) while
    also providing direct dictionary access to extracted_dfs for backwards compatibility.
    """
    def __new__(cls, extracted_dfs: Dict[str, pd.DataFrame], mask_dfs: Dict[str, pd.DataFrame]) -> "GriddedResult":
        """Construct dual-interface tuple containing extracted observations and quality masks."""
        return super().__new__(cls, (extracted_dfs, mask_dfs))

    def __getitem__(self, item: Any) -> Any:
        """Retrieve extracted group DataFrame by string key or tuple index."""
        if isinstance(item, str):
            return self[0][item]
        return super().__getitem__(item)

    def __contains__(self, item: Any) -> bool:
        """Check group membership in extracted features."""
        return item in self[0] or super().__contains__(item)

    def get(self, key: str, default: Any = None) -> Any:
        """Safely retrieve extracted group DataFrame or return default."""
        return self[0].get(key, default)

    def keys(self) -> Any:
        """Return gridded feature group keys."""
        return self[0].keys()

    def values(self) -> Any:
        """Return gridded feature group DataFrames."""
        return self[0].values()

    def items(self) -> Any:
        """Return (group_name, DataFrame) items."""
        return self[0].items()

def sliding_window_view(a: np.ndarray, window_shape: int) -> np.ndarray:
    """
    Create a sliding window view of an array.

    Args:
        a (np.ndarray): Input array.
        window_shape (int): Shape of the sliding window.

    Returns:
        np.ndarray: Sliding window view of the array.
    """
    from numpy.lib.stride_tricks import sliding_window_view as np_swv
    if hasattr(a, "values"):
        windows = np_swv(a.values, window_shape, axis=0)
    else:
        windows = np_swv(a, window_shape, axis=0)
    # Force contiguous memory layout for safe PyTorch tensor conversion
    return np.ascontiguousarray(np.moveaxis(windows, -1, 1))

def infer_temporal_metadata(ds: xr.Dataset, time_dim: str = "time") -> Dict[str, Any]:
    """
    Infer temporal coverage bounds and native frequency from xarray Dataset time coordinates.

    Mathematical docstring:
        Input time dimension shape: (T_timestamps,)
        Returns mapping with 'native_temporal_freq': Optional[str], 'time_start': pd.Timestamp, 'time_end': pd.Timestamp.
    """
    time_coord = ds[time_dim]
    time_start = pd.Timestamp(time_coord[0].values)
    time_end = pd.Timestamp(time_coord[-1].values)
    
    time_index = pd.DatetimeIndex(time_coord.values)
    inferred_freq = None
    if len(time_index) >= 3:
        try:
            inferred_freq = pd.infer_freq(time_index)
        except Exception:
            inferred_freq = None
    
    if inferred_freq is None and len(time_index) > 1:
        try:
            deltas = pd.Series(time_index).diff().dropna()
            median_delta = deltas.median()
            inferred_freq = pd.tseries.frequencies.to_offset(median_delta).freqstr
        except Exception:
            inferred_freq = None

    return {
        "native_temporal_freq": inferred_freq,
        "time_start": time_start,
        "time_end": time_end,
    }

def infer_spatial_metadata(ds: xr.Dataset, lat_dim: str = "lat", lon_dim: str = "lon") -> Dict[str, Tuple[float, float]]:
    """
    Infer spatial bounding box coverage from xarray Dataset coordinates.

    Mathematical docstring:
        Input latitude shape: (N_lat,), longitude shape: (N_lon,)
        Returns mapping with 'lat_bounds': (lat_min, lat_max), 'lon_bounds': (lon_min, lon_max).
    """
    lat = ds[lat_dim]
    lon = ds[lon_dim]
    
    return {
        "lat_bounds": (float(lat.min().values), float(lat.max().values)),
        "lon_bounds": (float(lon.min().values), float(lon.max().values)),
    }

def check_station_grid_resolution_compatibility(
    station_freq: Optional[str],
    grid_freq: Optional[str],
    product_name: str,
) -> None:
    """
    Verify temporal resolution compatibility between station observations and gridded products.
    """
    if station_freq is None or grid_freq is None:
        return
    try:
        station_td = pd.Timedelta(pd.tseries.frequencies.to_offset(station_freq))
        grid_td = pd.Timedelta(pd.tseries.frequencies.to_offset(grid_freq))
    except Exception:
        return

    if grid_td < station_td:
        warnings.warn(
            f"[{product_name}] Grid resolution ({grid_freq}) is finer than "
            f"station resolution ({station_freq}). Constraints or modeling "
            f"at sub-station resolution will rely on assumptions.",
            UserWarning,
        )
    elif station_td < grid_td:
        warnings.warn(
            f"[{product_name}] Station resolution ({station_freq}) is finer than "
            f"grid resolution ({grid_freq}). You are requesting sub-native "
            f"grid resolution which may involve unsafe upsampling.",
            UserWarning,
        )

def validate_gridded_dataset(
    ds: xr.Dataset,
    product_name: str,
    requested_freq: Optional[str],
    bbox: Optional[Tuple[float, float, float, float]],
) -> None:
    """
    Validate spatial bounding box containment and temporal frequency alignment of gridded datasets.
    """
    # Validate spatial bounds
    if bbox is not None:
        spatial_meta = infer_spatial_metadata(ds)
        lat_min, lon_min, lat_max, lon_max = bbox
        grid_lat_min, grid_lat_max = spatial_meta["lat_bounds"]
        grid_lon_min, grid_lon_max = spatial_meta["lon_bounds"]
        
        # If the requested bbox is entirely outside the dataset bounds, it's an error.
        if (lat_max < grid_lat_min or lat_min > grid_lat_max or 
            lon_max < grid_lon_min or lon_min > grid_lon_max):
            raise ValueError(
                f"[{product_name}] Requested bbox {bbox} is completely outside "
                f"the dataset's native bounds: lats={spatial_meta['lat_bounds']}, "
                f"lons={spatial_meta['lon_bounds']}."
            )

    # Validate temporal frequencies
    temporal_meta = infer_temporal_metadata(ds)
    native_freq = temporal_meta["native_temporal_freq"]
    
    if native_freq and requested_freq:
        check_station_grid_resolution_compatibility(requested_freq, native_freq, product_name)

def df_add_missing_columns(
    df: pd.DataFrame,
    col0: Sequence[Any],
    col1: Sequence[Any],
    fill_value: Any = np.nan,
) -> pd.DataFrame:
    """
    Ensure 2-level MultiIndex DataFrame contains full Cartesian product of station and parameter columns.

    Mathematical docstring:
        Input shape: (T_timestamps, N_existing_columns)
        Output shape: (T_timestamps, len(col0) * len(col1)) with missing entries populated by fill_value.
    """
    idx = pd.MultiIndex.from_product([col0, col1], names=['station_id', 'parameter'])
    missing = idx.difference(df.columns)
    if not missing.empty:
        df_missing = pd.DataFrame(fill_value, index=df.index, columns=missing)
        df = pd.concat([df, df_missing], axis=1)
        df = df.sort_index(axis=1)
    return df

class MasharikiWeatherDataset:
    """
    A comprehensive dataset class for the Mashariki Weather dataset, 
    providing direct access to historical weather observations and 
    topographical data for the African continent.
    """ 
    __version__ = "0.1.9"
    
    # Hardcoded Test Set Split Anchor
    test_set_start = pd.Timestamp("2023-01-01", tz="UTC")

    VALID_GRIDDED_GROUPS = [
        "era5", "imerg", "chirps", "tahmo", "topography", "tamsat",
        "cams", "merra", "merra2", "mera", "open_meteo", "openmeteo", "nasa_power"
    ]

    def _standardize_coordinates(self, ds: Optional[xr.Dataset]) -> Optional[xr.Dataset]:
        """
        Standardizes coordinate orientation across disparate datasets (e.g. CHIRPS vs IMERG).
        Ensures that latitudes and longitudes are monotonically ascending,
        so users can reliably use ds.sel(lat=slice(min, max), lon=slice(min, max)).
        """
        if ds is None:
            return None
            
        lat_dim = 'lat' if 'lat' in ds.dims else ('latitude' if 'latitude' in ds.dims else None)
        lon_dim = 'lon' if 'lon' in ds.dims else ('longitude' if 'longitude' in ds.dims else None)

        if lat_dim and len(ds[lat_dim]) > 1:
            if ds[lat_dim].values[0] > ds[lat_dim].values[-1]:
                ds = ds.isel({lat_dim: slice(None, None, -1)})

        if lon_dim and len(ds[lon_dim]) > 1:
            if ds[lon_dim].values[0] > ds[lon_dim].values[-1]:
                ds = ds.isel({lon_dim: slice(None, None, -1)})
                
        return ds

    def _validate_groups(self, groups: List[str]) -> None:
        """Validate that the requested groups are known and give helpful error if not."""
        for group in groups:
            if group not in self.VALID_GRIDDED_GROUPS:
                matches = difflib.get_close_matches(group, self.VALID_GRIDDED_GROUPS, n=1, cutoff=0.6)
                suggestion = f" Did you mean '{matches[0]}' instead of '{group}'?" if matches else ""
                raise ValueError(f"Invalid group '{group}'.{suggestion} Valid groups are: {self.VALID_GRIDDED_GROUPS}.")

    def _validate_years(self, years: List[int]) -> List[int]:
        """
        Validate requested years against the configured dataset range.
        Emits a warning for out-of-range years and raises ValueError if none remain.

        Args:
            years: List of requested years.

        Returns:
            List of valid years within the configured range.
        """
        valid = [y for y in years if y in self.years]
        invalid = [y for y in years if y not in self.years]
        if invalid:
            warnings.warn(
                f"Requested years {invalid} are outside the configured dataset range "
                f"{self.years}. These will be skipped. "
                f"To include them, pass `years={sorted(set(self.years + invalid))}` to the constructor.",
                UserWarning
            )
        if not valid:
            raise ValueError(
                f"None of the requested years {years} are within the configured range {self.years}."
            )
        return valid

    def __init__(
        self,
        repo_id: str = "DeKUT-DSAIL/weather-data",
        token: Optional[str] = None,
        root_dir: Optional[Union[str, Path]] = None,
        years: Optional[Sequence[int]] = None,
        source_obs: Union[str, List[str]] = "tahmo",
        freq: Optional[str] = None,
        test_set_start: Optional[pd.Timestamp] = None,
        parameters: Optional[Union[str, Sequence[str]]] = None,
        imputation_method: Optional[Union[str, Callable[[pd.DataFrame], pd.DataFrame]]] = None,
        imputation_limit: Optional[int] = None,
        pad_missing_values: bool = True,
        aggregation_methods: Optional[Dict[str, str]] = None,
        tabular_parquet_path: Optional[str] = None,
        rechunk: bool = False,
        swap: bool = False,
        use_hf: bool = False,
        root: Optional[Union[str, Path]] = None,
        prefer_zarr: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize MasharikiWeatherDataset with PeakWeather-inspired offline-first architecture.

        Primary data loading path is local disk (root_dir), reading Parquet files in
        root_dir/obs/{source} and Zarr groups in root_dir/gridded/{group}.
        If local partition files are missing, programmatically triggers Dagster extraction.
        Remote HuggingFace streaming is opt-in via use_hf=True or explicit token provision.

        Args:
            repo_id (str): Hugging Face repository ID.
            token (Optional[str]): Hugging Face authentication token.
            root_dir (Optional[Union[str, Path]]): Local directory storing observations, gridded datasets, and caches.
            years (Optional[Sequence[int]]): Years to include in dataset partitions.
            source_obs (Union[str, List[str]]): Source of observations ('tahmo', 'ghcn', 'cds_insitu', etc.).
            freq (Optional[str]): Temporal frequency of the data (e.g. 'h', 'D', or None for native resolution).
            test_set_start (Optional[pd.Timestamp]): Temporal split boundary for holdout testing.
            parameters (Optional[Union[str, Sequence[str]]]): Weather parameters to include or filter.
            imputation_method (Optional[Union[str, Callable[[pd.DataFrame], pd.DataFrame]]]):
                Imputation method ('locf', 'temporal_linear', 'diurnal_climatology', 'zero', None, or callable).
            imputation_limit (Optional[int]): Step limit on forward-fill or linear interpolation.
            pad_missing_values (bool): Whether to pad missing sensor coordinates.
            aggregation_methods (Optional[Dict[str, str]]): Per-variable aggregation overrides.
            tabular_parquet_path (Optional[str]): Path to cached tabular observations parquet file.
            rechunk (bool): Whether to rechunk gridded stores to daily steps upon open.
            swap (bool): Whether to perform atomic file swap upon rechunking.
            use_hf (bool): Whether to explicitly query remote HuggingFace filesystem if local files are absent.
            root (Optional[Union[str, Path]]): Convenience alias for root_dir.
            prefer_zarr (bool): Whether to prioritize Zarr reading over Parquet for TAHMO observations.
            **kwargs (Any): Additional configuration parameters.
        """
        actual_root = root if root is not None else (root_dir if root_dir is not None else "./mashariki_data")
        self.root_dir = Path(actual_root)
        self.repo_id = repo_id
        self.token = token

        raw_sources = [source_obs] if isinstance(source_obs, str) else list(source_obs)
        seen_sources = set()
        self.source_obs = []
        for s in raw_sources:
            norm_s = _normalize_source_name(s)
            if norm_s not in seen_sources:
                seen_sources.add(norm_s)
                self.source_obs.append(norm_s)
        self.pad_missing_values = pad_missing_values
        self.years = list(years) if years is not None else [2018, 2019, 2020, 2021, 2022, 2023, 2024]
        self.aggregation_methods = aggregation_methods or {}
        self.tabular_parquet_path = tabular_parquet_path
        if test_set_start is not None:
            self.test_set_start = test_set_start
        self.requested_parameters = parameters
        self.imputation_method = imputation_method
        self.imputation_limit = imputation_limit
        self.rechunk = rechunk
        self.swap = swap
        self.prefer_zarr = prefer_zarr or kwargs.get("use_zarr", False)
        self.use_hf = use_hf or (token is not None)

        if freq is not None:
            try:
                self.freq = pd.tseries.frequencies.to_offset(freq).freqstr
            except ValueError:
                raise ValueError(f"Value '{freq}' is not a valid pandas frequency string.")
        else:
            self.freq = None

        if self.use_hf:
            try:
                self.fs = fsspec.filesystem("hf", token=token)
            except Exception as e:
                logger.warning("Could not initialize HuggingFace filesystem: %s", e)
                self.fs = None
        else:
            self.fs = None

        self._obs_zarr = {}
        self._column_sources: Dict[Tuple[str, str], str] = {}
        self._topo_ds = None

        # Check local Zarr candidates when prefer_zarr is requested or when parquet is absent
        for src in self.source_obs:
            norm_s = _normalize_source_name(src)
            src_dir = self._resolve_obs_source_dir(src)
            has_parquet = src_dir.exists() and any(src_dir.glob("*.parquet"))
            if self.prefer_zarr or not has_parquet:
                candidate_paths = [
                    kwargs.get("zarr_path"),
                    kwargs.get(f"{norm_s}_zarr_path"),
                    kwargs.get(f"{src}_zarr_path"),
                    self.root_dir / "obs" / f"{norm_s}.zarr",
                    self.root_dir / "obs" / f"{norm_s}.zarr.zip",
                    self.root_dir / "obs" / f"{src}.zarr",
                    self.root_dir / "obs" / f"{src}.zarr.zip",
                ]
                for cp in candidate_paths:
                    if cp and Path(cp).exists():
                        try:
                            if str(cp).endswith(".zip"):
                                zds = self._open_zipped_zarr(str(cp), lazy=True)
                            else:
                                try:
                                    zds = xr.open_zarr(str(cp), consolidated=True)
                                except Exception:
                                    zds = xr.open_zarr(str(cp), consolidated=False)
                            if zds is not None:
                                self._obs_zarr[src] = zds
                                self._obs_zarr[norm_s] = zds
                                break
                        except Exception as err:
                            logger.warning("Could not open local Zarr store at %s: %s", cp, err)

        # Check local partition files on disk
        missing_partition_years = []
        for src in self.source_obs:
            norm_s = _normalize_source_name(src)
            if src in self._obs_zarr or norm_s in self._obs_zarr:
                continue
            src_dir = self._resolve_obs_source_dir(src)
            for yr in self.years:
                if not (src_dir / f"{yr}.parquet").exists():
                    missing_partition_years.append(yr)
        missing_partition_years = sorted(list(set(missing_partition_years)))

        found_local_obs = bool(self._obs_zarr)
        if not found_local_obs:
            for src in self.source_obs:
                src_dir = self._resolve_obs_source_dir(src)
                if src_dir.exists() and any(src_dir.glob("*.parquet")):
                    found_local_obs = True
                    break
        if not found_local_obs and (self.root_dir / "obs").exists():
            if any((self.root_dir / "obs").glob("*.parquet")):
                found_local_obs = True

        # Programmatic Dagster extraction trigger when missing local data
        if missing_partition_years and not found_local_obs and not self.use_hf and not (self.tabular_parquet_path and os.path.exists(self.tabular_parquet_path)):
            try:
                from .extract.apis.dagster_pipeline import materialize_extraction
                materialize_extraction(
                    sources=self.source_obs,
                    years=missing_partition_years,
                    root_dir=self.root_dir,
                )
                for src in self.source_obs:
                    src_dir = self._resolve_obs_source_dir(src)
                    if src_dir.exists() and any(src_dir.glob("*.parquet")):
                        found_local_obs = True
                        break
            except Exception as err:
                logger.debug("Automatic Dagster materialization skipped: %s", err)

        # Local cache control flow
        if self._is_local_cache_complete():
            self._load_local_registries()
        elif self.use_hf and self.fs is not None:
            for src in self.source_obs:
                os.makedirs(f"{self.root_dir}/obs/{src}", exist_ok=True)
                ds = self._open_zipped_zarr(f"data/obs/{src}.zarr.zip", lazy=True)
                if ds is None:
                    raise ConnectionError(f"Failed to connect to the observation dataset: {src}")
                self._obs_zarr[src] = ds
            self._load_registries()
        elif found_local_obs:
            self._load_or_create_local_registries()
        else:
            logger.info(
                "No local observational data found in %s. Initialized empty dataset. "
                "Use pull_station_data(...) or pull_weather_data(...) to fetch observations locally.",
                self.root_dir,
            )
            self.stations_table = pd.DataFrame(
                columns=["latitude", "longitude", "elevation", "station_name", "network", "country", "start_date", "end_date", "sensor_count"]
            ).set_index(pd.Index([], name="station_id"))
            self.parameters_table = pd.DataFrame(
                columns=["canonical_name", "standard_name", "units", "physical_type", "resampling_method", "description"]
            ).set_index(pd.Index([], name="parameter"))
            self.observations = ObservationDataFrame()
            self.mask = ObservationDataFrame()

        if self.tabular_parquet_path and os.path.exists(self.tabular_parquet_path):
            self._load_cached_parquet()
        elif hasattr(self, "observations") and not self.observations.empty:
            pass
        elif self._is_local_cache_complete() or (self.use_hf and self._obs_zarr) or bool(self._obs_zarr):
            self._prepare_observations()
        elif hasattr(self, "stations_table") and not self.stations_table.empty:
            self._prepare_observations()

        self.interpolate_topography()

    def _resolve_obs_source_dir(self, source: str) -> Path:
        """
        Locate the local observation directory on disk for a given source,
        checking canonical names and source aliases.
        """
        norm = _normalize_source_name(source)
        candidates = [norm]
        if norm == "ghcn":
            candidates.extend(["ghcnd", "ghcn_daily", "ghcn_hourly"])
        elif norm == "cds_insitu":
            candidates.extend(["cds", "insitu", "cds-insitu"])
        elif norm == "tahmo":
            candidates.extend(["tahmo_station", "tahmo_stations"])
        if source not in candidates:
            candidates.append(source)

        for c in candidates:
            p = self.root_dir / "obs" / c
            if p.exists() and p.is_dir():
                return p
        return self.root_dir / "obs" / norm

    def _load_or_create_local_registries(self) -> None:
        """
        Load metadata registries from local disk if available, or construct them
        dynamically by inspecting local Parquet observation files.
        """
        st_candidates = [
            Path(self.root_dir) / "metadata" / "installation.parquet",
            Path(self.root_dir) / "installation.parquet",
            Path(self.root_dir) / "stations_registry.parquet",
        ]
        pr_candidates = [
            Path(self.root_dir) / "metadata" / "parameters.parquet",
            Path(self.root_dir) / "parameters.parquet",
            Path(self.root_dir) / "parameters_registry.parquet",
        ]
        st_exists = any(p.exists() for p in st_candidates)
        pr_exists = any(p.exists() for p in pr_candidates)

        if st_exists and pr_exists:
            self._load_local_registries()
        else:
            self.stations_table = pd.DataFrame(
                columns=["latitude", "longitude", "elevation", "station_name", "network", "country", "start_date", "end_date", "sensor_count"]
            ).set_index(pd.Index([], name="station_id"))
            self.parameters_table = pd.DataFrame(
                columns=["canonical_name", "standard_name", "units", "physical_type", "resampling_method", "description"]
            ).set_index(pd.Index([], name="parameter"))
            self._prepare_observations()
            if hasattr(self, "observations") and not self.observations.empty:
                stations = self.observations.columns.get_level_values(0).unique()
                params = self.observations.columns.get_level_values(1).unique()

                st_rows = []
                for st in stations:
                    st_rows.append({
                        "station_id": st,
                        "latitude": np.nan,
                        "longitude": np.nan,
                        "elevation": np.nan,
                        "station_name": str(st),
                        "network": self.source_obs[0] if self.source_obs else "tahmo",
                        "country": "KE",
                        "start_date": self.observations.index[0] if len(self.observations) > 0 else pd.NaT,
                        "end_date": self.observations.index[-1] if len(self.observations) > 0 else pd.NaT,
                        "sensor_count": 1,
                    })
                self.stations_table = pd.DataFrame(st_rows).set_index("station_id")

                pr_rows = []
                for p in params:
                    base_name = p.split("_")[0] if "_" in p else p
                    meta = CANONICAL_VARIABLE_METADATA.get(p, CANONICAL_VARIABLE_METADATA.get(base_name, {}))
                    pr_rows.append({
                        "parameter": p,
                        "canonical_name": meta.get("canonical_name", p),
                        "standard_name": meta.get("standard_name", p),
                        "units": meta.get("units", "unknown"),
                        "physical_type": meta.get("physical_type", "PHYSICAL"),
                        "resampling_method": meta.get("resampling_method", "mean"),
                        "description": meta.get("description", f"Observed parameter {p}"),
                    })
                self.parameters_table = pd.DataFrame(pr_rows).set_index("parameter")

                meta_dir = Path(self.root_dir) / "metadata"
                meta_dir.mkdir(parents=True, exist_ok=True)
                self.stations_table.to_parquet(meta_dir / "installation.parquet")
                self.parameters_table.to_parquet(meta_dir / "parameters.parquet")
                self.stations_table.to_parquet(Path(self.root_dir) / "stations_registry.parquet")
                self.parameters_table.to_parquet(Path(self.root_dir) / "parameters_registry.parquet")

                self._filter_requested_parameters()

    # --- PEAKWEATHER OOP DATASET PROPERTIES ---
    @property
    def root(self) -> Path:
        """Convenience alias for root_dir (MeteoSwiss PeakWeather API)."""
        return Path(self.root_dir)

    @property
    def available_parameters(self) -> List[str]:
        """
        List of all harmonized weather parameter identifiers available in
        local disk catalogs or loaded observations.
        """
        params = set()
        pr_candidates = [
            Path(self.root_dir) / "metadata" / "parameters.parquet",
            Path(self.root_dir) / "parameters.parquet",
            Path(self.root_dir) / "parameters_registry.parquet",
        ]
        for pr_file in pr_candidates:
            if pr_file.exists():
                try:
                    pr_df = pd.read_parquet(pr_file)
                    params.update(pr_df.index.tolist())
                except Exception:
                    pass
        if hasattr(self, "parameters_table") and self.parameters_table is not None and not self.parameters_table.empty:
            params.update(self.parameters_table.index.tolist())
        if hasattr(self, "observations") and self.observations is not None and not self.observations.empty:
            if isinstance(self.observations.columns, pd.MultiIndex):
                params.update(self.observations.columns.get_level_values(1).unique().tolist())
            else:
                params.update(self.observations.columns.tolist())
        return sorted(list(params))

    @property
    def available_years(self) -> List[int]:
        """
        List of temporal calendar years available in local disk partition files.
        """
        years_found = set()
        obs_dir = Path(self.root_dir) / "obs"
        if obs_dir.exists():
            for p in obs_dir.rglob("*.parquet"):
                stem = p.stem
                if stem.isdigit() and len(stem) == 4 and 1900 <= int(stem) <= 2100:
                    years_found.add(int(stem))
                else:
                    for part in stem.split("_"):
                        if part.isdigit() and len(part) == 4 and 1900 <= int(part) <= 2100:
                            years_found.add(int(part))
        gridded_dir = Path(self.root_dir) / "gridded"
        if gridded_dir.exists():
            for p in gridded_dir.rglob("*.zarr"):
                for part in p.stem.split("_"):
                    if part.isdigit() and len(part) == 4 and 1900 <= int(part) <= 2100:
                        years_found.add(int(part))
        if hasattr(self, "observations") and self.observations is not None and not self.observations.empty:
            if isinstance(self.observations.index, pd.DatetimeIndex):
                years_found.update(self.observations.index.year.unique().tolist())
        return sorted(list(years_found))

    @property
    def installation(self) -> pd.DataFrame:
        """Station metadata inventory catalog (installation.parquet)."""
        if hasattr(self, "stations_table") and self.stations_table is not None:
            return self.stations_table
        return pd.DataFrame()

    @installation.setter
    def installation(self, df: pd.DataFrame) -> None:
        self.stations_table = df

    @property
    def stations(self) -> StationList:
        """List of active weather station identifiers."""
        if hasattr(self, "stations_table") and self.stations_table is not None:
            return StationList(self.stations_table.index.tolist())
        return StationList([])

    @stations.setter
    def stations(self, val: Sequence[str]) -> None:
        if hasattr(self, "stations_table") and self.stations_table is not None:
            self.stations_table = self.stations_table.reindex(val)
        else:
            self.stations_table = pd.DataFrame(index=pd.Index(val, name="station_id"))

    @property
    def parameters(self) -> ParameterList:
        """List of currently active/requested weather parameters."""
        if hasattr(self, "parameters_table") and self.parameters_table is not None:
            return ParameterList(self.parameters_table.index.tolist())
        return ParameterList([])

    @parameters.setter
    def parameters(self, val: Sequence[str]) -> None:
        if hasattr(self, "parameters_table") and self.parameters_table is not None:
            self.parameters_table = self.parameters_table.reindex(val)
        else:
            self.parameters_table = pd.DataFrame(index=pd.Index(val, name="parameter"))

    @property
    def num_stations(self) -> int:
        return len(self.stations_table)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters_table)

    @property
    def num_time_steps(self) -> int:
        return len(self.observations)

    @property
    def missing_values(self) -> pd.Series:
        if not hasattr(self, "mask") or self.mask is None or self.mask.empty:
            return pd.Series(dtype="float32", index=self.parameters)
        valid_columns = self.mask.loc[:, self.mask.any()]
        mean_missing = {}
        for param in self.parameters:
            if param in valid_columns.columns.get_level_values(1):
                mean_missing[param] = 1 - valid_columns.xs(param, axis=1, level=1).values.mean()
            else:
                mean_missing[param] = 1.0

        out = pd.Series(mean_missing, name="missing_perc", dtype="float32")
        return out.reindex(self.parameters)

    def load(self) -> ObservationDataFrame:
        """
        Eagerly loads and prepares processed, harmonized, quality-filtered observations into active memory.

        Mathematical docstring:
            Observation tensor shape: (T_timestamps, N_stations, C_parameters).
            Returns tabular MultiIndex representation of shape (T_timestamps, N_stations * C_parameters).

        Returns:
            ObservationDataFrame: Harmonized observations DataFrame with MultiIndex columns (station_id, parameter).
        """
        if not hasattr(self, "observations") or self.observations is None or self.observations.empty:
            self._prepare_observations()
        return self.observations

    def load_raw(self) -> ObservationDataFrame:
        """
        Loads raw unresampled, unmasked observations DataFrame for exploratory quality control.

        Mathematical docstring:
            Returns raw tabular MultiIndex representation of shape (T_raw_timestamps, N_stations * C_raw_parameters).

        Returns:
            ObservationDataFrame: Unresampled, unmasked raw observations DataFrame.
        """
        return self._load_raw_observations()

    def _load_raw_observations(self) -> ObservationDataFrame:
        """
        Read raw observation files without temporal resampling or quality masking/imputation.
        """
        yearly_dfs = []
        for src in self.source_obs:
            norm_s = _normalize_source_name(src)
            src_dir = self._resolve_obs_source_dir(src)
            zds = self._obs_zarr.get(src) if src in self._obs_zarr else self._obs_zarr.get(norm_s)
            use_zarr = (self.prefer_zarr and zds is not None) or (zds is not None and not (src_dir.exists() and any(src_dir.glob("*.parquet"))))
            for year in self.years:
                p_path = src_dir / f"{year}.parquet"
                if use_zarr:
                    try:
                        yearly_ds = zds.sel(time=str(year)).load()
                        yearly_dfs.append(_tag_source(yearly_ds.to_dataframe().reset_index(), norm_s))
                    except (KeyError, ValueError):
                        try:
                            yearly_ds = zds.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).load()
                            if len(yearly_ds.time) > 0:
                                yearly_dfs.append(_tag_source(yearly_ds.to_dataframe().reset_index(), norm_s))
                        except Exception:
                            continue
                elif p_path.exists():
                    yearly_dfs.append(_tag_source(pd.read_parquet(p_path), norm_s))
                elif zds is not None:
                    try:
                        yearly_ds = zds.sel(time=str(year)).load()
                        yearly_dfs.append(_tag_source(yearly_ds.to_dataframe().reset_index(), norm_s))
                    except KeyError:
                        continue
            if not yearly_dfs and zds is not None and use_zarr:
                try:
                    df_all = zds.to_dataframe().reset_index()
                    yearly_dfs.append(_tag_source(df_all, norm_s))
                except Exception:
                    pass
            if not yearly_dfs and src_dir.exists():
                for pf in sorted(src_dir.glob("*.parquet")):
                    try:
                        df_p = pd.read_parquet(pf)
                        if "station_id" not in df_p.columns and "station" not in df_p.columns:
                            stem_code = pf.stem.split("_")[0]
                            df_p["station_id"] = stem_code
                        yearly_dfs.append(_tag_source(df_p, norm_s))
                    except Exception:
                        pass

        if not yearly_dfs:
            return ObservationDataFrame()

        full_df = pd.concat(yearly_dfs)
        full_df = _normalize_time_column(full_df)
        if "station" in full_df.columns:
            full_df = full_df.rename(columns={"station": "station_id"})
        full_df = full_df.drop(columns=["__source__"], errors="ignore")

        param_cols = [c for c in full_df.columns if c not in ["time", "station_id"]]
        obs_df = full_df.pivot(index="time", columns="station_id", values=param_cols)
        obs_df = obs_df.reorder_levels([1, 0], axis=1).sort_index(axis=1)
        obs_df.columns.names = ["station_id", "parameter"]
        return ObservationDataFrame(obs_df)

    def resample(
        self,
        freq: str,
        aggregation_methods: Optional[Dict[str, str]] = None,
    ) -> "MasharikiWeatherDataset":
        r"""
        Resample the dataset observations to a new temporal frequency while preserving
        physical conservation laws (flux summation, thermodynamic state averaging).

        Mathematical docstring:
            Input observation shape: (T_in, N_stations * C_parameters) at freq f_in.
            Output observation shape: (T_out, N_stations * C_parameters) at freq f_out.
            Physical conservation:
                - Precipitation and radiative flux variables are summed:
                  Y_flux[t] = \sum_{k \in \Delta t} X_flux[k] with min_count=1.
                - Thermodynamic state variables (temperature, dewpoint, pressure, wind) are averaged:
                  Y_state[t] = \frac{1}{|\Delta t|} \sum_{k \in \Delta t} X_state[k].
                - Quality mask:
                  M_out[t] = \bigvee_{k \in \Delta t} M_in[k].

        Args:
            freq: Target temporal frequency string (e.g. 'h', '3h', 'D').
            aggregation_methods: Optional custom variable-to-method dictionary overrides.

        Returns:
            MasharikiWeatherDataset: A new resampled dataset instance.
        """
        import copy
        new_ds = copy.copy(self)
        try:
            new_freq = pd.tseries.frequencies.to_offset(freq).freqstr
        except ValueError:
            raise ValueError(f"Value '{freq}' is not a valid pandas frequency string.")
        new_ds.freq = new_freq

        current_agg = dict(getattr(self, "aggregation_methods", {}) or {})
        if aggregation_methods is not None:
            current_agg.update(aggregation_methods)
        new_ds.aggregation_methods = current_agg

        if hasattr(self, "observations") and self.observations is not None and not self.observations.empty:
            agg_dict = new_ds._build_aggregation_dict(self.observations.columns)
            res_obj = self.observations.resample(new_freq)

            methods: Dict[str, List[Any]] = {}
            for col, m in agg_dict.items():
                methods.setdefault(m, []).append(col)

            resampled_dfs = []
            for method, cols in methods.items():
                if method == "sum":
                    resampled_dfs.append(res_obj[cols].sum(min_count=1))
                elif method == "max":
                    resampled_dfs.append(res_obj[cols].max())
                else:
                    resampled_dfs.append(res_obj[cols].mean())

            new_obs = pd.concat(resampled_dfs, axis=1).sort_index(axis=1)
            new_ds.observations = ObservationDataFrame(new_obs)

            if hasattr(self, "mask") and self.mask is not None and not self.mask.empty:
                new_ds.mask = ObservationDataFrame(self.mask.resample(new_freq).max().astype(bool))
            else:
                new_ds.mask = ObservationDataFrame(~new_ds.observations.isna())

        return new_ds

    def _is_local_cache_complete(self) -> bool:
        has_stations = any((Path(self.root_dir) / p).exists() for p in [
            "metadata/installation.parquet", "installation.parquet", "stations_registry.parquet"
        ])
        has_params = any((Path(self.root_dir) / p).exists() for p in [
            "metadata/parameters.parquet", "parameters.parquet", "parameters_registry.parquet"
        ])
        if not (has_stations and has_params):
            return False

        for src in self.source_obs:
            norm_s = _normalize_source_name(src)
            if self.prefer_zarr and (src in self._obs_zarr or norm_s in self._obs_zarr):
                continue
            src_dir = self._resolve_obs_source_dir(src)
            for year in self.years:
                p_path = src_dir / f"{year}.parquet"
                if not p_path.exists():
                    if src not in self._obs_zarr and norm_s not in self._obs_zarr:
                        return False

        return True
  
    def _load_cached_parquet(self) -> None:
        if self.tabular_parquet_path and os.path.exists(self.tabular_parquet_path):
            self.observations = pd.read_parquet(self.tabular_parquet_path)
            
            self.mask = ~self.observations.isna()
            if self.imputation_method == "locf":
                self.observations = self.observations.ffill()
            elif self.imputation_method == "zero":
                self.observations = self.observations.fillna(0)
            elif self.imputation_method is not None:
                raise ValueError(f"Invalid imputation method: {self.imputation_method}.")
            
            stations = self.observations.columns.get_level_values(0).unique()
            parameters = self.observations.columns.get_level_values(1).unique()
            self.stations_table = pd.DataFrame(index=stations)
            self.parameters_table = pd.DataFrame(index=parameters)
            
        else:
            self.observations = pd.DataFrame()
            self.mask = pd.DataFrame()
            self.stations_table = pd.DataFrame()
            self.parameters_table = pd.DataFrame()

    def _open_zipped_zarr(self, rel_path: str, lazy: bool = True) -> Optional[xr.Dataset]:
        """
        Open a zipped Zarr dataset from Hugging Face or local path.
        """
        # 1. Prioritize Local Zipped Archives if available (Offline Mode)
        local_path_1 = f"{self.root_dir}/{rel_path.replace('data/', '')}"
        local_path_2 = f"{self.root_dir}/{rel_path}"
        
        local_zip = None
        if os.path.exists(local_path_1):
            local_zip = local_path_1
        elif os.path.exists(local_path_2):
            local_zip = local_path_2
            
        if local_zip:
            zip_mapper = fsspec.get_mapper(f"zip://::file://{os.path.abspath(local_zip)}")
            chunks = {} if lazy else None
            try:
                ds = xr.open_zarr(zip_mapper, chunks=chunks, consolidated=True, decode_timedelta=False, zarr_format=2)
            except Exception:
                ds = xr.open_zarr(zip_mapper, chunks=chunks, consolidated=False, decode_timedelta=False, zarr_format=2)
            return self._standardize_coordinates(ds)

        # 2. Fallback to Hugging Face Remote Streaming if enabled
        if not getattr(self, "use_hf", False) or self.fs is None:
            return None

        full_path = f"hf://datasets/{self.repo_id}/{rel_path}"
        try:
            if not self.fs.exists(full_path):
                logger.warning("Remote dataset not found: %s", full_path)
                return None
                
            remote_file = self.fs.open(full_path)
            zip_mapper = fsspec.get_mapper("zip://", fo=remote_file)
            chunks = {} if lazy else None
            
            try:
                ds = xr.open_zarr(zip_mapper, chunks=chunks, consolidated=True, decode_timedelta=False, zarr_format=2)
            except Exception:
                ds = xr.open_zarr(zip_mapper, chunks=chunks, consolidated=False, decode_timedelta=False, zarr_format=2)
            return self._standardize_coordinates(ds)
            
        except Exception as e:
            logger.warning("Could not open remote %s: %s", rel_path, e)
            return None

    def _open_gridded_zarr(self, group: str, year: int, lazy: bool = True) -> Optional[xr.Dataset]:
        """
        Traffic Controller: Dynamically resolves I/O paths based on optimization tier.
        Supports Automated Daily Re-chunking if self.rechunk=True.
        """
        optimized_path = f"{self.root_dir}/optimized/{group}/{year}.zarr"
        local_path = f"{self.root_dir}/gridded/{group}/{year}.zarr"
        
        # 1. Prioritize Daily Optimized Tier (O(1) Fetch)
        if os.path.exists(optimized_path):
            try:
                ds = xr.open_zarr(optimized_path, chunks={} if lazy else None, 
                                  consolidated=True, decode_timedelta=False, zarr_format=2)
            except Exception:
                ds = xr.open_zarr(optimized_path, chunks={} if lazy else None, 
                                  consolidated=False, decode_timedelta=False, zarr_format=2)
            return self._standardize_coordinates(ds)
        
        # 2. Check Raw Download Tier
        if os.path.exists(local_path):
            # Automated Optimization Hook
            if self.rechunk:
                rechunk_dataset(local_path, optimized_path, time_chunks=24, swap=getattr(self, "swap", False))
                return self._open_gridded_zarr(group, year, lazy)
                
            chunks = {} if lazy else None
            try:
                ds = xr.open_zarr(local_path, chunks=chunks, 
                                  consolidated=True, decode_timedelta=False, zarr_format=2)
            except Exception:
                ds = xr.open_zarr(local_path, chunks=chunks, 
                                  consolidated=False, decode_timedelta=False, zarr_format=2)
            return self._standardize_coordinates(ds)
                
        # 3. Fallback to Hugging Face lazy streaming pipeline if enabled
        if getattr(self, "use_hf", False) and self.fs is not None:
            return self._open_zipped_zarr(f"data/{group}/{year}.zarr.zip", lazy)
        return None
        
    def download_gridded_datasets(self, groups: List[str], years: Optional[Union[int, List[int]]] = None, 
                                 force: bool = False, optimize: bool = False, swap: Optional[bool] = None,
                                 num_workers: int = 16) -> None:
        """
        Single-Pass Ingest: Downloads Hugging Face Zarr arrays physically onto disk.
        
        Args:
            groups (List[str]): List of gridded data groups.
            years (Optional[Union[int, List[int]]]): Years to download.
            force (bool): Whether to overwrite existing files.
            optimize (bool): Whether to stream into training-optimized Daily (24-step) format.
            swap (Optional[bool]): Whether to delete raw cache if optimized succeeds.
            num_workers (int): Number of parallel HTTP streams for download.
        """
        self._validate_groups(groups)
        target_years = [years] if isinstance(years, int) else (years or self.years)
        target_years = self._validate_years(target_years)
        should_swap = swap if swap is not None else getattr(self, "swap", False)
        
        for group in groups:
            for year in target_years:
                local_path = f"{self.root_dir}/gridded/{group}/{year}.zarr"
                optimized_path = f"{self.root_dir}/optimized/{group}/{year}.zarr"
                
                # Check for existing data based on tier
                if os.path.exists(optimized_path):
                    logger.info("Skipping %s %s: Optimized tier already present.", group, year)
                    continue
                if not optimize and os.path.exists(local_path):
                    logger.info("Skipping %s %s: Raw tier already present.", group, year)
                    continue
                    
                target_write_path = optimized_path if optimize else local_path
                temp_target = f"{target_write_path}.tmp"
                
                logger.info("Streaming %s %s directly into %s format...", group, year, "Optimized" if optimize else "Raw")
                
                # Open remote lazy stream
                ds = self._open_zipped_zarr(f"data/{group}/{year}.zarr.zip", lazy=True)
                if ds is not None:
                    os.makedirs(os.path.dirname(target_write_path), exist_ok=True)
                    
                    if not optimize:
                        ds.to_zarr(local_path, consolidated=True, compute=True)
                        logger.info("Cached %s %s at %s", group, year, local_path)
                        continue

                    if os.path.exists(temp_target):
                        shutil.rmtree(temp_target)

                    # 1. Apply Daily Re-chunking lazily IF optimizing
                    chunks_dict = {
                        dim: (DEFAULT_TIME_CHUNKS if dim == 'time' else -1) 
                        for dim in ds.dims
                    }
                    ds = ds.chunk(chunks_dict)
                    
                    # 2. Get high-performance encoding settings
                    encoding = get_zarr_encoding(ds, DEFAULT_TIME_CHUNKS)

                    # 3. Trigger Parallel Single-Pass Write
                    with dask.config.set(scheduler='threads', num_workers=num_workers):
                        with ProgressBar():
                            ds.to_zarr(temp_target, mode='w', encoding=encoding, consolidated=True, compute=True, zarr_format=2)
                    
                    # 4. Atomic Swap
                    if os.path.exists(target_write_path):
                        shutil.rmtree(target_write_path)
                    os.rename(temp_target, target_write_path)
                    
                    logger.info("Ingested %s %s at %s", group, year, target_write_path)
                    
                    if optimize and should_swap and os.path.exists(local_path):
                        shutil.rmtree(local_path)

    def _load_registries(self) -> None:
        # Merge coordinates and parameter meta across all networks from Hugging Face
        st_dfs = []
        pr_dfs = []
        
        for src, ds in self._obs_zarr.items():
            df_s = pd.DataFrame({
                "latitude": ds.lat.values if 'lat' in ds.coords else np.nan,
                "longitude": ds.lon.values if 'lon' in ds.coords else np.nan,
                "network": src  
            }, index=ds.station.values)
            df_s.index.name = "station_id"
            st_dfs.append(df_s)
            
            df_p = pd.DataFrame({
                "unit": [ds[p].attrs.get('units', 'unknown') for p in ds.data_vars]
            }, index=list(ds.data_vars))
            pr_dfs.append(df_p)

        self.stations_table = pd.concat(st_dfs)
        self.stations_table = self.stations_table[~self.stations_table.index.duplicated(keep='first')]

        self.parameters_table = pd.concat(pr_dfs)
        self.parameters_table = self.parameters_table[~self.parameters_table.index.duplicated(keep='first')]

        # Save to local cache for future instantaneous loads
        os.makedirs(self.root_dir, exist_ok=True)
        self.stations_table.to_parquet(f"{self.root_dir}/stations_registry.parquet")
        self.parameters_table.to_parquet(f"{self.root_dir}/parameters_registry.parquet")

        self._filter_requested_parameters()

    def _load_local_registries(self) -> None:
        # Load metadata instantly from local disk catalogs
        st_candidates = [
            Path(self.root_dir) / "metadata" / "installation.parquet",
            Path(self.root_dir) / "installation.parquet",
            Path(self.root_dir) / "stations_registry.parquet",
        ]
        for st_p in st_candidates:
            if st_p.exists():
                self.stations_table = pd.read_parquet(st_p)
                break

        pr_candidates = [
            Path(self.root_dir) / "metadata" / "parameters.parquet",
            Path(self.root_dir) / "parameters.parquet",
            Path(self.root_dir) / "parameters_registry.parquet",
        ]
        for pr_p in pr_candidates:
            if pr_p.exists():
                self.parameters_table = pd.read_parquet(pr_p)
                break

        self._filter_requested_parameters()

    def _filter_requested_parameters(self) -> None:
        if self.requested_parameters is not None:
            params = [self.requested_parameters] if isinstance(self.requested_parameters, str) else list(self.requested_parameters)
            valid_params = []
            for p in params:
                canon = canonicalize_variable_name(p)
                for avail in self.parameters_table.index:
                    if avail == p or avail == canon or avail.startswith(p + "_") or avail.startswith(canon + "_"):
                        valid_params.append(avail)
                    elif p.lower() in CANONICAL_VARIABLE_METADATA.get(avail, {}).get("aliases", []):
                        valid_params.append(avail)
            valid_params = list(dict.fromkeys(valid_params))
            if not valid_params:
                raise ValueError(f"Requested parameters {params} not found in the dataset.")
            self.parameters_table = self.parameters_table.loc[valid_params]

    def _build_aggregation_dict(self, multi_index_columns) -> Dict[Tuple, str]:
        agg_dict = {}
        aggregation_methods = getattr(self, "aggregation_methods", {}) or {}
        for col in multi_index_columns:
            param = col[1].lower() if isinstance(col, tuple) else str(col).lower()
            if param in aggregation_methods:
                agg_dict[col] = aggregation_methods[param]
            elif any(kw in param for kw in ['cape', 'cin', 'convective_available', 'convective_inhibition']):
                agg_dict[col] = 'mean'
            elif any(kw in param for kw in ['precip', 'rain', 'tp', 'pr', 'total_precipitation']):
                agg_dict[col] = 'sum'
            elif any(kw in param for kw in ['solar', 'radiation', 'sr', 'rad', 'ssrd', 'surface_solar_radiation_downwards']):
                agg_dict[col] = 'sum'
            elif 'gust' in param:
                agg_dict[col] = 'max'
            else:
                agg_dict[col] = 'mean'
        return agg_dict

    def _prepare_observations(self) -> None:
        """
        Prepare observations by loading data from local Parquet files or Zarr files.
        Harmonizes weather variable nomenclature, preserves collocated sensor channels
        without destructive averaging, and respects physical conservation laws during resampling.
        """
        yearly_dfs = []
        for src in self.source_obs:
            norm_s = _normalize_source_name(src)
            src_dir = self._resolve_obs_source_dir(src)
            zds = self._obs_zarr.get(src) if src in self._obs_zarr else self._obs_zarr.get(norm_s)
            use_zarr = (self.prefer_zarr and zds is not None) or (zds is not None and not (src_dir.exists() and any(src_dir.glob("*.parquet"))))
            for year in self.years:
                p_path = src_dir / f"{year}.parquet"
                if use_zarr:
                    try:
                        yearly_ds = zds.sel(time=str(year)).load()
                        df_year = yearly_ds.to_dataframe().reset_index()
                        yearly_dfs.append(_tag_source(df_year, norm_s))
                    except (KeyError, ValueError):
                        try:
                            yearly_ds = zds.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).load()
                            if len(yearly_ds.time) > 0:
                                yearly_dfs.append(_tag_source(yearly_ds.to_dataframe().reset_index(), norm_s))
                        except Exception:
                            logger.warning("Year %s not found in Zarr store for %s. Skipping.", year, src)
                elif p_path.exists():
                    yearly_dfs.append(_tag_source(pd.read_parquet(p_path), norm_s))
                elif zds is not None:
                    try:
                        yearly_ds = zds.sel(time=str(year)).load()
                        df_year = yearly_ds.to_dataframe().reset_index()
                        src_dir.mkdir(parents=True, exist_ok=True)
                        df_year.to_parquet(p_path, compression="snappy", index=False)
                        yearly_dfs.append(_tag_source(df_year, norm_s))
                    except KeyError:
                        logger.warning("Year %s not found in %s. Skipping.", year, src)
                        continue
            if not yearly_dfs and zds is not None and use_zarr:
                try:
                    df_all = zds.to_dataframe().reset_index()
                    yearly_dfs.append(_tag_source(df_all, norm_s))
                except Exception as err:
                    logger.debug("Could not convert full Zarr to dataframe: %s", err)
            if not yearly_dfs and src_dir.exists():
                parquet_files = sorted(src_dir.glob("*.parquet"))
                for pf in parquet_files:
                    try:
                        df_p = pd.read_parquet(pf)
                        if "station_id" not in df_p.columns and "station" not in df_p.columns:
                            stem_code = pf.stem.split("_")[0]
                            df_p["station_id"] = stem_code
                        yearly_dfs.append(_tag_source(df_p, norm_s))
                    except Exception as err:
                        logger.warning("Could not read local parquet file %s: %s", pf, err)

        if not yearly_dfs:
            self.observations = ObservationDataFrame()
            self.mask = ObservationDataFrame()
            return

        full_df = pd.concat(yearly_dfs)
        full_df = _normalize_time_column(full_df)

        if "station" in full_df.columns:
            full_df = full_df.rename(columns={"station": "station_id"})
        full_df = self._extract_column_sources(full_df)

        param_cols = [c for c in full_df.columns if c not in ["time", "station_id"]]

        # Automatic variable harmonization to canonical nomenclature
        rename_map = {}
        for c in param_cols:
            canon = canonicalize_variable_name(c)
            if canon != c:
                rename_map[c] = canon
        if rename_map:
            full_df = full_df.rename(columns=rename_map)
            param_cols = [rename_map.get(c, c) for c in param_cols]

        obs_df = full_df.pivot(index="time", columns="station_id", values=param_cols)
        obs_df = obs_df.reorder_levels([1, 0], axis=1).sort_index(axis=1)
        obs_df.columns.names = ["station_id", "parameter"]

        # Filter parameters if requested
        if self.requested_parameters is not None:
            req_list = [self.requested_parameters] if isinstance(self.requested_parameters, str) else list(self.requested_parameters)
            valid_cols = []
            for col in obs_df.columns:
                p = col[1]
                for req in req_list:
                    req_canon = canonicalize_variable_name(req)
                    if p == req or p == req_canon or p.startswith(req + "_") or p.startswith(req_canon + "_"):
                        valid_cols.append(col)
                        break
                    elif req.lower() in CANONICAL_VARIABLE_METADATA.get(p, {}).get("aliases", []):
                        valid_cols.append(col)
                        break
            if valid_cols:
                obs_df = obs_df.loc[:, valid_cols]
        elif hasattr(self, "parameters_table") and not self.parameters_table.empty:
            valid_cols = obs_df.columns.intersection(
                pd.MultiIndex.from_product([obs_df.columns.levels[0], self.parameters])
            )
            if not valid_cols.empty:
                obs_df = obs_df.loc[:, valid_cols]

        if self.freq:
            agg_dict = self._build_aggregation_dict(obs_df.columns)
            resampled_dfs = []
            res_obj = obs_df.resample(self.freq)

            methods = {}
            for col, m in agg_dict.items():
                methods.setdefault(m, []).append(col)

            for method, cols in methods.items():
                if method == "sum":
                    resampled_dfs.append(res_obj[cols].sum(min_count=1))
                elif method == "max":
                    resampled_dfs.append(res_obj[cols].max())
                else:
                    resampled_dfs.append(res_obj[cols].mean())

            obs_df = pd.concat(resampled_dfs, axis=1).sort_index(axis=1)

        if self.pad_missing_values:
            obs_df = df_add_missing_columns(obs_df, self.stations, self.parameters)

        self.mask = ObservationDataFrame(~obs_df.isna())

        if self.imputation_method is None:
            pass
        elif callable(self.imputation_method):
            obs_df = self.imputation_method(obs_df)
        elif isinstance(self.imputation_method, str):
            method = self.imputation_method.lower()
            if method == "locf":
                limit = self.imputation_limit if self.imputation_limit is not None else 6
                obs_df = obs_df.ffill(limit=limit)
            elif method == "temporal_linear":
                limit = self.imputation_limit if self.imputation_limit is not None else 3
                obs_df = obs_df.interpolate(method="time", limit=limit)
            elif method == "diurnal_climatology":
                obs_df = obs_df.groupby([obs_df.index.month, obs_df.index.hour]).transform(
                    lambda x: x.fillna(x.median())
                )
            elif method == "zero":
                obs_df = obs_df.fillna(0)
            else:
                raise ValueError(f"Unknown imputation method: {method}")

        self.observations = ObservationDataFrame(obs_df)

    def load_topography(
        self,
        filename: str = "africa_static_priors",
        merge_stations: bool = False,
    ) -> Optional[Union[xr.Dataset, pd.DataFrame]]:
        """
        Load static topographic priors from disk or Hugging Face, optionally
        merging station metadata with topographic priors (DEM, Slope, Aspect, TPI_2000M).

        Applies cyclical encoding to Aspect (degrees -> sin/cos) to prevent
        the circular paradox where 359 deg and 1 deg appear maximally distant
        to a neural network, when physically they are adjacent.

        Mathematical docstring:
            Topographic grid shape: (Lat, Lon) for 2D fields.
            Station feature shape: (N_stations, C_topo_features) when merge_stations=True.

        Args:
            filename (str): Name of the topography file (without extension).
            merge_stations (bool): If True, merges interpolated static priors into
                self.stations_table / self.installation and returns updated station DataFrame.

        Returns:
            Optional[Union[xr.Dataset, pd.DataFrame]]: Topographic dataset or merged station DataFrame.
        """
        local_dir = f"{self.root_dir}/topography"
        local_path = f"{local_dir}/{filename}.nc"
        hf_path = f"hf://datasets/{self.repo_id}/data/topography/{filename}.nc"

        ds = None
        # Minimum-Latency Local Load
        if os.path.exists(local_path):
            try:
                ds = xr.open_dataset(local_path, engine="h5netcdf").load()
            except Exception as e:
                logger.warning("Local topography cache corrupted. Re-downloading... (%s)", e)

        # Backward compatibility check for legacy cached east_africa_static_priors.nc
        if ds is None:
            legacy_path = f"{local_dir}/east_africa_static_priors.nc"
            if os.path.exists(legacy_path):
                try:
                    ds = xr.open_dataset(legacy_path, engine="h5netcdf").load()
                except Exception as e:
                    logger.warning("Legacy topography cache corrupted: %s", e)

        # First-Time Fetch (Download to Cache)
        if ds is None:
            if not getattr(self, "use_hf", False) or self.fs is None:
                if merge_stations and hasattr(self, "stations_table"):
                    return self.stations_table
                return None
            try:
                os.makedirs(local_dir, exist_ok=True)
                self.fs.get(hf_path, local_path)
                ds = xr.open_dataset(local_path, engine="h5netcdf").load()
            except Exception as e:
                logger.warning("Could not fetch topography data from Hugging Face: %s", e)
                if merge_stations and hasattr(self, "stations_table"):
                    return self.stations_table
                return None

        # Cyclical Encoding: ALL Aspect variables (degrees) -> sin/cos pairs
        aspect_vars = [
            v for v in list(ds.data_vars)
            if 'aspect' in v.lower() and '_sin' not in v and '_cos' not in v
        ]
        for var_name in aspect_vars:
            aspect_rad = ds[var_name] * (np.pi / 180.0)
            ds[f'{var_name}_sin'] = np.sin(aspect_rad)
            ds[f'{var_name}_cos'] = np.cos(aspect_rad)
            ds = ds.drop_vars(var_name)

        if merge_stations and hasattr(self, "stations_table") and not self.stations_table.empty:
            self._topo_ds = ds
            self.interpolate_topography()
            return self.stations_table

        return ds

    def interpolate_topography(self) -> None:
        """
        Interpolate topography data to the station locations.
        """
        if self._topo_ds is None:
            # 1. Load the default static netCDF topography (Elevation, Slope, Aspect)
            self._topo_ds = self.load_topography()

        if self._topo_ds is None:
            return

        valid_stations = self.stations_table.dropna(subset=['latitude', 'longitude'])
        if valid_stations.empty:
            return

        lats = xr.DataArray(valid_stations['latitude'].values, dims="station", 
                            coords={"station": ("station", valid_stations.index.values)})
        lons = xr.DataArray(valid_stations['longitude'].values, dims="station", 
                            coords={"station": ("station", valid_stations.index.values)})

        # Nearest Neighbor for Categorical Variables
        # (Aspect has already been decomposed into continuous sin/cos in load_topography)
        nearest_vars = ['LandCover']
        valid_nearest = [v for v in nearest_vars if v in self._topo_ds.data_vars]
        if valid_nearest:
            sampled_nearest = self._topo_ds[valid_nearest].sel(lat=lats, lon=lons, method="nearest")
            df_nearest = sampled_nearest.to_dataframe().drop(columns=['lat', 'lon'], errors='ignore')
        else:
            df_nearest = pd.DataFrame(index=valid_stations.index)

        # Bilinear Interpolation for Continuous Physical Fields
        continuous_vars = [v for v in self._topo_ds.data_vars if v not in nearest_vars]
        if continuous_vars:
            sampled_bilinear = self._topo_ds[continuous_vars].interp(lat=lats, lon=lons, method="linear")
            df_bilinear = sampled_bilinear.to_dataframe().drop(columns=['lat', 'lon'], errors='ignore')
        else:
            df_bilinear = pd.DataFrame(index=valid_stations.index)

        # Combine and Join
        df_topo = pd.concat([df_nearest, df_bilinear], axis=1)
        
        # Physically sound defaults for ocean/coastal-masked terrain.
        # Pattern-based to handle all multi-scale variables dynamically:
        #   *_cos -> 1 (undefined direction defaults to unit vector on cos axis)
        #   LandCover -> 0 (water/ocean class)
        #   Everything else -> 0 (sea level, flat, no gradient)
        for col in df_topo.columns:
            if col == 'LandCover':
                default_val = 0
            elif col.endswith('_cos'):
                default_val = 1
            else:
                default_val = 0
            df_topo[col] = df_topo[col].fillna(default_val)

        self.stations_table = self.stations_table.join(df_topo)


    def validate_stations(self, stations: Optional[Union[str, List[str]]] = None) -> List[str]:
        """
        Validate that requested stations are available in the dataset inventory.
        Issues a soft advisory warning if missing stations are requested, continuing
        with available stations rather than terminating execution.

        Mathematical docstring:
            Input stations: Sequence of length N_requested.
            Output stations: List of length N_valid <= N_requested.

        Args:
            stations: Sequence of station codes, single station code, "all", or None.

        Returns:
            List[str]: Valid station identifiers present in the dataset.
        """
        if stations is None or stations == "all":
            if hasattr(self, "stations_table") and self.stations_table is not None:
                return list(self.stations_table.index)
            return []

        if isinstance(stations, str):
            stations = [stations]

        available = set()
        if hasattr(self, "stations_table") and self.stations_table is not None:
            available.update(self.stations_table.index)
        if hasattr(self, "observations") and self.observations is not None and not self.observations.empty:
            if isinstance(self.observations.columns, pd.MultiIndex):
                available.update(self.observations.columns.get_level_values(0).unique())

        valid = [s for s in stations if s in available]

        if len(valid) < len(stations):
            missing = sorted(list(set(stations) - set(valid)))
            logger.warning("Requested station(s) not in dataset or unpermitted: %s. Continuing with available stations.", missing)
            warnings.warn(f"Requested station(s) not in dataset: {missing}. Continuing with available stations.", UserWarning, stacklevel=2)

        return valid

    def get_stations_metadata(self, stations: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Get metadata for the requested stations.
        If stations is None, returns metadata for all available stations.
        """
        valid_stations = self.validate_stations(stations)
        return self.stations_table.loc[valid_stations]

    def get_spatial_split(
        self,
        test_fraction: float = 0.2,
        n_clusters: int = 5,
        seed: int = 42,
        stations: Optional[Union[str, List[str]]] = None,
        n_folds: Optional[int] = None,
    ) -> Union[Dict[str, List[str]], List[Dict[str, List[str]]]]:
        """
        Generate geographically stratified train/test station splits.

        Uses KMeans clustering on (latitude, longitude) to partition stations into
        spatial clusters, then samples the test set proportionally from each cluster.
        This ensures held-out test stations are spatially representative across the
        entire African domain, not clustered in one corner.

        Args:
            test_fraction (float): Fraction of stations to hold out (0.0 to 1.0).
            n_clusters (int): Number of geographic clusters to stratify across.
                Higher values give finer spatial balance. Should not exceed the
                number of stations.
            seed (int): Random seed for reproducibility.
            stations (Optional[Union[str, List[str]]]): Subset of stations to split.
                If None, uses all available stations with valid coordinates.
            n_folds (Optional[int]): If set, returns a list of n_folds non-overlapping
                spatial cross-validation folds instead of a single split.

        Returns:
            If n_folds is None:
                Dict with keys "train" and "test", each mapping to a list of station IDs.
            If n_folds is set:
                List of n_folds dicts, each with "train" and "test" station ID lists.
                Every station appears in the test set of exactly one fold.

        Example:
            split = ds.get_spatial_split(test_fraction=0.2, n_clusters=5)
            train_windows = ds.get_windows(stations=split["train"], ...)
            test_windows  = ds.get_windows(stations=split["test"], ...)
        """
        from sklearn.cluster import KMeans

        valid = self.validate_stations(stations)
        coords = self.stations_table.loc[valid].dropna(subset=["latitude", "longitude"])
        if coords.empty:
            raise ValueError("No stations with valid latitude/longitude found.")

        n_stations = len(coords)
        actual_clusters = min(n_clusters, n_stations)

        # Cluster stations geographically
        X = coords[["latitude", "longitude"]].values
        km = KMeans(n_clusters=actual_clusters, random_state=seed, n_init=10)
        labels = km.fit_predict(X)

        rng = np.random.default_rng(seed)

        if n_folds is not None:
            # Spatial K-Fold Cross-Validation
            # Assign each station a fold index, stratified by cluster
            fold_assignments = np.full(n_stations, -1, dtype=int)
            for cluster_id in range(actual_clusters):
                cluster_mask = labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                rng.shuffle(cluster_indices)
                # Round-robin assign to folds
                for i, idx in enumerate(cluster_indices):
                    fold_assignments[idx] = i % n_folds

            station_ids = coords.index.values
            folds = []
            for fold_idx in range(n_folds):
                test_mask = fold_assignments == fold_idx
                folds.append({
                    "train": station_ids[~test_mask].tolist(),
                    "test": station_ids[test_mask].tolist(),
                })
            return folds

        # Single train/test split, stratified by cluster
        train_ids, test_ids = [], []
        station_ids = coords.index.values

        for cluster_id in range(actual_clusters):
            cluster_mask = labels == cluster_id
            cluster_stations = np.array(station_ids[cluster_mask], copy=True)
            rng.shuffle(cluster_stations)

            n_test = max(1, int(len(cluster_stations) * test_fraction))
            test_ids.extend(cluster_stations[:n_test].tolist())
            train_ids.extend(cluster_stations[n_test:].tolist())

        return {"train": train_ids, "test": test_ids}


    def _apply_lapse_rate_correction_xr(
        self, 
        temp_k: xr.DataArray, 
        grid_elev: xr.DataArray, 
        station_elev: xr.DataArray,
        lapse_rate_k_per_m: xr.DataArray
    ) -> xr.DataArray:
        """
        Applies a dynamic empirical lapse rate to correct temperature using Xarray native broadcasting.
        """
        elev_diff = station_elev - grid_elev
        return temp_k - (lapse_rate_k_per_m * elev_diff)

    def _apply_barometric_correction_xr(
        self, 
        pressure_pa: xr.DataArray, 
        temp_k: xr.DataArray, 
        grid_elev: xr.DataArray, 
        station_elev: xr.DataArray,
        lapse_rate_k_per_m: xr.DataArray
    ) -> xr.DataArray:
        """
        Applies the rigorous hydrostatic barometric formula utilizing a dynamic lapse rate,
        with a mathematical fallback for isothermal atmospheric layers.
        """
        g = 9.80665 # Gravitational acceleration
        R = 287.05 # Specific gas constant for dry air (J/(kg*K))
        # where Gamma is the lapse rate (K/m)
        elev_diff = station_elev - grid_elev
        
        # Calculate dynamic exponent: g / (R * Gamma)
        dynamic_exponent = xr.where(lapse_rate_k_per_m != 0, g / (R * lapse_rate_k_per_m), 1.0)
        
        # Standard Hydrostatic Equation (Gamma != 0)
        standard_correction = (temp_k / (temp_k + (lapse_rate_k_per_m * elev_diff))) ** dynamic_exponent
        
        # Isothermal Equation (Gamma == 0)
        isothermal_correction = np.exp((-g * elev_diff) / (R * temp_k))
        
        # Apply the correct multiplier
        corr_factor = xr.where(lapse_rate_k_per_m != 0, standard_correction, isothermal_correction)
        
        return pressure_pa * corr_factor

    def _get_optimal_extraction_method(
        self, 
        variable_name: str, 
        overrides: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Dynamically routes the variable to its extraction method, with an option to override.
        """
        # 1. Check for manual override from the user
        if overrides and variable_name in overrides:
            return overrides[variable_name]
            
        var_lower = variable_name.lower()
        
        # 2. Mass Conservation (Precipitation)
        if any(kw in var_lower for kw in ['precip', 'rain', 'tp', 'pr']):
            return "patch_flat"
            
        # 3. Extreme / Turbulent Variables (Wind Gusts)
        elif any(kw in var_lower for kw in ['gust', 'fg10']):
            return "patch_max"
            
        # 4. Thermodynamic State (Temperature) -> Requires Lapse Rate Correction
        elif any(kw in var_lower for kw in ['temp', 't2m']):
            return "thermodynamic"
            
        # 5. Hydrostatic State (Pressure) -> Requires Barometric Correction
        elif any(kw in var_lower for kw in ['pressure', 'sp', 'msl']):
            return "hydrostatic"
            
        # 6. Vectorial Variables & 3D Profiles -> Standard Bilinear
        elif any(kw in var_lower for kw in ['u10', 'v10', 'u_comp', 'v_comp', 'humidity', 'rh', 'geopotential']):
            return "linear"
            
        return "nearest"

    @staticmethod
    def infer_variable_types(
        variables: Sequence[str]
    ) -> Dict[str, Literal["PHYSICAL", "PRECIPITATION", "STATIC", "FLUX"]]:
        """
        Classifies variables into physical (Gaussian), precipitation (Tweedie), 
        static (Topography), or flux (Radiation/Accumulation) based on standard meteorological nomenclature.
        
        Args:
            variables: List of variable names to classify.
            
        Returns:
            Dictionary mapping variable name to its scientific type.
        """
        types = {}
        for var in variables:
            v_lower = var.lower()
            # 1. Thermodynamic Stability (Intensive Physical state, not cumulative flux)
            if any(kw in v_lower for kw in ['cape', 'cin', 'convective_available_potential_energy', 'convective_inhibition']):
                types[var] = "PHYSICAL"
            # 2. Precipitation (Non-Gaussian, zero-inflated mass flux)
            elif any(kw in v_lower for kw in ['precip', 'rain', 'tp', 'pr', 'convective_precip', 'convective_rain', 'convective_snow', 'snow']):
                types[var] = "PRECIPITATION"
            # 3. Fluxes and Accumulated Variables
            elif any(kw in v_lower for kw in ['radiation', 'rad', 'ssrd', 'flux']):
                types[var] = "FLUX"
            # 4. Static / Categorical variables
            elif any(kw in v_lower for kw in ['dem', 'slope', 'aspect', 'mask', 'soil', 'veg', 'topo']):
                types[var] = "STATIC"
            # 5. Physical / Thermodynamic variables (Gaussian-like)
            else:
                types[var] = "PHYSICAL"
        return types

    def _extract_continuous_fields_xr(self, ds, coords, continuous_methods, method_groups, grid_lats, grid_lons, lat_dim, lon_dim, has_dem):
        c_vars = []
        for m in continuous_methods:
            c_vars.extend(method_groups[m])
        
        # Define spatial bounding box
        lat_idx_1 = np.abs(grid_lats - coords['latitude'].min()).argmin()
        lat_idx_2 = np.abs(grid_lats - coords['latitude'].max()).argmin()
        lon_idx_1 = np.abs(grid_lons - coords['longitude'].min()).argmin()
        lon_idx_2 = np.abs(grid_lons - coords['longitude'].max()).argmin()
        
        l_min, l_max = min(lat_idx_1, lat_idx_2), max(lat_idx_1, lat_idx_2)
        ln_min, ln_max = min(lon_idx_1, lon_idx_2), max(lon_idx_1, lon_idx_2)
        
        lat_slice = slice(max(0, l_min - 2), min(len(grid_lats), l_max + 3))
        lon_slice = slice(max(0, ln_min - 2), min(len(grid_lons), ln_max + 3))
        
        # Auxiliary Variable Management: Fetch 3D levels if doing physical corrections
        fetch_vars = c_vars.copy()
        needs_physics = has_dem and ("thermodynamic" in method_groups or "hydrostatic" in method_groups)
        
        sfc_z_var = 'geopotential_at_surface' if 'geopotential_at_surface' in ds.data_vars else ('z' if 'z' in ds.data_vars else None)
        if needs_physics and sfc_z_var and sfc_z_var not in fetch_vars:
            fetch_vars.append(sfc_z_var)
            
        if needs_physics and 'temperature' in ds.data_vars and 'geopotential' in ds.data_vars and 'level' in ds.dims:
            if 'temperature' not in fetch_vars: fetch_vars.append('temperature')
            if 'geopotential' not in fetch_vars: fetch_vars.append('geopotential')

        # Load bounding box definitively into RAM (Fixes catastrophic Dask Scheduler Hangs over HTTP chunks)
        local_ds = ds[fetch_vars].isel({lat_dim: lat_slice, lon_dim: lon_slice}).load()
        
        # Xarray spatial interpolation (runs purely in-memory on Numpy arrays)
        lats = xr.DataArray(coords.latitude.values, dims="station", coords={"station": coords.index.values})
        lons = xr.DataArray(coords.longitude.values, dims="station", coords={"station": coords.index.values})
        
        # Interpolate identically
        sampled_continuous = local_ds.interp({lat_dim: lats, lon_dim: lons}, method="linear")
        
        # PHYSICAL CORRECTIONS 
        if needs_physics and sfc_z_var:
            g = 9.80665
            grid_elev = sampled_continuous[sfc_z_var] / g
            station_elev = xr.DataArray(coords['dem'].values, dims=["station"], coords={"station": coords.index.values})
            
            # Calculate Dynamic Empirical Lapse Rate (EELR) across time and stations
            try:
                t_850 = sampled_continuous['temperature'].sel(level=850)
                t_700 = sampled_continuous['temperature'].sel(level=700)
                z_850 = sampled_continuous['geopotential'].sel(level=850) / g
                z_700 = sampled_continuous['geopotential'].sel(level=700) / g
                eelr = (t_850 - t_700) / (z_700 - z_850)
                # Force positive (handle rare inversions) and clamp to physical bounds:
                # Moist adiabatic ~0.004 K/m to dry adiabatic ~0.0098 K/m
                eelr = xr.where(eelr < 0, np.abs(eelr), eelr)
                eelr = eelr.clip(min=0.004, max=0.0098).fillna(0.0065)
            except KeyError:
                # Fallback to standard 6.5 C/km if 850/700hPa levels aren't in the dataset
                eelr = xr.full_like(grid_elev, 0.0065)
            
            # Apply Thermodynamic Corrections
            if "thermodynamic" in method_groups:
                for temp_var in method_groups["thermodynamic"]:
                    sampled_continuous[temp_var] = self._apply_lapse_rate_correction_xr(
                        sampled_continuous[temp_var], grid_elev, station_elev, eelr
                    )
                    
            # Apply Hydrostatic Corrections
            if "hydrostatic" in method_groups:
                for pres_var in method_groups["hydrostatic"]:
                    temp_col = next((v for v in sampled_continuous.data_vars if 't2m' in v.lower() or '2m_temperature' in v.lower()), None)
                    if temp_col:
                        sampled_continuous[pres_var] = self._apply_barometric_correction_xr(
                            sampled_continuous[pres_var], sampled_continuous[temp_col], grid_elev, station_elev, eelr
                        )
                        
        # Cleanup auxiliary variables not explicitly requested by user
        vars_to_drop = [v for v in sampled_continuous.data_vars if v not in c_vars]
        sampled_continuous = sampled_continuous.drop_vars(vars_to_drop)
        return sampled_continuous

    def _extract_nearest_fields_xr(self, ds, coords, n_vars, grid_lats, grid_lons, lat_dim, lon_dim):
        lat_idx = [np.abs(grid_lats - lat).argmin() for lat in coords['latitude']]
        lon_idx = [np.abs(grid_lons - lon).argmin() for lon in coords['longitude']]
        
        lat_idx_da = xr.DataArray(lat_idx, dims="station")
        lon_idx_da = xr.DataArray(lon_idx, dims="station")
        
        sampled_nearest = ds[n_vars].isel({lat_dim: lat_idx_da, lon_dim: lon_idx_da})
        sampled_nearest = sampled_nearest.assign_coords(station=("station", coords.index.values)).load()
        return sampled_nearest

    def _extract_patch_fields_xr(self, ds, coords, patch_methods, method_groups, grid_lats, grid_lons, lat_dim, lon_dim, patch_radius):
        p_vars = []
        for m in patch_methods:
            p_vars.extend(method_groups[m])
            
        ds_vars = ds[p_vars]
        
        # 1. Calculate optimal center nodes in O(1)
        lat_diff = np.abs(grid_lats[:, None] - coords['latitude'].values)
        center_lat_idx = lat_diff.argmin(axis=0)
        
        lon_diff = np.abs(grid_lons[:, None] - coords['longitude'].values)
        center_lon_idx = lon_diff.argmin(axis=0)
        
        # 2. Build 3x3 Vectorized Patch Meshes natively across all stations
        patch_mesh = np.arange(-patch_radius, patch_radius + 1)
        
        # Mathematically clamp edges 
        lat_idx = np.clip(center_lat_idx[:, None] + patch_mesh, 0, len(grid_lats) - 1)
        lon_idx = np.clip(center_lon_idx[:, None] + patch_mesh, 0, len(grid_lons) - 1)
        
        # 3. Construct the Advanced Pointwise Xarray Target Coordinates
        stations_coord = coords.index.values
        lat_da = xr.DataArray(lat_idx, dims=["station", "patch_y"], coords={"station": stations_coord})
        lon_da = xr.DataArray(lon_idx, dims=["station", "patch_x"], coords={"station": stations_coord})
        
        # 4. Fire the massive concurrent HTTP sparse Zarr fetch entirely in one C-call (Skipping empty space)
        all_patches = ds_vars.isel({lat_dim: lat_da, lon_dim: lon_da}).load()
        
        # 5. Process Unrollings natively in RAM O(1)
        processed_patches = []
        if "patch_max" in patch_methods:
            reduced_max = all_patches[method_groups["patch_max"]].max(dim=["patch_y", "patch_x"], skipna=True)
            processed_patches.append(reduced_max)
            
        if "patch_flat" in patch_methods:
            flat_ds = all_patches[method_groups["patch_flat"]]
            stacked = flat_ds.stack(spatial=["patch_y", "patch_x"])
            
            flat_vars = {}
            for var in stacked.data_vars:
                var_da = stacked[var]
                for i in range(len(stacked.spatial)):
                    new_name = f"{var}_px{i}"
                    flat_vars[new_name] = var_da.isel(spatial=i).drop_vars(
                        ["spatial", lat_dim, lon_dim, "patch_y", "patch_x"], errors="ignore"
                    )
            
            reduced_flat = xr.Dataset(flat_vars)
            processed_patches.append(reduced_flat)
            
        sampled_patch = xr.merge(processed_patches)
        return sampled_patch

    def get_gridded_for_stations(
        self, 
        groups: List[str], 
        stations: Optional[Union[str, List[str]]] = None, 
        years: Optional[Union[int, List[int]]] = None,
        variables: Optional[List[str]] = None,
        method_overrides: Optional[Dict[str, str]] = None,
        patch_radius: int = 1,
        interpolation: Optional[str] = None,
        first_date: Optional[Union[str, pd.Timestamp]] = None,
        last_date: Optional[Union[str, pd.Timestamp]] = None,
        pressure_levels: Optional[List[int]] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:

        """
        Extract gridded weather data for a list of stations.

        Args:
            groups (List[str]): List of gridded products to extract.
            stations (Optional[Union[str, List[str]]]): List of station IDs to extract data for.
            years (Optional[Union[int, List[int]]]): List of years to extract data for.
            variables (Optional[List[str]]): List of variables to extract.
            method_overrides (Optional[Dict[str, str]]): Dictionary of method overrides
            for specific variables (e.g. {"temperature": "patch_max", "wind_speed": "nearest", "total_precipitation": "linear"}).
            patch_radius (int): Radius of the patch to extract, 1 = 3x3 grid, 2 = 5x5 grid, etc.
            interpolation (Optional[str]): Overall interpolation method for variables ("nearest", "bilinear", "peak_preserving", "patch").
            first_date (Optional[Union[str, pd.Timestamp]]): First date to extract data for.
            last_date (Optional[Union[str, pd.Timestamp]]): Last date to extract data for.

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]: 
                - First Dict: Contains native temporal resolution DataFrames for each group.
                - Second Dict: Contains matched Boolean mask DataFrames indicating structurally missing coordinates (1 for missing).
        """
        
        self._validate_groups(groups)
        valid_stations = self.validate_stations(stations)
        if not valid_stations:
            return GriddedResult({}, {})
            
        if interpolation:
            if method_overrides is None:
                method_overrides = {}
            interp_map = {
                "patch": "patch_flat",
                "peak_preserving": "patch_max",
                "bilinear": "linear",
                "nearest": "nearest"
            }
            mapped_method = interp_map.get(interpolation, interpolation)
            if variables:
                for var in variables:
                    if var not in method_overrides:
                        method_overrides[var] = mapped_method

        target_years = [years] if isinstance(years, int) else (years or self.years)
        target_years = self._validate_years(target_years)
        
        elev_col = next((c for c in ['DEM', 'dem', 'elevation', 'Elevation', 'z'] if c in self.stations_table.columns), None)
        has_dem = elev_col is not None
        if not has_dem:
            print("Warning: Station DEM/Elevation not found. Topographic corrections will be bypassed.")
            coords = self.stations_table.loc[valid_stations].dropna(subset=['latitude', 'longitude'])
        else:
            coords = self.stations_table.loc[valid_stations].dropna(subset=['latitude', 'longitude', elev_col])
            # Ensure it's consistently named 'dem' internally for downstream
            coords = coords.rename(columns={elev_col: 'dem'})

        extracted_dfs = {}
        mask_dfs = {}
        for group in groups:
            yearly_dfs = []
            
            # Fast-fail: validate that requested variables exist in this group
            # before entering the expensive per-year extraction loop.
            if variables:
                peek_ds = None
                for peek_year in target_years:
                    peek_ds = self._open_gridded_zarr(group, peek_year, lazy=True)
                    if peek_ds is not None:
                        break
                if peek_ds is not None:
                    available_vars = list(peek_ds.data_vars)
                    missing_vars = [v for v in variables if v not in available_vars]
                    if len(missing_vars) == len(variables):
                        raise ValueError(
                            f"None of the requested variables {variables} exist in "
                            f"'{group}'. Available variables: {available_vars}"
                        )
                    if missing_vars:
                        print(
                            f"Warning: Variables {missing_vars} not found in '{group}'. "
                            f"Proceeding with: {[v for v in variables if v in available_vars]}"
                        )
            
            for year in tqdm(target_years, desc=f"Extracting {group}"):
                ds = self._open_gridded_zarr(group, year, lazy=True)
                if ds is None:
                    continue
                
                # Validate constraints dynamically
                validate_gridded_dataset(
                    ds, 
                    product_name=group, 
                    requested_freq=self.freq,
                    bbox=None
                )
                
                if pressure_levels is not None and 'level' in ds.coords:
                    v_levels = [l for l in pressure_levels if l in ds.level.values]
                    if v_levels:
                        ds = ds.sel(level=v_levels)
                
                # Apply explicit temporal constraints to violently slash HTTP payload downloads
                if first_date or last_date:
                    f_dt = pd.to_datetime(first_date, utc=True).tz_localize(None) if first_date else None
                    l_dt = pd.to_datetime(last_date, utc=True).tz_localize(None) if last_date else None
                    ds = ds.sel(time=slice(f_dt, l_dt))
                    
                if len(ds.time) == 0:
                    continue
                
                requested_vars = variables if variables else list(ds.data_vars)
                valid_vars = [v for v in requested_vars if v in ds.data_vars]
                if not valid_vars:
                    continue
                
                # Group variables by their scientifically correct extraction method
                method_groups = {}
                for var in valid_vars:
                    method = self._get_optimal_extraction_method(var, method_overrides)
                    method_groups.setdefault(method, []).append(var)

                lat_dim = 'lat' if 'lat' in ds.coords else 'latitude'
                lon_dim = 'lon' if 'lon' in ds.coords else 'longitude'
                grid_lats = ds[lat_dim].values
                grid_lons = ds[lon_dim].values

                group_results = []

                # ---  CONTINUOUS FIELDS (Linear, Thermodynamic, Hydrostatic) ---
                continuous_methods = [m for m in ["linear", "thermodynamic", "hydrostatic"] if m in method_groups]
                if continuous_methods:
                    sampled_continuous = self._extract_continuous_fields_xr(
                        ds, coords, continuous_methods, method_groups, grid_lats, grid_lons, lat_dim, lon_dim, has_dem
                    )
                    group_results.append(sampled_continuous)

                # --- NEAREST FIELDS (LandCover, Categorical) ---
                if "nearest" in method_groups:
                    sampled_nearest = self._extract_nearest_fields_xr(
                        ds, coords, method_groups["nearest"], grid_lats, grid_lons, lat_dim, lon_dim
                    )
                    group_results.append(sampled_nearest)

                # --- PATCH FIELDS (Precipitation / Wind Gusts) ---
                patch_methods = [m for m in ["patch_max", "patch_flat"] if m in method_groups]
                if patch_methods:
                    sampled_patch = self._extract_patch_fields_xr(
                        ds, coords, patch_methods, method_groups, grid_lats, grid_lons, lat_dim, lon_dim, patch_radius
                    )
                    group_results.append(sampled_patch)

                # --- Merge and Flatten to DataFrame ---
                # Drop nuisance coordinates PRE-MERGE to avoid strict Xarray spatial dimension conflicts
                nuisance_coords = [lat_dim, lon_dim, 'latitude', 'longitude', 'step',
                                   'number', 'surface', 'valid_time', 'height',
                                   'spatial', 'band', 'spatial_ref', 'patch_y', 'patch_x']
                
                clean_results = []
                for res_ds in group_results:
                    drop_c = [c for c in nuisance_coords if c in res_ds.coords or c in res_ds.data_vars]
                    clean_results.append(res_ds.drop_vars(drop_c, errors='ignore'))
                    
                merged_yearly_ds = xr.merge(clean_results, compat="override")
                df = merged_yearly_ds.to_dataframe().reset_index()
                
                # Cleanup residual columns explicitly in pandas
                cols_to_drop = [c for c in nuisance_coords if c in df.columns]
                df = df.drop(columns=cols_to_drop, errors='ignore')
                
                # If 'level' exists, we merge it into the parameter string (e.g., 'temperature_850')
                if 'level' in df.columns:
                    # Dynamically capture variables that were renamed during patch_flat (e.g. _px0, _px1)
                    actual_val_vars = []
                    level_vars = []
                    for col in df.columns:
                        if col in valid_vars:
                            actual_val_vars.append(col)
                            if 'level' in merged_yearly_ds[col].dims:
                                level_vars.append(col)
                        else:
                            for base_var in valid_vars:
                                if col.startswith(f"{base_var}_px"):
                                    actual_val_vars.append(col)
                                    if 'level' in merged_yearly_ds[col].dims:
                                        level_vars.append(col)
                                    break
                                    
                    # Melt and reconstruct to combine variable name with level
                    id_vars = [c for c in df.columns if c not in actual_val_vars and c != 'level']
                    df_melt = df.melt(id_vars=id_vars + ['level'], value_vars=actual_val_vars, var_name='base_param')
                    
                    # Physically separate 3D variables from erroneously broadcasted 2D surface variables
                    is_level_var = df_melt['base_param'].isin(level_vars)
                    df_3d = df_melt[is_level_var].copy()
                    df_2d = df_melt[~is_level_var].drop_duplicates(subset=['time', 'station', 'base_param']).copy()
                    df_melt = pd.concat([df_3d, df_2d])
                    
                    df_melt['parameter_level'] = df_melt.apply(
                        lambda row: f"_{int(row['level'])}" if pd.notnull(row['level']) and row['base_param'] in level_vars else "", axis=1
                    )
                    df_melt['parameter'] = df_melt['base_param'] + df_melt['parameter_level']
                    df = df_melt.pivot_table(index=['time', 'station'], columns='parameter', values='value').reset_index()

                yearly_dfs.append(df)
                
            if not yearly_dfs:
                continue
                
            group_df = pd.concat(yearly_dfs, ignore_index=True)
            pivot_col = 'station' if 'station' in group_df.columns else 'station_id'
            param_cols = [c for c in group_df.columns if c not in ['time', pivot_col]]
            
            pivoted_df = group_df.pivot(index='time', columns=pivot_col, values=param_cols)
            pivoted_df = pivoted_df.reorder_levels([1, 0], axis=1).sort_index(axis=1)
            pivoted_df.columns.names = ['station_id', 'parameter']
            
            # Extract missing mask before any imputation
            mask_df = pivoted_df.isna().astype(int)
            
            extracted_dfs[group] = pivoted_df
            mask_dfs[group] = mask_df
            
        return GriddedResult(extracted_dfs, mask_dfs)

    @staticmethod
    def spatial_sel(ds: xr.Dataset, lat_slice: Optional[slice] = None, lon_slice: Optional[slice] = None) -> xr.Dataset:
        """
        Order-agnostic spatial selection utility. Auto-corrects flipped slices 
        (e.g., max to min) and emits a gentle warning, proceeding without failure.
        """
       
        sel_dict = {}
        
        lat_dim = 'lat' if 'lat' in ds.coords else ('latitude' if 'latitude' in ds.coords else None)
        lon_dim = 'lon' if 'lon' in ds.coords else ('longitude' if 'longitude' in ds.coords else None)
        
        if lat_slice and lat_dim:
            if lat_slice.start is not None and lat_slice.stop is not None and lat_slice.start > lat_slice.stop:
                warnings.warn(f"Flipped latitude slice detected ({lat_slice.start} to {lat_slice.stop}). Auto-correcting to (min, max).", UserWarning)
                lat_slice = slice(lat_slice.stop, lat_slice.start)
            sel_dict[lat_dim] = lat_slice
            
        if lon_slice and lon_dim:
            if lon_slice.start is not None and lon_slice.stop is not None and lon_slice.start > lon_slice.stop:
                warnings.warn(f"Flipped longitude slice detected ({lon_slice.start} to {lon_slice.stop}). Auto-correcting to (min, max).", UserWarning)
                lon_slice = slice(lon_slice.stop, lon_slice.start)
            sel_dict[lon_dim] = lon_slice
            
        return ds.sel(sel_dict) if sel_dict else ds

    def _export_df(self, df: pd.DataFrame, as_numpy: bool = False, copy: bool = False, fill_value=None) -> FrameArray:
        if fill_value is not None:
            df = df.fillna(fill_value)
            
        if as_numpy:
            # Enforce strict lexicographic ordering before numpy flatten
            # to prevent station/parameter axis scrambling
            df = df.sort_index(axis=1)
            stations = df.columns.unique(0)
            parameters = df.columns.unique(1)
            if len(stations) > 0 and len(parameters) > 0:
                res = df.values.reshape(-1, len(stations), len(parameters))
            else:
                res = np.empty((len(df), 0, 0))
            return res
        return df.copy() if copy else df

    def _extract_column_sources(self, full_df: pd.DataFrame) -> pd.DataFrame:
        """
        Record the observation source of every (station, parameter) series.

        Args:
            full_df: Long-format frame with columns time, station_id, parameters
                and the '__source__' tag column.

        Returns:
            Frame without the '__source__' column. The mapping
            {(station_id, parameter): source} is stored in self._column_sources.
        """
        if "__source__" not in full_df.columns:
            return full_df
        meta = {"time", "station_id", "__source__"}
        values = [c for c in full_df.columns if c not in meta]
        sources: Dict[Tuple[str, str], str] = {}
        if values and "station_id" in full_df.columns:
            canon = {c: canonicalize_variable_name(c) for c in values}
            present = full_df.groupby(["station_id", "__source__"])[values].count() > 0
            for (station, src), row in present.iterrows():
                for col in row.index[row.to_numpy()]:
                    sources.setdefault((station, canon[col]), src)
        self._column_sources = sources
        return full_df.drop(columns=["__source__"])

    def _source_columns(self, source: str) -> pd.MultiIndex:
        """
        Select the (station, parameter) columns of self.observations that came from one source.

        Args:
            source: Observation source name ('tahmo', 'ghcn', 'cds_insitu').

        Returns:
            MultiIndex of (station_id, parameter) pairs, length N_cols_source.
        """
        columns = self.observations.columns
        mapping = getattr(self, "_column_sources", None) or {}
        norm_source = _normalize_source_name(source)
        if not mapping:
            if len(self.source_obs) == 1 and _normalize_source_name(self.source_obs[0]) == norm_source:
                return columns
            return columns[:0]
        keep = [i for i, col in enumerate(columns) if _normalize_source_name(mapping.get(tuple(col), "")) == norm_source]
        return columns[keep]

    @staticmethod
    def _empty_observation_frame() -> "ObservationDataFrame":
        """
        Return an empty frame with a naive 'time' index and (station_id, parameter) columns.
        """
        index = pd.DatetimeIndex([], name="time")
        columns = pd.MultiIndex.from_arrays([[], []], names=["station_id", "parameter"])
        return ObservationDataFrame(index=index, columns=columns)

    @staticmethod
    def _standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce a tz-naive DatetimeIndex named 'time' and columns named (station_id, parameter).

        Args:
            df: Frame of shape (T, N_cols) with two-level columns.

        Returns:
            Frame of shape (T, N_cols) with standardized index and column names.
        """
        index = pd.DatetimeIndex(df.index)
        if index.tz is not None:
            index = index.tz_convert("UTC").tz_localize(None)
        df.index = index.rename("time")
        df.columns = df.columns.set_names(["station_id", "parameter"])
        return df.sort_index()

    def _get_source_observations(
        self,
        source: str,
        stations: Optional[Union[str, List[str]]] = None,
        parameters: Optional[Union[str, List[str]]] = None,
        first_date: Optional[Union[str, pd.Timestamp]] = None,
        last_date: Optional[Union[str, pd.Timestamp]] = None,
        return_mask: bool = False,
        as_numpy: bool = False,
        split: Optional[Literal["train", "test"]] = None,
        copy: bool = False,
    ) -> Union[ObservationDataFrame, np.ndarray, Tuple[Union[ObservationDataFrame, np.ndarray], Union[ObservationDataFrame, np.ndarray]]]:
        """
        Return observations restricted to one source.

        Shapes: (T, N_stations_source * C_parameters_source) as DataFrame, or
        (T, N_stations_source, C_parameters_source) when as_numpy=True.
        """
        obs, mask = self.get_observations(
            stations=stations,
            parameters=parameters,
            first_date=first_date,
            last_date=last_date,
            return_mask=True,
            as_numpy=False,
            split=split,
            copy=True,
        )
        if not isinstance(obs, pd.DataFrame) or obs.empty:
            obs = self._empty_observation_frame()
            mask = self._empty_observation_frame()
        else:
            cols = self._source_columns(source)
            keep = obs.columns.intersection(cols, sort=False)
            obs = obs.loc[:, keep]
            mask = mask.loc[:, keep] if isinstance(mask, pd.DataFrame) and not mask.empty else ~obs.isna()
            obs = ObservationDataFrame(self._standardize_frame(obs))
            mask = ObservationDataFrame(self._standardize_frame(mask))
        if as_numpy:
            if obs.empty:
                obs_out, mask_out = np.empty((0, 0, 0)), np.empty((0, 0, 0), dtype=bool)
            else:
                full = pd.MultiIndex.from_product(
                    [obs.columns.unique(0), obs.columns.unique(1)], names=obs.columns.names
                )
                obs_out = self._export_df(obs.reindex(columns=full), as_numpy=True, copy=copy)
                mask_out = self._export_df(mask.reindex(columns=full, fill_value=False), as_numpy=True, copy=copy)
        else:
            obs_out, mask_out = (obs.copy(), mask.copy()) if copy else (obs, mask)
        return (obs_out, mask_out) if return_mask else obs_out

    def get_tahmo_observations(
        self,
        stations: Optional[Union[str, List[str]]] = None,
        parameters: Optional[Union[str, List[str]]] = None,
        first_date: Optional[Union[str, pd.Timestamp]] = None,
        last_date: Optional[Union[str, pd.Timestamp]] = None,
        return_mask: bool = False,
        as_numpy: bool = False,
        split: Optional[Literal["train", "test"]] = None,
        copy: bool = False,
    ) -> Union[ObservationDataFrame, np.ndarray, Tuple[Union[ObservationDataFrame, np.ndarray], Union[ObservationDataFrame, np.ndarray]]]:
        """
        Get TAHMO station observations only.

        Mathematical docstring:
            DataFrame shape: (T, N_tahmo_stations * C_parameters), naive DatetimeIndex
            named 'time', columns MultiIndex (station_id, parameter).
            Numpy shape (as_numpy=True): (T, N_tahmo_stations, C_parameters).
            Mask has the same shape as the data (True = valid reading).

        Args:
            stations: Station codes, a single code, or None for all TAHMO stations.
            parameters: Canonical or alias parameter names, or None for all.
            first_date: First date (inclusive).
            last_date: Last date (inclusive).
            return_mask: Whether to also return the validity mask.
            as_numpy: Whether to export 3D numpy arrays.
            split: 'train' or 'test' partition anchored at test_set_start.
            copy: Whether to return copies.

        Returns:
            Observations, or (observations, mask) when return_mask is True.
        """
        return self._get_source_observations(
            "tahmo", stations, parameters, first_date, last_date, return_mask, as_numpy, split, copy
        )

    def get_cds_insitu_observations(
        self,
        stations: Optional[Union[str, List[str]]] = None,
        parameters: Optional[Union[str, List[str]]] = None,
        first_date: Optional[Union[str, pd.Timestamp]] = None,
        last_date: Optional[Union[str, pd.Timestamp]] = None,
        return_mask: bool = False,
        as_numpy: bool = False,
        split: Optional[Literal["train", "test"]] = None,
        copy: bool = False,
    ) -> Union[ObservationDataFrame, np.ndarray, Tuple[Union[ObservationDataFrame, np.ndarray], Union[ObservationDataFrame, np.ndarray]]]:
        """
        Get Copernicus CDS in-situ station observations only.

        Mathematical docstring:
            DataFrame shape: (T, N_cds_stations * C_parameters), naive DatetimeIndex
            named 'time', columns MultiIndex (station_id, parameter).
            Numpy shape (as_numpy=True): (T, N_cds_stations, C_parameters).
            Mask has the same shape as the data (True = valid reading).

        Args:
            stations: Station codes, a single code, or None for all CDS stations.
            parameters: Canonical or alias parameter names, or None for all.
            first_date: First date (inclusive).
            last_date: Last date (inclusive).
            return_mask: Whether to also return the validity mask.
            as_numpy: Whether to export 3D numpy arrays.
            split: 'train' or 'test' partition anchored at test_set_start.
            copy: Whether to return copies.

        Returns:
            Observations, or (observations, mask) when return_mask is True.
        """
        return self._get_source_observations(
            "cds_insitu", stations, parameters, first_date, last_date, return_mask, as_numpy, split, copy
        )

    def get_ghcn_observations(
        self,
        stations: Optional[Union[str, List[str]]] = None,
        parameters: Optional[Union[str, List[str]]] = None,
        first_date: Optional[Union[str, pd.Timestamp]] = None,
        last_date: Optional[Union[str, pd.Timestamp]] = None,
        return_mask: bool = False,
        as_numpy: bool = False,
        split: Optional[Literal["train", "test"]] = None,
        copy: bool = False,
    ) -> Union[ObservationDataFrame, np.ndarray, Tuple[Union[ObservationDataFrame, np.ndarray], Union[ObservationDataFrame, np.ndarray]]]:
        """
        Get GHCN station observations only.

        Mathematical docstring:
            DataFrame shape: (T, N_ghcn_stations * C_parameters), naive DatetimeIndex
            named 'time', columns MultiIndex (station_id, parameter).
            Numpy shape (as_numpy=True): (T, N_ghcn_stations, C_parameters).
            Mask has the same shape as the data (True = valid reading).

        Args:
            stations: Station codes, a single code, or None for all GHCN stations.
            parameters: Canonical or alias parameter names, or None for all.
            first_date: First date (inclusive).
            last_date: Last date (inclusive).
            return_mask: Whether to also return the validity mask.
            as_numpy: Whether to export 3D numpy arrays.
            split: 'train' or 'test' partition anchored at test_set_start.
            copy: Whether to return copies.

        Returns:
            Observations, or (observations, mask) when return_mask is True.
        """
        return self._get_source_observations(
            "ghcn", stations, parameters, first_date, last_date, return_mask, as_numpy, split, copy
        )

    @staticmethod
    def align_observations(
        strands: Dict[str, pd.DataFrame],
        how: Literal["outer", "inner"] = "outer",
    ) -> pd.DataFrame:
        """
        Merge per-source observation frames onto a common time index.

        Mathematical docstring:
            Inputs: {name: (T_k, N_k * C_k)}. Output: (T_union, sum_k N_k * C_k) with
            a naive DatetimeIndex named 'time' and columns (source, station_id, parameter).

        Args:
            strands: Mapping from source name to its observation DataFrame.
            how: 'outer' keeps the union of timestamps, 'inner' the intersection.

        Returns:
            Concatenated DataFrame with three-level columns.
        """
        frames = {k: v for k, v in strands.items() if v is not None and not v.empty}
        if not frames:
            cols = pd.MultiIndex.from_tuples([], names=["source", "station_id", "parameter"])
            return pd.DataFrame(index=pd.DatetimeIndex([], name="time"), columns=cols)
        out = pd.concat(frames, axis=1, join=how, keys=list(frames.keys()))
        out.columns = out.columns.set_names(["source", "station_id", "parameter"])
        index = pd.DatetimeIndex(out.index)
        if index.tz is not None:
            index = index.tz_convert("UTC").tz_localize(None)
        out.index = index.rename("time")
        return out.sort_index()

    def get_observations(
        self,
        stations: Optional[Union[str, List[str]]] = None,
        parameters: Optional[Union[str, List[str]]] = None,
        first_date: Optional[Union[str, pd.Timestamp]] = None,
        last_date: Optional[Union[str, pd.Timestamp]] = None,
        return_mask: bool = False,
        as_numpy: bool = False,
        split: Optional[Literal["train", "test"]] = None,
        copy: bool = False,
    ) -> Union[ObservationDataFrame, np.ndarray, Tuple[Union[ObservationDataFrame, np.ndarray], Union[ObservationDataFrame, np.ndarray]]]:
        """
        Get observation time series and quality masks across stations and parameters.

        Mathematical docstring:
            Observation tensor shape: (T_timestamps, N_stations, C_parameters) if as_numpy=True.
            Observation tabular shape: (T_timestamps, N_stations * C_parameters) as MultiIndex DataFrame.
            Quality mask shape: Identical to observation shape, with boolean indicators
            (True = valid measured reading, False = missing/corrupted/masked).

        Args:
            stations: Sequence of station codes, single station code, "all", or None for all available stations.
            parameters: Sequence of parameter names, single parameter name, or None for all active parameters.
            first_date: First date (inclusive) to slice observations.
            last_date: Last date (inclusive) to slice observations.
            return_mask: Whether to return a mask of the observations alongside the data.
            as_numpy: Whether to export observations as 3D numpy arrays of shape (T, N_stations, C_params).
            split: Temporal split partition ('train' or 'test') anchored at test_set_start.
            copy: Whether to return an explicit copy of the observations.

        Returns:
            ObservationDataFrame or np.ndarray, or tuple (data, mask).
        """
        valid_stations = self.validate_stations(stations)
        if not valid_stations:
            out_obs = np.empty((0, 0, 0)) if as_numpy else ObservationDataFrame()
            out_mask = np.empty((0, 0, 0), dtype=bool) if as_numpy else ObservationDataFrame()
            return (out_obs, out_mask) if return_mask else out_obs

        stations_loc = slice(None) if stations is None else valid_stations

        if split is not None:
            if split not in ["train", "test"]:
                raise ValueError(f"Invalid split value: {split}. Expected 'train' or 'test'.")
            assert first_date is None and last_date is None, (
                "Can not provide both a split and first_date and/or last_date."
            )
            if split == "test":
                first_date = self.test_set_start
            elif split == "train":
                last_date = self.test_set_start

        first_date = pd.to_datetime(first_date, utc=True).tz_localize(None) if first_date else None
        last_date = (pd.to_datetime(last_date, utc=True) - pd.Timedelta(minutes=1)).tz_localize(None) if last_date else None

        index_loc = slice(first_date, last_date)

        if not hasattr(self, "observations") or self.observations is None or self.observations.empty:
            out_obs = np.empty((0, 0, 0)) if as_numpy else ObservationDataFrame()
            out_mask = np.empty((0, 0, 0), dtype=bool) if as_numpy else ObservationDataFrame()
            return (out_obs, out_mask) if return_mask else out_obs

        # Match parameters across canonical, alias, and sensor-qualified names
        if parameters is not None:
            req_params = [parameters] if isinstance(parameters, str) else list(parameters)
            col_params = self.observations.columns.get_level_values(1).unique()
            matched_params = []
            for req in req_params:
                req_canon = canonicalize_variable_name(req)
                for c in col_params:
                    if c == req or c == req_canon or c.startswith(req + "_") or c.startswith(req_canon + "_"):
                        matched_params.append(c)
                    elif req.lower() in CANONICAL_VARIABLE_METADATA.get(c, {}).get("aliases", []):
                        matched_params.append(c)
                    elif c.lower() in CANONICAL_VARIABLE_METADATA.get(req_canon, {}).get("aliases", []):
                        matched_params.append(c)
            matched_params = list(dict.fromkeys(matched_params))
            if not matched_params:
                logger.warning("None of requested parameters %s found in dataset.", req_params)
                warnings.warn(f"None of requested parameters {req_params} found in dataset.", UserWarning, stacklevel=2)
                out_obs = np.empty((0, 0, 0)) if as_numpy else ObservationDataFrame()
                out_mask = np.empty((0, 0, 0), dtype=bool) if as_numpy else ObservationDataFrame()
                return (out_obs, out_mask) if return_mask else out_obs
            parameters_loc = matched_params
        else:
            parameters_loc = slice(None)

        obs_slice = self.observations.loc[index_loc, (stations_loc, parameters_loc)]
        mask_slice = self.mask.loc[index_loc, (stations_loc, parameters_loc)] if hasattr(self, "mask") and not self.mask.empty else (~obs_slice.isna())

        obs_out = self._export_df(obs_slice, as_numpy=as_numpy, copy=copy)
        mask_out = self._export_df(mask_slice, as_numpy=as_numpy, copy=copy)

        if not as_numpy:
            obs_out = ObservationDataFrame(obs_out)
            mask_out = ObservationDataFrame(mask_out)

        return (obs_out, mask_out) if return_mask else obs_out

    def load_gridded_datasets(
        self, groups: List[str], 
        years: Optional[Union[int, List[int]]] = None
    ) -> Dict[str, xr.Dataset]:
        """
        Load gridded datasets for the requested groups and years.
        
        Args:
            groups: List of groups to load data from.
            years: List of years to load data for.
            
        Returns:
            Dictionary of loaded datasets.
        """
        self._validate_groups(groups)
        target_years = [years] if isinstance(years, int) else (years or self.years)
        target_years = self._validate_years(target_years)
        gridded_dict = {}
        
        for group in groups:
            datasets = []
            for year in target_years:
                ds = self._open_gridded_zarr(group, year, lazy=True)
                if ds is not None:
                    datasets.append(ds)
                    
            if datasets:
                if len(datasets) == 1:
                    gridded_dict[group] = datasets[0]
                else:
                    gridded_dict[group] = xr.concat(
                        datasets, 
                        dim="time",
                        coords="minimal",
                        data_vars="minimal",
                        compat="override",
                        join="override"
                    )
                    
        return gridded_dict

    VALID_CONSTRAINT_GROUPS = ["chirps", "tamsat"]

    def get_mass_constraints(
        self,
        windows: Windows,
        groups: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        stations: Optional[Union[str, List[str]]] = None,
        first_date: Optional[Union[str, pd.Timestamp]] = None,
        last_date: Optional[Union[str, pd.Timestamp]] = None,
        constraint_freq: str = "1D",
    ) -> Windows:
        """
        Optional: Extract aggregated precipitation totals from CHIRPS/TAMSAT and
        attach them to an existing Windows object as mass conservation targets.

        This method does NOT modify the sequential inputs (station_x, grids_x).
        It populates windows.constraint_targets and windows.temporal_masks exclusively.

        Only groups in ["chirps", "tamsat"] are accepted. All others raise ValueError.

        Mathematical justification:
            The mass conservation constraint enforces:
                sum_{t in constraint_freq} P_hat(t) = P_target
            where P_hat(t) are the hourly predictions and P_target is the
            observed aggregated total from CHIRPS or TAMSAT. This is evaluated
            via Tweedie Deviance Loss in the training loop.

        Args:
            windows: An existing Windows object from get_windows().
            groups: Daily precipitation products to extract. Defaults to ["chirps"].
            years: Years to extract. Defaults to self.years.
            stations: Stations to extract for. Defaults to windows.station_labels.
            first_date: Optional temporal lower bound.
            last_date: Optional temporal upper bound.
            constraint_freq: Pandas frequency string for mass conservation aggregation (e.g., '1D', 'W', 'ME').

        Returns:
            The same Windows object with constraint_targets and temporal_masks populated.
        """
        import warnings

        if groups is None:
            groups = ["chirps"]

        # Strict validation: only daily precipitation products allowed
        invalid_groups = [g for g in groups if g not in self.VALID_CONSTRAINT_GROUPS]
        if invalid_groups:
            raise ValueError(
                f"Invalid mass conservation groups: {invalid_groups}. "
                f"Only {self.VALID_CONSTRAINT_GROUPS} are valid constraint targets. "
                f"Hourly products (ERA5, IMERG) should be passed via `gridded_url` in get_windows()."
            )

        self._validate_groups(groups)

        # Determine stations from the Windows object if not explicitly provided
        constraint_stations = stations or getattr(windows, 'station_labels', None)
        target_years = years or self.years

        # Extract daily products via the existing station extraction pipeline
        ext_dfs, mask_dfs = self.get_gridded_for_stations(
            groups=groups,
            stations=constraint_stations,
            years=target_years,
            first_date=first_date,
            last_date=last_date,
        )

        if not ext_dfs:
            warnings.warn(
                f"No data extracted for mass conservation groups {groups}. "
                "Windows will have empty constraint_targets.",
                UserWarning
            )
            return windows

        # Align target totals to the hourly window temporal extent
        # Use the hourly observation index from the windows to determine constraint boundaries
        if windows.index_x is not None and len(windows.index_x) > 0:
            # The last timestamp of each window defines its reference time
            hourly_index = pd.DatetimeIndex(windows.index_x[:, -1])
        else:
            warnings.warn(
                "Cannot align constraints: windows.index_x is empty.",
                UserWarning
            )
            return windows

        for group in groups:
            if group not in ext_dfs:
                continue

            df = ext_dfs[group]
            mask_df = mask_dfs.get(group, df.isna().astype(int))

            # Ensure timezone-naive for alignment
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if mask_df.index.tz is not None:
                mask_df.index = mask_df.index.tz_localize(None)

            # Resample to requested constraint freq
            target_df = df.resample(constraint_freq).sum(min_count=1)
            target_mask = mask_df.resample(constraint_freq).max()  # Missing if ANY timestep is missing

            # Map each hourly window reference time to its corresponding constraint interval
            ref_dates = hourly_index.floor(constraint_freq) if hasattr(hourly_index, 'floor') else pd.DatetimeIndex([d.floor(constraint_freq) for d in hourly_index])
            unique_targets = pd.DatetimeIndex(sorted(ref_dates.unique()))

            # Find overlapping intervals between the aggregated product and the window span
            overlap_targets = unique_targets.intersection(target_df.index)
            if len(overlap_targets) == 0:
                warnings.warn(
                    f"No temporal overlap between {group} {constraint_freq} aggregated data and window span. "
                    f"Skipping constraint for {group}.",
                    UserWarning
                )
                continue

            # Build the constraint_index mapping: for each window sample, which interval does it belong to?
            target_to_idx = {d: i for i, d in enumerate(overlap_targets)}
            target_index = np.array([target_to_idx.get(d, -1) for d in ref_dates], dtype=np.int64)

            # Filter windows that have valid interval mappings
            valid_mask = target_index >= 0

            # Extract the constraint values as numpy arrays
            # Shape: (n_intervals, n_stations * n_params)
            constraint_vals = target_df.loc[overlap_targets].values.astype(np.float32)
            constraint_mask_vals = target_mask.loc[overlap_targets].values.astype(np.float32)

            # Replace NaN with 0 in constraint values (masked out by the mask anyway)
            constraint_vals = np.nan_to_num(constraint_vals, nan=0.0)

            windows.constraint_targets[group] = constraint_vals
            windows.temporal_masks[group] = constraint_mask_vals
            windows.constraint_resolution = constraint_freq

        # Build the constraint_index from the last processed group (shared temporal axis)
        if windows.index_x is not None and len(windows.index_x) > 0:
            hourly_index = pd.DatetimeIndex(windows.index_x[:, -1])
            ref_dates = hourly_index.floor(constraint_freq) if hasattr(hourly_index, 'floor') else pd.DatetimeIndex([d.floor(constraint_freq) for d in hourly_index])
            unique_targets = pd.DatetimeIndex(sorted(ref_dates.unique()))
            target_to_idx = {d: i for i, d in enumerate(unique_targets)}
            windows.constraint_time_index = np.array(
                [target_to_idx.get(d, -1) for d in ref_dates], dtype=np.int64
            )

        return windows

    def load_gridded_data(self, gridded_url: Union[str, List[str]]) -> xr.Dataset:
        """
        Load gridded data for the requested groups and years.
        
        Args:
            gridded_url: List of groups to load data from.
            
        Returns:
            Dictionary of loaded datasets.
        """
        groups = [gridded_url] if isinstance(gridded_url, str) else gridded_url
        gridded_dict = self.load_gridded_datasets(groups=groups)
        
        if not gridded_dict:
            return xr.Dataset()
        elif len(gridded_dict) == 1:
            return list(gridded_dict.values())[0]
        else:
            return xr.merge(gridded_dict.values())

    def get_observation_windows(
        self,
        window_size: int,
        horizon_size: int,
        delay: int = 0,
        stride: int = 1,
        stations: Optional[Union[str, List[str]]] = None,
        parameters: Optional[Union[str, List[str]]] = None,
        first_date: Optional[Union[str, pd.Timestamp]] = None,
        last_date: Optional[Union[str, pd.Timestamp]] = None,
        split: Optional[Literal["train", "test"]] = None,
        as_xarray: bool = False,
    ) -> Windows:
        """
        Get observation windows for the requested stations.
        
        Args:
            window_size: Size of the observation window.
            horizon_size: Size of the horizon window.
            delay: Number of steps between the end of the input window and start of the horizon.
            stride: Number of steps to advance the sliding window at each extraction.
            stations: List of stations to get observations for.
            parameters: List of parameters to get observations for.
            first_date: First date to get observations for.
            last_date: Last date to get observations for.
            split: Split to use for the observations.
            as_xarray: Whether to return the observations as xarray datasets.
            
        Returns:
            Windows: Windows of observations.
        """
        
        if split is not None and split not in ["train", "test"]:
            raise ValueError(f"Invalid split value: {split}. Expected 'train' or 'test'.")

        observations, mask = self.get_observations(
            stations=stations,
            parameters=parameters,
            first_date=first_date,
            last_date=last_date,
            split=split,
            as_numpy=False,
            return_mask=True,
            copy=True,
        )

        sample_span = window_size + delay + horizon_size
        if len(observations) < sample_span:
            empty_arr = np.array([])
            return Windows(
                station_x=empty_arr, station_mask_x=empty_arr,
                station_y=empty_arr, station_mask_y=empty_arr,
                delay=delay
            )

        index = observations.index
        obs_np = self._export_df(observations, as_numpy=True, copy=False)
        mask_np = self._export_df(mask, as_numpy=True, copy=False, fill_value=False)

        x_slice = slice(None) if (horizon_size + delay == 0) else slice(None, len(obs_np) - horizon_size - delay)
        x = sliding_window_view(obs_np[x_slice], window_size)[::stride]
        mask_x = sliding_window_view(mask_np[x_slice], window_size)[::stride]
        index_x = sliding_window_view(index[x_slice].values, window_size)[::stride]
        
        y_start = window_size + delay
        if horizon_size > 0:
            y = sliding_window_view(obs_np[y_start:], horizon_size)[::stride]
            mask_y = sliding_window_view(mask_np[y_start:], horizon_size)[::stride]
            index_y = sliding_window_view(index[y_start:].values, horizon_size)[::stride]
        else:
            y, mask_y, index_y = np.array([]), np.array([]), np.array([])

        windows = Windows(
            station_x=x, station_mask_x=mask_x, 
            station_y=y, station_mask_y=mask_y, 
            index_x=index_x, index_y=index_y,
            delay=delay
        )

        if not as_xarray:
            return windows

        return self.np_windows_as_xr(windows, stations, parameters)

    def get_windows(
        self,
        window_size: int,
        horizon_size: int,
        delay: int = 0,
        stride: int = 1,
        lazy: bool = True,
        stations: Optional[Union[str, List[str]]] = None,
        parameters: Optional[Union[str, List[str]]] = None,
        first_date: Optional[Union[str, pd.Timestamp]] = None,
        last_date: Optional[Union[str, pd.Timestamp]] = None,
        split: Optional[Literal["train", "test"]] = None,
        gridded_url: Optional[Union[str, List[str]]] = None,
        gridded_variables: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        method_overrides: Optional[Dict[str, str]] = None,
        patch_radius: int = 1,
        crop_to_stations: bool = False,
        pressure_levels: Optional[List[int]] = None,
        ml_task: Literal["downscaling", "forecasting", "correction", "anomaly_detection"] = "downscaling",
        spatial_mode: Literal["stations", "grid"] = "stations",
        target_node: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        interpolation: Optional[str] = None,
        as_xarray: bool = False,
        temporal_resolution: Optional[str] = None,
    ) -> Windows:
        """
        N-Grid Heterogeneous Extraction mapping native resolutions organically.

        Args:
            window_size (int): The number of time steps to include in the input sequence.
            horizon_size (int): The number of time steps to predict.
            delay (int): The number of time steps to delay the input sequence.
            stride (int): The number of time steps to skip between windows.
            lazy (bool): Whether to use lazy loading.
            stations (Optional[Union[str, List[str]]]): List of station IDs to extract data for.
            parameters (Optional[Union[str, List[str]]]): List of variables to extract.
            first_date (Optional[Union[str, pd.Timestamp]]): First date to extract data for.
            last_date (Optional[Union[str, pd.Timestamp]]): Last date to extract data for.
            split (Optional[Literal["train", "test"]]): Split to extract data for.
            gridded_url (Optional[Union[str, List[str]]]): List of gridded URLs to extract data for.
            gridded_variables (Optional[Union[List[str], Dict[str, List[str]]]]): List of gridded variables to extract.
            method_overrides (Optional[Dict[str, str]]): Dictionary of method overrides e.g
            {"temperature": "patch_max", "wind_speed": "nearest", "total_precipitation": "linear"}.
            patch_radius (int): Radius of the patch to extract, 1 = 3x3 grid, 2 = 5x5 grid, etc.
            crop_to_stations (bool): Whether to crop the gridded data to the stations.
            pressure_levels (Optional[List[int]]): List of pressure levels to extract data for.
            ml_task (Literal["downscaling", "forecasting", "correction", "anomaly_detection"]): Machine learning task to perform.
            spatial_mode (Literal["stations", "grid"]): Spatial mode to use for the data.
            target_node (Optional[str]): Target node to extract data for.
            bbox (Optional[Tuple[float, float, float, float]]): Bounding box to extract data for.
            interpolation (Optional[str]): Global interpolation method overriding default ("nearest", "bilinear", "peak_preserving", "patch").
            as_xarray (bool): Whether to return the data as xarray datasets.
            
        Returns:
            Windows: Windows of data.
        """
        if split is not None and split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split value: {split}. Expected 'train', 'val', or 'test'.")

        if ml_task not in ["downscaling", "forecasting", "correction", "anomaly_detection"]:
            raise ValueError("ml_task must be 'downscaling', 'forecasting', 'correction', or 'anomaly_detection'.")

        if spatial_mode not in ["stations", "grid"]:
            raise ValueError("spatial_mode must be either 'stations' or 'grid'.")

        if not lazy:
            lazy = True # Mandatory logic enforcing isolated VRAM handling Dask per dict entry
            print("Warning: N-Grid architectures natively map `lazy=True`. Eager is bypassed.")
            
        if interpolation:
            if method_overrides is None:
                method_overrides = {}
            interp_map = {
                "patch": "patch_flat",
                "peak_preserving": "patch_max",
                "bilinear": "linear",
                "nearest": "nearest"
            }
            mapped_method = interp_map.get(interpolation, interpolation)
            if gridded_variables:
                for grp, vars_list in (gridded_variables.items() if isinstance(gridded_variables, dict) else {g: gridded_variables for g in (gridded_url if isinstance(gridded_url, list) else [gridded_url])}.items()):
                    for var in vars_list:
                        if var not in method_overrides:
                            method_overrides[var] = mapped_method

        observations, mask = self.get_observations(
            stations=stations,
            parameters=parameters,
            first_date=first_date,
            last_date=last_date,
            split=split,
            as_numpy=False,
            return_mask=True,
            copy=True,
        )

        valid_stations_subset = self.validate_stations(stations) if stations else self.stations_table.index.tolist()

        # Geographic bounding box filtering: filter stations to those within bbox
        if bbox is not None:
            lat_min, lon_min, lat_max, lon_max = bbox
            st = self.stations_table.loc[valid_stations_subset]
            in_bbox = (
                (st['latitude'] >= lat_min) & (st['latitude'] <= lat_max) &
                (st['longitude'] >= lon_min) & (st['longitude'] <= lon_max)
            )
            valid_stations_subset = st[in_bbox].index.tolist()
            if not valid_stations_subset:
                raise ValueError(
                    f"No stations found within bbox={bbox}. "
                    f"Expand the region or check coordinate order (lat_min, lon_min, lat_max, lon_max)."
                )
            # Re-extract observations for the bbox-filtered stations only
            observations, mask = self.get_observations(
                stations=valid_stations_subset,
                parameters=parameters,
                first_date=first_date,
                last_date=last_date,
                split=split,
                as_numpy=False,
                return_mask=True,
                copy=True,
            )

        # Unified Multi-Rate Temporal Resampling for Observations
        if temporal_resolution is not None and temporal_resolution.lower() not in ["1h", "h", "native"]:
            # Use permissive min_count=1 for all temporal resolutions (e.g. 3h, 6h, 1D) 
            # to preserve partial precipitation accumulations for anomaly detection
            precip_min_count = 1
                
            param_list = [parameters] if isinstance(parameters, str) else (parameters or self.parameters)
            types = self.infer_variable_types(param_list)
            
            resampled_cols = {}
            for param in param_list:
                if param in observations.columns.get_level_values(1):
                    param_df = observations.xs(param, axis=1, level=1, drop_level=False)
                    p_type = types.get(param, "PHYSICAL")
                    if p_type == "PRECIPITATION" or p_type == "FLUX":
                        resampled_cols[param] = param_df.resample(temporal_resolution).sum(min_count=precip_min_count)
                    elif p_type == "STATIC":
                        resampled_cols[param] = param_df.resample(temporal_resolution).first()
                    else:
                        resampled_cols[param] = param_df.resample(temporal_resolution).mean()
            
            if resampled_cols:
                observations = pd.concat(list(resampled_cols.values()), axis=1).sort_index(axis=1)
                
            # For the missing mask, if any timestamp within the interval was missing, the interval might be considered missing.
            # Using max() correctly flags True if any sub-interval had True. But since we use min_count=21, 
            # maybe we use sum() > (hours_in_res - precip_min_count). For now, strict max() is safest for physical fields.
            mask = mask.resample(temporal_resolution).max()

        station_static_whitelist = [
            'dem', 'Slope', 'Aspect_sin', 'Aspect_cos', 
            'LandCover', 'DistanceToWater', 'TPI_2000M'
        ]
        available_static = [c for c in station_static_whitelist if c in self.stations_table.columns]
        
        cat_cols = ['LandCover']
        valid_cat = [c for c in cat_cols if c in available_static]
        valid_cont = [c for c in available_static if c not in valid_cat]

        windows = Windows(
            station_x=np.array([]), station_mask_x=np.array([]),
            station_y=np.array([]), station_mask_y=np.array([]),
            delay=delay, target_node=target_node or "tahmo"
        )
        
        sample_span = window_size + delay + horizon_size
        if len(observations) < sample_span:
            return windows

        # 1. Point Stations Sliding Vectors
        obs_np = self._export_df(observations, as_numpy=True, copy=False)
        if np.issubdtype(obs_np.dtype, np.floating): obs_np = obs_np.astype(np.float32, copy=False)
        mask_np = self._export_df(mask, as_numpy=True, copy=False, fill_value=False)

        # RETROSPECTIVE TASKS: Symmetric Windowing [t-tau, t, t+tau] 
        # This allows the GNN to "scan" the sequence to align ERA5 phase/timing errors.
        if ml_task in ["correction", "anomaly_detection"] and horizon_size == 1:
            # window_size is total context (e.g. 48h). Y is the midpoint.
            mid = window_size // 2
            
            x_slice = slice(None)
            windows.station_x = sliding_window_view(obs_np[x_slice], window_size)[::stride]
            windows.station_mask_x = sliding_window_view(mask_np[x_slice], window_size)[::stride]
            windows.index_x = sliding_window_view(observations.index.values[x_slice], window_size)[::stride]
            
            y_slice = slice(mid, len(obs_np) - (window_size - mid - 1))
            y_is_single_point = True
            
            # Y extracted as the midpoint observation
            y_pts = obs_np[y_slice]
            m_pts = mask_np[y_slice]
            i_pts = observations.index.values[y_slice]
            
            # Ensure batch alignment (stride applied after slicing)
            windows.station_y = y_pts[::stride][:, np.newaxis, ...]
            windows.station_mask_y = m_pts[::stride][:, np.newaxis, ...]
            windows.index_y = i_pts[::stride][:, np.newaxis]
            
            # Valid indices: 0 to len - window_size
            windows.valid_batch_indices = np.arange(0, len(obs_np) - window_size + 1, stride)
            
        # FORECASTING / DOWNSCALING / MULTI-HORIZON: Strict Causal Windowing [t-tau, t] -> [t+delay, t+delay+horizon]
        else:
            sample_span = window_size + delay + horizon_size
            if len(observations) < sample_span:
                return windows

            x_slice = slice(None, len(obs_np) - horizon_size - delay)
            windows.station_x = sliding_window_view(obs_np[x_slice], window_size)[::stride]
            windows.station_mask_x = sliding_window_view(mask_np[x_slice], window_size)[::stride]
            windows.index_x = sliding_window_view(observations.index.values[x_slice], window_size)[::stride]
            
            y_start = window_size + delay
            y_slice = slice(y_start, None)
            y_is_single_point = False
            
            windows.station_y = sliding_window_view(obs_np[y_slice], horizon_size)[::stride]
            windows.station_mask_y = sliding_window_view(mask_np[y_slice], horizon_size)[::stride]
            windows.index_y = sliding_window_view(observations.index.values[y_slice], horizon_size)[::stride]
            windows.valid_batch_indices = np.arange(0, len(observations) - sample_span + 1, stride)
            
        if valid_cont:
             windows.station_static = self.stations_table.loc[valid_stations_subset, valid_cont].values.astype(np.float32, copy=False)
        if valid_cat:
             windows.station_static_categorical = np.nan_to_num(self.stations_table.loc[valid_stations_subset, valid_cat].values, nan=0).astype(np.int64, copy=False)

        # 2. Dynamic N-Grids Mapping
        if gridded_url:
            groups = [gridded_url] if isinstance(gridded_url, str) else list(gridded_url)
            
            # Physically Prohibit Daily Products as Sequential PyG Inputs
            daily_conflicts = [g for g in groups if g in ["chirps", "tamsat"]]
            if daily_conflicts:
                import warnings
                warnings.warn(
                    "CHIRPS and TAMSAT are native daily products"
                )

            target_years = list(set(observations.index.year)) if not observations.empty else self.years
            
            per_grid_datasets = {}
            for group in groups:
                 if spatial_mode == "stations":
                      ext_dfs, mask_dfs = self.get_gridded_for_stations(
                           groups=[group],
                           variables=gridded_variables[group] if isinstance(gridded_variables, dict) else gridded_variables,
                           method_overrides=method_overrides,
                           patch_radius=patch_radius,
                           first_date=first_date,
                           last_date=last_date
                      )
                      if ext_dfs and group in ext_dfs:
                           per_grid_datasets[group] = ext_dfs[group]
                 elif spatial_mode == "grid":
                      yearly_ds_list = []
                      for year in target_years:
                           ds = self._open_gridded_zarr(group, year, lazy=True)
                           if ds is None: continue
                           
                           # Validate constraints dynamically
                           validate_gridded_dataset(
                               ds, 
                               product_name=group, 
                               requested_freq=temporal_resolution or self.freq,
                               bbox=bbox
                           )
                           
                           if first_date or last_date:
                               f_dt = pd.to_datetime(first_date, utc=True).tz_localize(None) if first_date else None
                               l_dt = pd.to_datetime(last_date, utc=True).tz_localize(None) if last_date else None
                               ds = ds.sel(time=slice(f_dt, l_dt))
                               
                           if len(ds.time) == 0: continue
                           
                           req_vars = gridded_variables[group] if isinstance(gridded_variables, dict) else (gridded_variables if gridded_variables else list(ds.data_vars))
                           valid_vars = [v for v in req_vars if v in ds.data_vars]
                           if not valid_vars: continue
                           
                           # Determine spatial crop bounds
                           should_crop = crop_to_stations or (bbox is not None)
                           if should_crop:
                                lat_dim = 'lat' if 'lat' in ds.coords else 'latitude'
                                lon_dim = 'lon' if 'lon' in ds.coords else 'longitude'

                                if bbox is not None:
                                    # Explicit bbox supersedes crop_to_stations
                                    # Add 0.25-degree buffer per LAM best practice
                                    buf = 0.25
                                    min_lat, min_lon = bbox[0] - buf, bbox[1] - buf
                                    max_lat, max_lon = bbox[2] + buf, bbox[3] + buf
                                else:
                                    # Fall back to station-extent cropping
                                    latitudes = self.stations_table.loc[valid_stations_subset, 'latitude']
                                    longitudes = self.stations_table.loc[valid_stations_subset, 'longitude']
                                    min_lat, max_lat = latitudes.min(), latitudes.max()
                                    min_lon, max_lon = longitudes.min(), longitudes.max()

                                lats = ds[lat_dim].values
                                lons = ds[lon_dim].values
                                
                                lat1, lat2 = np.abs(lats - min_lat).argmin(), np.abs(lats - max_lat).argmin()
                                lon1, lon2 = np.abs(lons - min_lon).argmin(), np.abs(lons - max_lon).argmin()
                                
                                l_min, l_max = min(lat1, lat2), max(lat1, lat2)
                                n_min, n_max = min(lon1, lon2), max(lon1, lon2)
                                
                                ds = ds[valid_vars].isel({
                                    lat_dim: slice(max(0, l_min), min(len(lats), l_max + 1)),
                                    lon_dim: slice(max(0, n_min), min(len(lons), n_max + 1))
                                })
                           else:
                                ds = ds[valid_vars]
                                
                           yearly_ds_list.append(ds)
                           
                      if yearly_ds_list:
                           concat_ds = xr.concat(yearly_ds_list, dim="time", compat="override", coords="minimal")
                           if pressure_levels is not None and "pressure" in concat_ds.coords:
                                valid_levels = [p for p in pressure_levels if p in concat_ds.pressure.values]
                                if valid_levels:
                                     concat_ds = concat_ds.sel(pressure=valid_levels)
                           per_grid_datasets[group] = concat_ds

            if per_grid_datasets:
                 if temporal_resolution is not None and temporal_resolution.lower() not in ["1h", "h", "native"]:
                     # Apply temporal resolution alignment to all loaded grid structures
                     for g_name, g_ds in list(per_grid_datasets.items()):
                         if isinstance(g_ds, pd.DataFrame):
                             var_types = self.infer_variable_types(g_ds.columns.get_level_values(1).unique())
                             resampled_cols = {}
                             for param in g_ds.columns.get_level_values(1).unique():
                                 param_df = g_ds.xs(param, axis=1, level=1, drop_level=False)
                                 p_type = var_types.get(param, "PHYSICAL")
                                 if p_type == "PRECIPITATION" or p_type == "FLUX":
                                     resampled_cols[param] = param_df.resample(temporal_resolution).sum(min_count=1)
                                 else:
                                     resampled_cols[param] = param_df.resample(temporal_resolution).mean()
                             if resampled_cols:
                                 per_grid_datasets[g_name] = pd.concat(list(resampled_cols.values()), axis=1).sort_index(axis=1)
                         else:
                             # Xarray resampling
                             var_types = self.infer_variable_types(list(g_ds.data_vars))
                             precip_vars = [v for v, t in var_types.items() if t in ["PRECIPITATION", "FLUX"]]
                             phys_vars = [v for v, t in var_types.items() if t not in ["PRECIPITATION", "FLUX"]]
                             
                             resampled_parts = []
                             if precip_vars:
                                 resampled_parts.append(g_ds[precip_vars].resample(time=temporal_resolution).sum(skipna=False))
                             if phys_vars:
                                 resampled_parts.append(g_ds[phys_vars].resample(time=temporal_resolution).mean(skipna=False))
                             
                             if resampled_parts:
                                 per_grid_datasets[g_name] = xr.merge(resampled_parts, compat="override")

                 common_times = observations.index
                 for g_name, g_ds in per_grid_datasets.items():
                      if isinstance(g_ds, pd.DataFrame):
                           common_times = common_times.intersection(g_ds.index)
                      else:
                           common_times = common_times.intersection(pd.DatetimeIndex(g_ds.time.values))
                           
                 obs_mask = np.isin(observations.index, common_times)
                 x_idx_mask = np.isin(windows.index_x[:, -1], common_times)
                 if windows.index_y is not None and len(windows.index_y) > 0:
                     if getattr(windows.index_y, "ndim", 0) > 1 and windows.index_y.shape[1] > 0:
                         x_idx_mask &= np.isin(windows.index_y[:, -1], common_times)
                     elif getattr(windows.index_y, "ndim", 0) == 1:
                         x_idx_mask &= np.isin(windows.index_y, common_times)
                     
                 windows.station_x = windows.station_x[x_idx_mask]
                 if windows.station_y is not None and len(windows.station_y) > 0:
                     windows.station_y = windows.station_y[x_idx_mask]
                 windows.station_mask_x = windows.station_mask_x[x_idx_mask]
                 if windows.station_mask_y is not None and len(windows.station_mask_y) > 0:
                     windows.station_mask_y = windows.station_mask_y[x_idx_mask]
                 windows.index_x = windows.index_x[x_idx_mask]
                 if windows.index_y is not None and len(windows.index_y) > 0:
                     windows.index_y = windows.index_y[x_idx_mask]
                 if windows.valid_batch_indices is not None:
                      windows.valid_batch_indices = windows.valid_batch_indices[x_idx_mask]

                 for g_name, g_ds in per_grid_datasets.items():
                      if isinstance(g_ds, pd.DataFrame):
                           g_df = g_ds.loc[common_times]
                           grid_np = self._export_df(g_df, as_numpy=True, copy=False).astype(np.float32, copy=False)
                           windows.grids_features[g_name] = g_df.columns.unique(level=1).tolist()
                           windows.grids_x[g_name] = sliding_window_view(grid_np[x_slice], window_size, axis=0)[::stride][x_idx_mask]
                           if y_is_single_point:
                               windows.grids_y[g_name] = grid_np[y_slice][::stride][x_idx_mask][:, np.newaxis, ...]
                           else:
                               windows.grids_y[g_name] = sliding_window_view(grid_np[y_slice], horizon_size, axis=0)[::stride][x_idx_mask]
                      else:
                           g_ds = g_ds.sel(time=common_times)
                           windows.grids_features[g_name] = list(g_ds.data_vars)
                           windows.grids_x[g_name] = g_ds
                           
                           lat_dim = 'lat' if 'lat' in g_ds.coords else 'latitude'
                           lon_dim = 'lon' if 'lon' in g_ds.coords else 'longitude'
                           g_lats = g_ds[lat_dim].values
                           g_lons = g_ds[lon_dim].values
                           
                           windows.grids_lats[g_name] = g_lats
                           windows.grids_lons[g_name] = g_lons
                           
                           if ml_task in ["forecasting", "downscaling", "correction"] and horizon_size > 0:
                                windows.grids_y[g_name] = g_ds

        if self._topo_ds is not None:
            all_lats, all_lons = [], []
            for lat_arr in windows.grids_lats.values(): all_lats.extend(lat_arr)
            for lon_arr in windows.grids_lons.values(): all_lons.extend(lon_arr)
            if not all_lats and len(valid_stations_subset) > 0:
                loc = self.stations_table.loc[valid_stations_subset]
                all_lats.extend(loc["latitude"].values)
                all_lons.extend(loc["longitude"].values)
                
            if all_lats and all_lons:
                min_lat, max_lat = min(all_lats) - 0.5, max(all_lats) + 0.5
                min_lon, max_lon = min(all_lons) - 0.5, max(all_lons) + 0.5
                
                t_lat_dim = 'lat' if 'lat' in self._topo_ds.coords else 'latitude'
                t_lon_dim = 'lon' if 'lon' in self._topo_ds.coords else 'longitude'
                
                lat_dnm = self._topo_ds[t_lat_dim].values
                lon_dnm = self._topo_ds[t_lon_dim].values
                lat_slice = slice(max_lat, min_lat) if lat_dnm[0] > lat_dnm[-1] else slice(min_lat, max_lat)
                lon_slice = slice(max_lon, min_lon) if lon_dnm[0] > lon_dnm[-1] else slice(min_lon, max_lon)
                    
                topo_sliced = self._topo_ds.sel({t_lat_dim: lat_slice, t_lon_dim: lon_slice}).load()
                
                topo_vars = list(topo_sliced.data_vars)
                grid_cat = [v for v in topo_vars if v in cat_cols]
                grid_cont = [v for v in topo_vars if v not in cat_cols]
                
                c_arr_list = [topo_sliced[tv].values.astype(np.float32)[np.newaxis, ...] for tv in grid_cont]
                if c_arr_list: windows.native_topography = np.concatenate(c_arr_list, axis=0)
                
                cat_arr_list = [np.nan_to_num(topo_sliced[tv].values, nan=0).astype(np.int64)[np.newaxis, ...] for tv in grid_cat]
                if cat_arr_list: windows.native_topography_categorical = np.concatenate(cat_arr_list, axis=0)
                
                windows.native_topography_lats = topo_sliced[t_lat_dim].values
                windows.native_topography_lons = topo_sliced[t_lon_dim].values

        windows.station_labels = valid_stations_subset
        windows.feature_labels = [parameters] if isinstance(parameters, str) else list(parameters or self.parameters)
        windows.static_feature_labels = valid_cont
        windows.static_categorical_labels = valid_cat

        # Station geographic coordinates for graph edge construction
        coord_cols = ['latitude', 'longitude']
        if all(c in self.stations_table.columns for c in coord_cols):
            windows.station_coords = self.stations_table.loc[valid_stations_subset, coord_cols].values.astype(np.float32, copy=False)

        if as_xarray:
            return self.np_windows_as_xr(windows, stations, parameters)

        return windows
    def np_windows_as_xr(
        self,
        windows: Windows,
        stations: Optional[Union[str, List[str]]],
        parameters: Optional[Union[str, List[str]]],
    ) -> Windows:

        """
        Convert numpy windows to xarray windows.

        Args:
            windows (Windows): Numpy windows.
            stations (Optional[Union[str, List[str]]]): List of station IDs.
            parameters (Optional[Union[str, List[str]]]): List of parameters.

        Returns:
            Windows: Xarray windows.
        """
        
        coord_info = self._get_windows_coordinates(windows, stations, parameters)
        return self._pack_windows_as_xarray(
            windows,
            feature=coord_info["features"],
            reftime=coord_info["reftime"],
            lagtime=coord_info["lagtime"],
            leadtime=coord_info["leadtime"],
            stations=coord_info["station"],
        )

    def get_xarray_batch(
        self,
        windows: Windows,
        batch_idx: int,
        stations: Optional[Union[str, List[str]]] = None,
        parameters: Optional[Union[str, List[str]]] = None,
    ) -> Windows:
        """
        Dynamically extracts a single window (batch index) and converts exactly that slice into 
        an Xarray Dataset. This guarantees a O(1) memory footprint for visualization/inspection, 
        bypassing the contiguous RAM expansion of mapping all rolling windows.

        Args:
            windows (Windows): Numpy windows.
            batch_idx (int): Batch index.
            stations (Optional[Union[str, List[str]]]): List of station IDs.
            parameters (Optional[Union[str, List[str]]]): List of parameters.

        Returns:
            Windows: Xarray windows.
        """
        import copy
        single = copy.copy(windows)
        
        single.station_x = single.station_x[batch_idx:batch_idx+1]
        single.station_mask_x = single.station_mask_x[batch_idx:batch_idx+1]
        if single.station_y is not None and len(single.station_y) > 0:
            single.station_y = single.station_y[batch_idx:batch_idx+1]
            single.station_mask_y = single.station_mask_y[batch_idx:batch_idx+1]
            
        single.index_x = single.index_x[batch_idx:batch_idx+1]
        if single.index_y is not None and len(single.index_y) > 0:
            single.index_y = single.index_y[batch_idx:batch_idx+1]
            
        for g_name in list(single.grids_x.keys()):
            single.grids_x[g_name] = single.grids_x[g_name][batch_idx:batch_idx+1]
            if g_name in single.grids_y:
                single.grids_y[g_name] = single.grids_y[g_name][batch_idx:batch_idx+1]
            
        use_stations = stations if stations is not None else windows.station_labels
        use_parameters = parameters if parameters is not None else windows.feature_labels
            
        return self.np_windows_as_xr(single, use_stations, use_parameters)

    def _get_windows_coordinates(self, windows: Windows, stations=None, parameters=None) -> dict:
        reftime = pd.DatetimeIndex(windows.index_x[:, -1]).tz_localize(None)
        window_size = windows.station_x.shape[1]
        horizon_size = windows.station_y.shape[1] if windows.station_y is not None and len(windows.station_y) > 0 else 0
        
        # Use the class-level frequency instead of hardcoded 1h fallback
        parsed_freq = pd.tseries.frequencies.to_offset(self.freq) if self.freq else pd.Timedelta("1h")
        if self.observations.empty:
            freq = parsed_freq
        else:
            freq = self.observations.index.freq if self.observations.index.freq else parsed_freq
            
        leadtime = pd.timedelta_range(start=1 * freq, periods=horizon_size, freq=freq)
        lagtime = pd.timedelta_range(start=-(window_size - 1) * freq, end=0, freq=freq)

        stations_arr = self.stations if stations is None else pd.Index([stations] if isinstance(stations, str) else stations)
        features_arr = self.parameters if parameters is None else pd.Index([parameters] if isinstance(parameters, str) else parameters)

        return {
            "reftime": reftime,
            "leadtime": leadtime,
            "lagtime": lagtime,
            "station": stations_arr,
            "features": features_arr,
        }

    def _pack_windows_as_xarray(
        self,
        windows: Windows,
        feature: Union[pd.Index, np.ndarray],
        reftime: Union[np.ndarray, pd.DatetimeIndex],
        lagtime: Sequence[pd.Timedelta],
        leadtime: Sequence[pd.Timedelta],
        stations: Union[pd.Index, np.ndarray],
    ) -> Windows:
        
        def get_data_vars(data, feature_names, x_or_y: str, is_grid: bool = False):
            if data is None:
                return {}
            second_dim = "lag" if x_or_y == "x" else "lead"
            
            if is_grid:
                # Shape: (Batch, Time, Channels, Lat, Lon)
                return {
                    str(f): (["reftime", second_dim, "lat", "lon"], data[:, :, i, :, :])
                    for i, f in enumerate(feature_names)
                }
            else:
                # Shape: (Batch, Time, Station, Features)
                if data.ndim == 3:
                    data = np.expand_dims(data, axis=-1)
                return {
                    str(f): (["reftime", second_dim, "station"], data[..., i])
                    for i, f in enumerate(feature_names)
                }

        coords_x = {
            "reftime": ("reftime", reftime.values if hasattr(reftime, 'values') else np.array(reftime)),
            "lag": ("lag", lagtime.values if hasattr(lagtime, 'values') else np.array(lagtime)),
            "station": ("station", stations.values if hasattr(stations, 'values') else np.array(stations))
        }
        
        # Base coords for stations
        coords_y = coords_x.copy()
        del coords_y["lag"]
        coords_y["lead"] = ("lead", leadtime.values if hasattr(leadtime, 'values') else np.array(leadtime))

        # Pack Ground Truth (Always Station-based)
        windows.station_x = xr.Dataset(get_data_vars(windows.station_x, feature, "x", is_grid=False), coords=coords_x)
        windows.station_mask_x = xr.Dataset(get_data_vars(windows.station_mask_x, feature, "x", is_grid=False), coords=coords_x)
        if windows.station_y is not None and len(windows.station_y) > 0:
            windows.station_y = xr.Dataset(get_data_vars(windows.station_y, feature, "y", is_grid=False), coords=coords_y)
            windows.station_mask_y = xr.Dataset(get_data_vars(windows.station_mask_y, feature, "y", is_grid=False), coords=coords_y)
        
        # Pack Gridded Tensors if they exist
        for g_name, gx_data in windows.grids_x.items():
            is_grid_mode = gx_data.ndim == 5 if hasattr(gx_data, "ndim") else False
            gx_feats = windows.grids_features[g_name]
            
            if is_grid_mode:
                grid_coords_x = {
                    "reftime": coords_x["reftime"],
                    "lag": coords_x["lag"],
                    "lat": ("lat", windows.grids_lats[g_name]),
                    "lon": ("lon", windows.grids_lons[g_name])
                }
                windows.grids_x[g_name] = xr.Dataset(get_data_vars(gx_data, gx_feats, "x", is_grid=True), coords=grid_coords_x)
                if g_name in windows.grids_y:
                    grid_coords_y = {
                        "reftime": coords_y["reftime"],
                        "lead": coords_y["lead"],
                        "lat": ("lat", windows.grids_lats[g_name]),
                        "lon": ("lon", windows.grids_lons[g_name])
                    }
                    windows.grids_y[g_name] = xr.Dataset(get_data_vars(windows.grids_y[g_name], gx_feats, "y", is_grid=True), coords=grid_coords_y)
            else:
                windows.grids_x[g_name] = xr.Dataset(get_data_vars(gx_data, gx_feats, "x", is_grid=False), coords=coords_x)
                if g_name in windows.grids_y:
                    windows.grids_y[g_name] = xr.Dataset(get_data_vars(windows.grids_y[g_name], gx_feats, "y", is_grid=False), coords=coords_y)

        if windows.native_topography is not None:
            topo_coords = {"lat": ("lat", windows.native_topography_lats), "lon": ("lon", windows.native_topography_lons)}
            topo_dict = {}
            for i, f in enumerate(windows.static_feature_labels or []):
                if i < len(windows.native_topography):
                    topo_dict[f] = (["lat", "lon"], windows.native_topography[i])
            windows.native_topography = xr.Dataset(topo_dict, coords=topo_coords)

        return windows


def extract_stations(
    source: str = "tahmo",
    start_date: Union[str, datetime, date, pd.Timestamp] = "2023-01-01",
    end_date: Union[str, datetime, date, pd.Timestamp] = "2023-01-05",
    bbox: Optional[Union[SpatialBoundingBox, Sequence[float]]] = None,
    variables: Optional[Sequence[str]] = None,
    output_dir: Union[str, Path] = "./mashariki_data/obs",
    save_parquet: bool = True,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    **kwargs: Any,
) -> Union[pd.DataFrame, Dict[str, Path]]:
    """
    Extract in-situ ground station observations across Africa.

    Supported sources:
        - "tahmo": Trans-African Hydro-Meteorological Observatory stations.
        - "ghcn" / "ghcnd": Global Historical Climatology Network (Daily/Hourly).
        - "cds_insitu": Copernicus Climate Data Store In-Situ observations.

    Mathematical docstring:
        Observation shape: (T_timestamps, N_stations * C_variables) or (T_timestamps, C_variables) per station.
        Output Parquet files maintain un-averaged collocated sensor channels (<var>_<sensor>).

    Args:
        source: Meteorological ground observation source network.
        start_date: Initial observation timestamp.
        end_date: Terminal observation timestamp.
        bbox: Spatial boundary as SpatialBoundingBox or sequence [lat_min, lon_min, lat_max, lon_max].
        variables: Optional sequence of target physical variables.
        output_dir: Local filesystem directory for saving non-destructive Parquet files.
        save_parquet: If True, writes individual station Parquet files and returns a mapping
            of station IDs to Paths. If False, returns a unified pd.DataFrame.
        api_key: Optional API key for services requiring authentication (e.g. TAHMO).
        api_secret: Optional API secret for services requiring authentication (e.g. TAHMO).
        **kwargs: Additional parameters forwarded to the underlying extractor.

    Returns:
        Union[pd.DataFrame, Dict[str, Path]]: Pandas DataFrame if save_parquet=False,
            or dictionary mapping station codes to saved Parquet paths if save_parquet=True.
    """
    s_box: SpatialBoundingBox = AFRICA_BBOX
    if bbox is not None:
        if isinstance(bbox, SpatialBoundingBox):
            s_box = bbox
        elif isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) == 4:
            s_box = SpatialBoundingBox(
                lat_min=float(bbox[0]),
                lon_min=float(bbox[1]),
                lat_max=float(bbox[2]),
                lon_max=float(bbox[3]),
            )
        else:
            raise ValueError(
                f"Invalid bbox format: {bbox}. Expected SpatialBoundingBox or [lat_min, lon_min, lat_max, lon_max]."
            )

    src = source.lower().strip()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if src == "tahmo":
        from .extract.apis.tahmo import TAHMOExtractor
        extractor = TAHMOExtractor(
            output_dir=out_dir,
            api_key=api_key,
            api_secret=api_secret,
            **kwargs,
        )
        res = extractor.extract(
            start_date=start_date,
            end_date=end_date,
            bbox=s_box,
            variables=variables,
            save_parquet=save_parquet,
        )
        if save_parquet:
            return {str(k): Path(v) for k, v in res.items()}
        else:
            if not res:
                return pd.DataFrame()
            dfs = []
            for code, df_st in res.items():
                if isinstance(df_st, pd.DataFrame) and not df_st.empty:
                    df_copy = df_st.copy()
                    if "station_id" not in df_copy.columns:
                        df_copy["station_id"] = code
                    dfs.append(df_copy)
            return pd.concat(dfs, axis=0) if dfs else pd.DataFrame()

    elif src in ("ghcn", "ghcnd", "ghcn_daily", "ghcn_hourly"):
        from .extract.apis.ghcnd import GHCNExtractor
        res_type = "hourly" if "hour" in src else kwargs.pop("resolution", "daily")
        extractor = GHCNExtractor(output_dir=out_dir, resolution=res_type, **kwargs)
        res = extractor.extract(
            start_date=start_date,
            end_date=end_date,
            bbox=s_box,
            variables=variables,
            save_parquet=save_parquet,
            **kwargs,
        )
        if save_parquet:
            if isinstance(res, (str, Path)):
                return {f"ghcn_{res_type}": Path(res)}
            elif isinstance(res, dict):
                return {str(k): Path(v) for k, v in res.items()}
            return {"ghcn": Path(res)}
        else:
            return res if isinstance(res, pd.DataFrame) else pd.read_parquet(res)

    elif src in ("cds_insitu", "cds", "insitu"):
        from .extract.apis.cds_insitu import CDSInSituExtractor
        extractor = CDSInSituExtractor(output_dir=out_dir, **kwargs)
        paths = extractor.extract(
            start_date=start_date,
            end_date=end_date,
            bbox=s_box,
            variables=variables,
            **kwargs,
        )
        if save_parquet:
            return {f"cds_insitu_{p.stem}": Path(p) for p in paths}
        else:
            dfs = [pd.read_parquet(p) for p in paths if Path(p).exists()]
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    else:
        raise ValueError(
            f"Unsupported station data source: '{source}'. "
            f"Supported sources are: 'tahmo', 'ghcn', 'ghcnd', 'cds_insitu'."
        )


def pull_station_data(
    source: str = "tahmo",
    start_date: Union[str, datetime, date, pd.Timestamp] = "2023-01-01",
    end_date: Union[str, datetime, date, pd.Timestamp] = "2023-01-05",
    bbox: Optional[Union[SpatialBoundingBox, Sequence[float]]] = None,
    variables: Optional[Sequence[str]] = None,
    output_dir: Union[str, Path] = "./mashariki_data/obs",
    save_parquet: bool = True,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    **kwargs: Any,
) -> Union[pd.DataFrame, Dict[str, Path]]:
    """
    Pull in-situ ground station observations across Africa.
    Delegates to extract_stations.
    """
    return extract_stations(
        source=source,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        variables=variables,
        output_dir=output_dir,
        save_parquet=save_parquet,
        api_key=api_key,
        api_secret=api_secret,
        **kwargs,
    )


def extract_gridded(
    source: str = "era5",
    start_date: Union[str, datetime, date, pd.Timestamp] = "2023-01-01",
    end_date: Union[str, datetime, date, pd.Timestamp] = "2023-01-05",
    bbox: Optional[Union[SpatialBoundingBox, Sequence[float]]] = None,
    variables: Optional[Sequence[str]] = None,
    levels: Optional[Sequence[int]] = None,
    output_dir: Union[str, Path] = "./mashariki_data/gridded",
    save_zarr: bool = True,
    use_legacy: bool = True,
    use_cds_arco: bool = False,
    use_google_arco: bool = False,
    data_format: str = "netcdf4",
    use_arco: Optional[bool] = None,
    force_cds: Optional[bool] = None,
    chunk_days: int = 5,
    chunking: Optional[str] = None,
    stream: Optional[str] = None,
    cds_key: Optional[str] = None,
    cds_url: Optional[str] = None,
    cds_arco_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs: Any,
) -> Union[xr.Dataset, Path]:
    """
    Extract gridded atmospheric reanalysis, satellite, and numerical model fields for Africa.

    Supported sources:
        - "era5": ECMWF ERA5 reanalysis (prioritizes Google Cloud ARCO, falls back to Copernicus CDS).
        - "imerg": NASA GPM IMERG satellite precipitation via non-interactive earthaccess.
        - "cams": Copernicus Atmosphere Monitoring Service reanalysis.
        - "merra2": NASA MERRA-2 atmospheric reanalysis.
        - "open_meteo" / "openmeteo": Open-Meteo historical reanalysis API.
        - "nasa_power" / "power": NASA POWER solar and meteorological dataset.

    Mathematical docstring:
        ERA5 2D surface tensor shape: (T_time, Lat, Lon) per variable.
        ERA5 3D pressure-level tensor shape: (T_time, Levels, Lat, Lon) per variable.
        IMERG precipitationCal tensor shape: (T_time, Lat, Lon) in mm/hr.

    Args:
        source: Gridded atmospheric or satellite data product.
        start_date: Initial timestamp of extraction window.
        end_date: Terminal timestamp of extraction window.
        bbox: Spatial boundary as SpatialBoundingBox or sequence [lat_min, lon_min, lat_max, lon_max].
        variables: Optional sequence of physical variable identifiers.
        levels: Optional sequence of atmospheric pressure levels in hPa (for 3D ERA5 variables).
        output_dir: Local filesystem destination directory for Zarr storage.
        save_zarr: If True, writes consolidated Zarr store to disk and returns Path;
            if False, returns in-memory or lazy xr.Dataset.
        use_arco: If True, prioritizes Google Cloud ARCO ERA5 cloud store over CDS.
        force_cds: If True, bypasses ARCO and queries Copernicus CDS API directly.
        chunk_days: Temporal chunk window size in days for CDS queries.
        cds_key: Optional Copernicus CDS API key.
        cds_url: Optional Copernicus CDS API URL.
        username: Optional NASA Earthdata username.
        password: Optional NASA Earthdata password.
        **kwargs: Additional parameters forwarded to the underlying extractor.

    Returns:
        Union[xr.Dataset, Path]: xr.Dataset if save_zarr=False, or Path to Zarr group if save_zarr=True.
    """
    s_box: SpatialBoundingBox = AFRICA_BBOX
    if bbox is not None:
        if isinstance(bbox, SpatialBoundingBox):
            s_box = bbox
        elif isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) == 4:
            s_box = SpatialBoundingBox(
                lat_min=float(bbox[0]),
                lon_min=float(bbox[1]),
                lat_max=float(bbox[2]),
                lon_max=float(bbox[3]),
            )
        else:
            raise ValueError(
                f"Invalid bbox format: {bbox}. Expected SpatialBoundingBox or [lat_min, lon_min, lat_max, lon_max]."
            )

    src = source.lower().strip()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if src == "era5":
        from .extract.apis.era5 import ERA5Extractor
        extractor = ERA5Extractor(
            output_dir=out_dir,
            use_legacy=use_legacy,
            use_cds_arco=use_cds_arco,
            use_google_arco=use_google_arco,
            data_format=data_format,
            prefer_arco=use_arco,
            force_cds=force_cds,
            cds_key=cds_key,
            cds_url=cds_url,
            cds_arco_url=cds_arco_url,
            chunking=kwargs.get("chunking", chunking),
            stream=kwargs.get("stream", stream),
            **kwargs,
        )
        res = extractor.extract(
            start_date=start_date,
            end_date=end_date,
            bbox=s_box,
            variables=variables,
            levels=levels,
            save_zarr=save_zarr,
            use_legacy=use_legacy,
            use_cds_arco=use_cds_arco,
            use_google_arco=use_google_arco,
            data_format=data_format,
            force_cds=force_cds,
            prefer_arco=use_arco,
            chunk_days=chunk_days,
            chunking=kwargs.get("chunking", chunking),
            stream=kwargs.get("stream", stream),
            cds_key=cds_key,
            cds_url=cds_url,
            cds_arco_url=cds_arco_url,
        )
        return Path(res) if isinstance(res, (str, Path)) else res

    elif src == "imerg":
        from .extract.apis.imerg import IMERGExtractor
        extractor = IMERGExtractor(
            output_dir=out_dir,
            username=username,
            password=password,
            **kwargs,
        )
        res = extractor.extract(
            start_date=start_date,
            end_date=end_date,
            bbox=s_box,
            variables=variables,
            convert_to_zarr=save_zarr,
            username=username,
            password=password,
            **kwargs,
        )
        if save_zarr:
            if res.zarr_stores:
                return Path(res.zarr_stores[0])
            return Path(res.zarr_dir or out_dir)
        else:
            return res.open()

    elif src in ("cams", "copernicus_cams", "merra2", "nasa_merra2", "open_meteo", "openmeteo"):
        from .extract.apis.reanalysis import ReanalysisExtractor
        extractor = ReanalysisExtractor(output_dir=out_dir, source=src, **kwargs)
        df_or_path = extractor.extract(
            start_date=start_date,
            end_date=end_date,
            bbox=s_box,
            variables=variables,
            save_parquet=False,
            **kwargs,
        )
        if isinstance(df_or_path, pd.DataFrame):
            ds = df_or_path.to_xarray()
        else:
            df = pd.read_parquet(df_or_path)
            ds = df.to_xarray()
        if save_zarr:
            s_str = pd.to_datetime(start_date).strftime("%Y%m%d")
            e_str = pd.to_datetime(end_date).strftime("%Y%m%d")
            zarr_path = out_dir / f"{src}_{s_str}_{e_str}.zarr"
            ds.to_zarr(zarr_path, mode="w", consolidated=True)
            return zarr_path
        return ds

    elif src in ("nasa_power", "power", "pvgis", "sarah3", "landsaf", "tropomi"):
        from .extract.apis.satellite import SatelliteExtractor
        extractor = SatelliteExtractor(output_dir=out_dir, source=src, **kwargs)
        df_or_path = extractor.extract(
            start_date=start_date,
            end_date=end_date,
            bbox=s_box,
            variables=variables,
            save_parquet=False,
            **kwargs,
        )
        if isinstance(df_or_path, pd.DataFrame):
            ds = df_or_path.to_xarray()
        else:
            df = pd.read_parquet(df_or_path)
            ds = df.to_xarray()
        if save_zarr:
            s_str = pd.to_datetime(start_date).strftime("%Y%m%d")
            e_str = pd.to_datetime(end_date).strftime("%Y%m%d")
            zarr_path = out_dir / f"{src}_{s_str}_{e_str}.zarr"
            ds.to_zarr(zarr_path, mode="w", consolidated=True)
            return zarr_path
        return ds

    else:
        raise ValueError(
            f"Unsupported gridded source: '{source}'. "
            f"Supported sources include: 'era5', 'imerg', 'cams', 'merra2', 'open_meteo', 'nasa_power'."
        )


def pull_gridded_data(
    source: str = "era5",
    start_date: Union[str, datetime, date, pd.Timestamp] = "2023-01-01",
    end_date: Union[str, datetime, date, pd.Timestamp] = "2023-01-05",
    bbox: Optional[Union[SpatialBoundingBox, Sequence[float]]] = None,
    variables: Optional[Sequence[str]] = None,
    levels: Optional[Sequence[int]] = None,
    output_dir: Union[str, Path] = "./mashariki_data/gridded",
    save_zarr: bool = True,
    use_legacy: bool = True,
    use_cds_arco: bool = False,
    use_google_arco: bool = False,
    data_format: str = "netcdf4",
    use_arco: Optional[bool] = None,
    force_cds: Optional[bool] = None,
    chunk_days: int = 5,
    chunking: Optional[str] = None,
    stream: Optional[str] = None,
    cds_key: Optional[str] = None,
    cds_url: Optional[str] = None,
    cds_arco_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs: Any,
) -> Union[xr.Dataset, Path]:
    """
    Pull gridded atmospheric reanalysis, satellite, and numerical model fields for Africa.
    Delegates to extract_gridded.
    """
    return extract_gridded(
        source=source,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        variables=variables,
        levels=levels,
        output_dir=output_dir,
        save_zarr=save_zarr,
        use_legacy=use_legacy,
        use_cds_arco=use_cds_arco,
        use_google_arco=use_google_arco,
        data_format=data_format,
        use_arco=use_arco,
        force_cds=force_cds,
        chunk_days=chunk_days,
        chunking=chunking,
        stream=stream,
        cds_key=cds_key,
        cds_url=cds_url,
        cds_arco_url=cds_arco_url,
        username=username,
        password=password,
        **kwargs,
    )


def pull_weather_data(
    start_date: Union[str, datetime, pd.Timestamp],
    end_date: Union[str, datetime, pd.Timestamp],
    bbox: Optional[Union[SpatialBoundingBox, Sequence[float]]] = None,
    station_source: Optional[Union[str, Sequence[str]]] = "tahmo",
    gridded_source: Optional[Union[str, Sequence[str]]] = "era5",
    station_variables: Optional[Sequence[str]] = None,
    gridded_variables: Optional[Sequence[str]] = None,
    output_dir: Union[str, Path] = "./mashariki_data",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Unified master entrypoint to pull multi-source meteorological observations and
    gridded atmospheric fields across Africa in a single function call.

    Mathematical docstring:
        Executes unified extraction across point sensor networks and 2D/3D gridded models.
        Returns a dictionary mapping source categories to extracted paths or data structures,
        along with an initialized MasharikiWeatherDataset instance when local data exists.

    Args:
        start_date: Initial timestamp for observation and gridded extraction.
        end_date: Terminal timestamp for observation and gridded extraction.
        bbox: Spatial boundary as SpatialBoundingBox or sequence [lat_min, lon_min, lat_max, lon_max].
        station_source: Single station source name or sequence of sources (e.g. "tahmo", ["tahmo", "ghcn"]).
        gridded_source: Single gridded source name or sequence of sources (e.g. "era5", ["era5", "imerg"]).
        station_variables: Optional sequence of variable names for station extraction.
        gridded_variables: Optional sequence of variable names for gridded extraction.
        output_dir: Root storage directory for observational Parquet and gridded Zarr outputs.
        **kwargs: Additional keyword arguments passed to pull_station_data and pull_gridded_data.

    Returns:
        Dict[str, Any]: Results dictionary containing:
            - "stations": Extracted station paths or DataFrames.
            - "gridded": Extracted gridded Zarr paths or Datasets.
            - "dataset": Optional[MasharikiWeatherDataset] loaded from local storage.
    """
    root_p = Path(output_dir)
    obs_root = root_p / "obs"
    gridded_root = root_p / "gridded"
    target_bbox = bbox if bbox is not None else AFRICA_BBOX

    save_parquet_val = kwargs.pop("save_parquet", True)
    save_zarr_val = kwargs.pop("save_zarr", True)

    station_results: Dict[str, Any] = {}
    if station_source is not None:
        st_sources = [station_source] if isinstance(station_source, str) else list(station_source)
        for st_src in st_sources:
            st_out = obs_root / st_src
            try:
                st_data = pull_station_data(
                    source=st_src,
                    start_date=start_date,
                    end_date=end_date,
                    bbox=target_bbox,
                    variables=station_variables,
                    output_dir=st_out,
                    save_parquet=save_parquet_val,
                    **kwargs,
                )
                station_results[st_src] = st_data
            except Exception as e:
                logger.error("Failed pulling station data from %s: %s", st_src, e)
                station_results[st_src] = None

    gridded_results: Dict[str, Any] = {}
    if gridded_source is not None:
        gr_sources = [gridded_source] if isinstance(gridded_source, str) else list(gridded_source)
        for gr_src in gr_sources:
            gr_out = gridded_root / gr_src
            try:
                gr_data = pull_gridded_data(
                    source=gr_src,
                    start_date=start_date,
                    end_date=end_date,
                    bbox=target_bbox,
                    variables=gridded_variables,
                    output_dir=gr_out,
                    save_zarr=save_zarr_val,
                    **kwargs,
                )
                gridded_results[gr_src] = gr_data
            except Exception as e:
                logger.error("Failed pulling gridded data from %s: %s", gr_src, e)
                gridded_results[gr_src] = None

    # Instantiate MasharikiWeatherDataset locally
    ds_inst: Optional[MasharikiWeatherDataset] = None
    try:
        active_sources = [s for s, res in station_results.items() if res is not None]
        if not active_sources and station_source:
            active_sources = [station_source] if isinstance(station_source, str) else list(station_source)
        ds_inst = MasharikiWeatherDataset(
            root_dir=str(root_p),
            source_obs=active_sources or ["tahmo"],
            use_hf=False,
        )
    except Exception as e:
        logger.debug("Automatic MasharikiWeatherDataset initialization skipped: %s", e)

    stations_out = (
        station_results
        if (isinstance(station_source, (list, tuple)) or not station_source)
        else station_results.get(station_source, station_results)
    )
    gridded_out = (
        gridded_results
        if (isinstance(gridded_source, (list, tuple)) or not gridded_source)
        else gridded_results.get(gridded_source, gridded_results)
    )

    return {
        "stations": stations_out,
        "gridded": gridded_out,
        "dataset": ds_inst,
    }
