import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple, Union, Dict, Callable
import numpy as np
import pandas as pd
import xarray as xr
import fsspec
import shutil
import difflib
from tqdm import tqdm
from .rechunker import rechunk_dataset, get_zarr_encoding, DEFAULT_TIME_CHUNKS
import dask
from dask.diagnostics import ProgressBar
from multiprocessing.pool import ThreadPool

FrameArray = Union[pd.DataFrame, np.ndarray]

@dataclass
class Windows:
    """
    A dynamic multi-modal container for sliding window views across N-Grids and Stations natively.
    """
    # 1. Point Stations (Eager Temporal Vectors)
    station_x: Union[pd.DataFrame, np.ndarray, xr.Dataset]
    station_mask_x: Union[pd.DataFrame, np.ndarray, xr.Dataset]
    station_y: Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]] = None
    station_mask_y: Optional[Union[pd.DataFrame, np.ndarray, xr.Dataset]] = None
    
    index_x: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None
    index_y: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None

    # 2. Dynamic N-Grids Mapping (Keys derived dynamically from gridded_url)
    grids_x: Dict[str, Union[np.ndarray, xr.Dataset]] = field(default_factory=dict)
    grids_y: Dict[str, Union[np.ndarray, xr.Dataset]] = field(default_factory=dict)
    grids_features: Dict[str, Sequence[str]] = field(default_factory=dict)
    grids_lats: Dict[str, np.ndarray] = field(default_factory=dict)
    grids_lons: Dict[str, np.ndarray] = field(default_factory=dict)

    # 3. Static Priors (Mapped strictly at native resolutions per-layer)
    station_static: Optional[np.ndarray] = None
    station_static_categorical: Optional[np.ndarray] = None
    native_topography: Optional[np.ndarray] = None
    native_topography_categorical: Optional[np.ndarray] = None
    native_topography_lats: Optional[np.ndarray] = None
    native_topography_lons: Optional[np.ndarray] = None

    station_coords: Optional[np.ndarray] = None  # (N_stations, 2) lat/lon for edge construction
    station_mask_2d: Optional[np.ndarray] = None # Optional physical boolean grid constraints
    
    # 4. Target Routing Engine
    target_node: Optional[str] = None # E.g., 'tahmo', 'era5', or 'imerg' to define the loss output layer.

    # 5. Temporal Engine & TSL Delay Math
    delay: int = 0
    valid_batch_indices: Optional[np.ndarray] = None

    # Mathematical Coordinates
    station_labels: Optional[Sequence[str]] = None
    feature_labels: Optional[Sequence[str]] = None
    static_feature_labels: Optional[Sequence[str]] = None
    static_categorical_labels: Optional[Sequence[str]] = None
    grid_lats: Optional[np.ndarray] = None
    grid_lons: Optional[np.ndarray] = None

    # Scientific Metadata
    variable_types: Dict[str, Literal["PHYSICAL", "PRECIPITATION", "STATIC"]] = field(default_factory=dict)

    # 6. Optional Mass Conservation Targets (CHIRPS/TAMSAT)
    # These are NEVER used as sequential inputs. They exist solely for
    # physics-informed loss constraints in the training loop.
    # Keys are group names (e.g. 'chirps', 'tamsat').
    daily_constraints: Dict[str, np.ndarray] = field(default_factory=dict)
    daily_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    daily_time_index: Optional[np.ndarray] = None  # Maps hourly steps -> day index

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

def df_add_missing_columns(df: pd.DataFrame, col0, col1, fill_value=np.nan) -> pd.DataFrame:
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
    providing seamless access to historical weather observations and 
    topographical data for the East Africa region.
    """ 
    __version__ = "0.1.9"
    
    # Hardcoded Test Set Split Anchor
    test_set_start = pd.Timestamp("2023-01-01", tz="UTC")

    VALID_GRIDDED_GROUPS = ["era5", "imerg", "chirps", "tahmo", "topography", "tamsat"]

    def _standardize_coordinates(self, ds: Optional[xr.Dataset]) -> Optional[xr.Dataset]:
        """
        Standardizes coordinate orientation across disparate datasets (e.g. CHIRPS vs IMERG).
        Ensures that latitudes and longitudes are strictly ascending,
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

    def _validate_groups(self, groups: List[str]):
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
        import warnings
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
        root_dir: str = "./mashariki_data",
        years: Optional[List[int]] = None,
        source_obs: Union[str, List[str]] = "tahmo",
        freq: str = "h",
        test_set_start: Optional[pd.Timestamp] = None,
        parameters: Optional[Union[str, Sequence[str]]] = None,
        imputation_method: Optional[Union[str, Callable[[pd.DataFrame], pd.DataFrame]]] = None,
        imputation_limit: Optional[int] = None,
        pad_missing_values: bool = True,
        aggregation_methods: Optional[Dict[str, str]] = None,
        tabular_parquet_path: Optional[str] = None,
        rechunk: bool = False,
        swap: bool = False
    ):
        """
        Args:
            repo_id (str): Hugging Face repository ID.
            token (str): Hugging Face authentication token.
            root_dir (str): Local directory to store the dataset.
            years (List[int]): Years to include in the dataset.
            source_obs (Union[str, List[str]]): Source of observations.
            freq (str): Temporal frequency of the data.
            test_set_start (pd.Timestamp): Start date of the test set.
            parameters (Union[str, Sequence[str]]): Parameters to include in the dataset.
            imputation_method (Optional[Union[str, Callable[[pd.DataFrame], pd.DataFrame]]]): Method to impute missing values, 
                "locf": Forward fill with strict limit to prevent weeks of stale data
                "temporal_linear": Linear interpolation for continuous variables (limit 3 hours)
                "diurnal_climatology": Fill missing with the historical median for that Month & Hour
                "zero": Fill missing with 0 (Dangerous for precipitation targets, but acceptable for inputs if masked)
                None: Leave NaNs intact (Best for PyTorch loss masking)
                Callable[[pd.DataFrame], pd.DataFrame]: Custom imputation function
            pad_missing_values (bool): Whether to pad missing values.
            aggregation_methods (Optional[Dict[str, str]]): Methods to aggregate data.
            tabular_parquet_path (Optional[str]): Path to tabular parquet file.
        """
        self.repo_id = repo_id
        self.token = token
        self.root_dir = root_dir
        
        # Ensure array format for Multi-Rate Fusion (Different temporal resolutions)
        self.source_obs = [source_obs] if isinstance(source_obs, str) else source_obs
        
        self.pad_missing_values = pad_missing_values
        self.years = years if years else [2018, 2019, 2020, 2021, 2022, 2023, 2024]
        self.aggregation_methods = aggregation_methods or {}
        self.tabular_parquet_path = tabular_parquet_path
        if test_set_start is not None:
            self.test_set_start = test_set_start
        self.requested_parameters = parameters
        self.imputation_method = imputation_method
        self.imputation_limit = imputation_limit
        self.rechunk = rechunk
        self.swap = swap
        
        try:
            self.freq = pd.tseries.frequencies.to_offset(freq).freqstr
        except ValueError:
            raise ValueError(f"Value '{freq}' is not a valid pandas frequency string.")
        
        self.fs = fsspec.filesystem("hf", token=token)
        self._obs_zarr = {}
        self._topo_ds = None
        
        # Local cache control flow
        if self._is_local_cache_complete():
            # Bypass Hugging Face entirely
            self._load_local_registries()
        else:
            # Connect to Hugging Face to fetch missing metadata/years
            for src in self.source_obs:
                os.makedirs(f"{self.root_dir}/obs/{src}", exist_ok=True)
                ds = self._open_zipped_zarr(f"data/obs/{src}.zarr.zip", lazy=True)
                if ds is None:
                    raise ConnectionError(f"Failed to connect to the observation dataset: {src}")
                self._obs_zarr[src] = ds
            self._load_registries()
        
        if self.tabular_parquet_path and os.path.exists(self.tabular_parquet_path):
            self._load_cached_parquet()
        else:
            self._prepare_observations()
            
        self.interpolate_topography()

    # ---DATA INSPECTION PROPERTIES ---
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
    def stations(self) -> pd.Index:
        return self.stations_table.index

    @property
    def parameters(self) -> pd.Index:
        return self.parameters_table.index

    @property
    def missing_values(self) -> pd.Series:
        valid_columns = self.mask.loc[:, self.mask.any()]
        mean_missing = {}
        for param in self.parameters:
            if param in valid_columns.columns.get_level_values(1):
                mean_missing[param] = 1 - valid_columns.xs(param, axis=1, level=1).values.mean()
            else:
                mean_missing[param] = 1.0 
                
        out = pd.Series(mean_missing, name="missing_perc", dtype="float32")
        return out.reindex(self.parameters)

    def _is_local_cache_complete(self) -> bool:
        # Check if metadata registries exist locally
        if not os.path.exists(f"{self.root_dir}/stations_registry.parquet"):
            return False
        if not os.path.exists(f"{self.root_dir}/parameters_registry.parquet"):
            return False
            
        # Check if all requested years for all networks exist
        for src in self.source_obs:
            for year in self.years:
                p_path = f"{self.root_dir}/obs/{src}/{year}.parquet"
                if not os.path.exists(p_path):
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
        # Check both {root_dir}/obs/tahmo.zarr.zip and {root_dir}/data/obs/tahmo.zarr.zip
        local_path_1 = f"{self.root_dir}/{rel_path.replace('data/', '')}"
        local_path_2 = f"{self.root_dir}/{rel_path}"
        
        local_zip = None
        if os.path.exists(local_path_1):
            local_zip = local_path_1
        elif os.path.exists(local_path_2):
            local_zip = local_path_2
            
        if local_zip:
            # Create fsspec mapper using local protocol chaining
            zip_mapper = fsspec.get_mapper(f"zip://::file://{os.path.abspath(local_zip)}")
            chunks = {} if lazy else None
            try:
                ds = xr.open_zarr(zip_mapper, chunks=chunks, consolidated=True, decode_timedelta=False, zarr_format=2)
            except Exception:
                ds = xr.open_zarr(zip_mapper, chunks=chunks, consolidated=False, decode_timedelta=False, zarr_format=2)
            return self._standardize_coordinates(ds)

        # 2. Fallback to Hugging Face Remote Streaming
        full_path = f"hf://datasets/{self.repo_id}/{rel_path}"
        try:
            # Safety Check: Verify file exists and token is valid
            if not self.fs.exists(full_path):
                print(f"Authentication Error: Cannot find {full_path}.")
                print("Check that your HF Token is correct and has access to the private repository.")
                return None
                
            # Explicitly open the remote file object first using the authenticated filesystem
            remote_file = self.fs.open(full_path)
            
            # Pass the active file object to the zip mapper
            zip_mapper = fsspec.get_mapper("zip://", fo=remote_file)
            
            chunks = {} if lazy else None
            
            try:
                ds = xr.open_zarr(zip_mapper, chunks=chunks, consolidated=True, decode_timedelta=False, zarr_format=2)
            except Exception:
                ds = xr.open_zarr(zip_mapper, chunks=chunks, consolidated=False, decode_timedelta=False, zarr_format=2)
            return self._standardize_coordinates(ds)
            
        except Exception as e:
            print(f"Error: Could not open {rel_path}. {e}")
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
                rechunk_dataset(local_path, optimized_path, time_chunks=24, swap=self.swap)
                return self._open_gridded_zarr(group, year, lazy) # Recurse once to return optimized handle
                
            chunks = {} if lazy else None
            try:
                ds = xr.open_zarr(local_path, chunks=chunks, 
                                  consolidated=True, decode_timedelta=False, zarr_format=2)
            except Exception:
                ds = xr.open_zarr(local_path, chunks=chunks, 
                                  consolidated=False, decode_timedelta=False, zarr_format=2)
            return self._standardize_coordinates(ds)
                
        # 3. Fallback to Hugging Face lazy streaming pipeline
        return self._open_zipped_zarr(f"data/{group}/{year}.zarr.zip", lazy)
        
    def download_gridded_datasets(self, groups: List[str], years: Optional[Union[int, List[int]]] = None, 
                                 force: bool = False, optimize: bool = True, swap: Optional[bool] = None,
                                 num_workers: int = 16):
        """
        High-Performance Single-Pass Ingest: Downloads Hugging Face Zarr arrays 
        and streams them into training-optimized Daily (24-step) format in one pass.
        
        Args:
            num_workers (int): Number of parallel HTTP streams for the download.
        """
        self._validate_groups(groups)
        target_years = [years] if isinstance(years, int) else (years or self.years)
        target_years = self._validate_years(target_years)
        # Use class preference if caller doesn't specify swap
        should_swap = swap if swap is not None else self.swap
        
        if not force:
            print(f"WARNING: You are about to statically download dense Zarr arrays for {len(groups)} groups across {len(target_years)} years.")
            print(f"This operation will use {num_workers} parallel streams to saturate your connection.")
            user_input = input("Are you absolutely sure you want to proceed with the native C-dump? [y/N]: ")
            if user_input.lower() not in ['y', 'yes', '']:
                print("Download securely aborted by user.")
                return
        
        for group in groups:
            for year in target_years:
                local_path = f"{self.root_dir}/gridded/{group}/{year}.zarr"
                optimized_path = f"{self.root_dir}/optimized/{group}/{year}.zarr"
                
                # Check for existing data based on tier
                if os.path.exists(optimized_path):
                    print(f"Skipping {group} {year}: Optimized tier already present.")
                    continue
                if not optimize and os.path.exists(local_path):
                    print(f"Skipping {group} {year}: Raw tier already present.")
                    continue
                    
                target_write_path = optimized_path if optimize else local_path
                temp_target = f"{target_write_path}.tmp"
                
                print(f"Streaming {group} {year} directly into {'Optimized' if optimize else 'Raw'} format...")
                
                # Open remote lazy stream
                ds = self._open_zipped_zarr(f"data/{group}/{year}.zarr.zip", lazy=True)
                if ds is not None:
                    os.makedirs(os.path.dirname(target_write_path), exist_ok=True)
                    
                    if os.path.exists(temp_target):
                        shutil.rmtree(temp_target)

                    # 1. Apply Daily Re-chunking lazily IF optimizing
                    if optimize:
                        # Dimension-agnostic: chunk 'time' at step 24; collocate everything else
                        chunks_dict = {
                            dim: (DEFAULT_TIME_CHUNKS if dim == 'time' else -1) 
                            for dim in ds.dims
                        }
                        ds = ds.chunk(chunks_dict)
                    
                    # 2. Get high-performance encoding settings
                    # (Uses Daily-aligned chunks even for raw if we're feeling bold, 
                    # but here we follow the optimize flag).
                    encoding = get_zarr_encoding(ds, DEFAULT_TIME_CHUNKS if optimize else 1)

                    # 3. Trigger Parallel Single-Pass Write
                    with dask.config.set(scheduler='threads', num_workers=num_workers):
                        with ProgressBar():
                            ds.to_zarr(temp_target, mode='w', encoding=encoding, consolidated=True, compute=True, zarr_format=2)
                    
                    # 4. Atomic Swap
                    if os.path.exists(target_write_path):
                        shutil.rmtree(target_write_path)
                    os.rename(temp_target, target_write_path)
                    
                    print(f"Successfully ingested {group} {year} at {target_write_path}!")
                    
                    # Optional: Cleanup raw if we just optimized it from HF directly (not needed here 
                    # as we wrote to a single file, but logic preserved for 'swap' users).
                    if optimize and should_swap and os.path.exists(local_path):
                        shutil.rmtree(local_path)

    def _load_registries(self):
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

    def _load_local_registries(self):
        # Load metadata instantly from local disk
        self.stations_table = pd.read_parquet(f"{self.root_dir}/stations_registry.parquet")
        self.parameters_table = pd.read_parquet(f"{self.root_dir}/parameters_registry.parquet")
        self._filter_requested_parameters()

    def _filter_requested_parameters(self):
        if self.requested_parameters is not None:
            params = [self.requested_parameters] if isinstance(self.requested_parameters, str) else self.requested_parameters
            valid_params = [p for p in params if p in self.parameters_table.index]
            if not valid_params:
                raise ValueError(f"Requested parameters {params} not found in the dataset.")
            self.parameters_table = self.parameters_table.loc[valid_params]

    def _build_aggregation_dict(self, multi_index_columns) -> Dict[Tuple, str]:
        agg_dict = {}
        for col in multi_index_columns:
            param = col[1].lower()
            if param in self.aggregation_methods:
                agg_dict[col] = self.aggregation_methods[param]
            elif any(kw in param for kw in ['precip', 'rain', 'tp', 'pr', 'total_precipitation']): # mass conservation variables e.g precipitation
                agg_dict[col] = 'sum' # state variables e.g temperature
            elif 'gust' in param:
                agg_dict[col] = 'max'
            else:
                agg_dict[col] = 'mean'
        return agg_dict

    def _prepare_observations(self):
        """
        Prepare observations by loading data from Zarr files and creating a pandas DataFrame.
        """
        # Iterate all sources, triggering Outer Join during pivot
        yearly_dfs = []
        for src in self.source_obs:
            for year in self.years:
                p_path = f"{self.root_dir}/obs/{src}/{year}.parquet"
                if not os.path.exists(p_path):
                    try:
                        yearly_ds = self._obs_zarr[src].sel(time=str(year)).load()
                        df_year = yearly_ds.to_dataframe().reset_index()
                        df_year.to_parquet(p_path, compression='snappy', index=False)
                    except KeyError:
                        print(f"Warning: Year {year} not found in {src}. Skipping.")
                        continue
                if os.path.exists(p_path):
                    yearly_dfs.append(pd.read_parquet(p_path))
            
        full_df = pd.concat(yearly_dfs)
        full_df['time'] = pd.to_datetime(full_df['time'])
        
        if 'station' in full_df.columns:
            full_df = full_df.rename(columns={'station': 'station_id'})

        param_cols = [c for c in full_df.columns if c not in ['time', 'station_id']]
        obs_df = full_df.pivot(index='time', columns='station_id', values=param_cols)
        obs_df = obs_df.reorder_levels([1, 0], axis=1).sort_index(axis=1)
        obs_df.columns.names = ['station_id', 'parameter']

        valid_cols = obs_df.columns.intersection(
            pd.MultiIndex.from_product([obs_df.columns.levels[0], self.parameters])
        )
        obs_df = obs_df.loc[:, valid_cols]

        if self.freq:
            agg_dict = self._build_aggregation_dict(obs_df.columns)
            
            # Vectorized sparse-safe resampling
            resampled_dfs = []
            res_obj = obs_df.resample(self.freq)
            
            methods = {}
            for col, m in agg_dict.items():
                methods.setdefault(m, []).append(col)
                
            for method, cols in methods.items():
                if method == 'sum':
                    # min_count=1 strictly forces empty bins (GHCNd at non-midnight) to remain NaN
                    resampled_dfs.append(res_obj[cols].sum(min_count=1))
                elif method == 'max':
                    resampled_dfs.append(res_obj[cols].max())
                else:
                    resampled_dfs.append(res_obj[cols].mean())
                    
            obs_df = pd.concat(resampled_dfs, axis=1).sort_index(axis=1)

        if self.pad_missing_values:
            obs_df = df_add_missing_columns(obs_df, self.stations, self.parameters)

        # Ensure mask is saved BEFORE any imputation happens
        self.mask = ~obs_df.isna()
        
        if self.imputation_method is None:
            pass # Leave NaNs intact (Best for PyTorch loss masking)
            
        elif callable(self.imputation_method):
            # Execute User-Defined Custom Function
            obs_df = self.imputation_method(obs_df)
            
        elif isinstance(self.imputation_method, str):
            method = self.imputation_method.lower()
            
            if method == "locf":
                # Forward fill with strict limit to prevent weeks of stale data
                limit = self.imputation_limit if self.imputation_limit is not None else 6
                obs_df = obs_df.ffill(limit=limit)
                
            elif method == "temporal_linear":
                # Linear interpolation for continuous variables (limit 3 hours)
                # Note: PyTorch/Pandas struggles to linear-interpolate MultiIndexes easily,
                # so we apply it down the time axis.
                limit = self.imputation_limit if self.imputation_limit is not None else 3
                obs_df = obs_df.interpolate(method='time', limit=limit)
                
            elif method == "diurnal_climatology":
                # Fill missing with the historical median for that Month & Hour
                obs_df = obs_df.groupby([obs_df.index.month, obs_df.index.hour]).transform(
                    lambda x: x.fillna(x.median())
                )
                
            elif method == "zero":
                # DANGEROUS for precipitation targets, but acceptable for inputs if masked
                obs_df = obs_df.fillna(0)
                
            else:
                raise ValueError(f"Unknown imputation method: {method}")
                
        self.observations = obs_df

    def load_topography(self, filename: str = "east_africa_static_priors") -> Optional[xr.Dataset]:
        """
        Load static topographic priors from disk or Hugging Face.

        Applies cyclical encoding to Aspect (degrees -> sin/cos) to prevent
        the circular paradox where 359 deg and 1 deg appear maximally distant
        to a neural network, when physically they are adjacent.

        Args:
            filename (str): Name of the topography file (without extension).

        Returns:
            Optional[xr.Dataset]: Topographic dataset with Aspect decomposed.
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
                print(f"Warning: Local topography cache corrupted. Re-downloading... ({e})")

        # First-Time Fetch (Download to Cache)
        if ds is None:
            try:
                os.makedirs(local_dir, exist_ok=True)
                self.fs.get(hf_path, local_path)
                ds = xr.open_dataset(local_path, engine="h5netcdf").load()
            except Exception as e:
                print(f"Warning: Could not fetch topography data from Hugging Face. {e}")
                return None

        # Cyclical Encoding: ALL Aspect variables (degrees) -> sin/cos pairs
        # This maps circular angular variables onto a 2D unit circle so that
        # 359 deg and 1 deg are correctly adjacent in feature space.
        # Handles: 'Aspect', 'ASPECT_2000M_SIGRATIO1', 'ASPECT_10000M_SIGRATIO1', etc.
        aspect_vars = [
            v for v in list(ds.data_vars)
            if 'aspect' in v.lower() and '_sin' not in v and '_cos' not in v
        ]
        for var_name in aspect_vars:
            # convert to radians
            aspect_rad = ds[var_name] * (np.pi / 180.0)
            
            # compute sin and cos
            ds[f'{var_name}_sin'] = np.sin(aspect_rad)
            ds[f'{var_name}_cos'] = np.cos(aspect_rad)
            
            # drop the original variable
            ds = ds.drop_vars(var_name)

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

        # Strictly Nearest Neighbor for Categorical Variables
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
        Validate that the requested stations are available in the dataset.
        If stations is None, returns all available stations.

        Args:
            stations (Optional[Union[str, List[str]]]): List of station IDs.

        Returns:
            List[str]: List of valid station IDs.
        """
        if stations is None or stations == "all":
            return list(self.stations_table.index)
            
        if isinstance(stations, str):
            stations = [stations]
            
        available = self.stations_table.index
        valid = [s for s in stations if s in available]
        
        if len(valid) < len(stations):
            missing = set(stations) - set(valid)
            print(f"Warning: {len(missing)} requested stations are not in the data source.")
            
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
        entire East Africa domain, not clustered in one corner.

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
            cluster_stations = station_ids[cluster_mask]
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
    ) -> Dict[str, Literal["PHYSICAL", "PRECIPITATION", "STATIC"]]:
        """
        Classifies variables into physical (Gaussian), precipitation (Tweedie), 
        or static (Topography) based on standard meteorological nomenclature.
        
        Args:
            variables: List of variable names to classify.
            
        Returns:
            Dictionary mapping variable name to its scientific type.
        """
        types = {}
        for var in variables:
            v_lower = var.lower()
            # 1. Precipitation (Non-Gaussian, zero-inflated)
            if any(kw in v_lower for kw in ['precip', 'rain', 'tp', 'pr', 'convective', 'snow']):
                types[var] = "PRECIPITATION"
            # 2. Static / Categorical variables
            elif any(kw in v_lower for kw in ['dem', 'slope', 'aspect', 'mask', 'soil', 'veg', 'topo']):
                types[var] = "STATIC"
            # 3. Physical / Thermodynamic variables (Gaussian-like)
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
                - First Dict: Contains strictly native temporal resolution DataFrames for each group.
                - Second Dict: Contains matched Boolean mask DataFrames indicating structurally missing coordinates (1 for missing).
        """
        
        self._validate_groups(groups)
        valid_stations = self.validate_stations(stations)
        if not valid_stations:
            return {}, {}
            
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
            
        return extracted_dfs, mask_dfs

    @staticmethod
    def spatial_sel(ds: xr.Dataset, lat_slice: Optional[slice] = None, lon_slice: Optional[slice] = None) -> xr.Dataset:
        """
        Order-agnostic spatial selection utility. Auto-corrects flipped slices 
        (e.g., max to min) and emits a gentle warning, proceeding without failure.
        """
        import warnings
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
    ) -> Union[FrameArray, Tuple[FrameArray, FrameArray]]:
        """
        Get observations for the requested stations.
        
        Args:
            stations: List of stations to get observations for. If None, all available stations are used.
            parameters: List of parameters to get observations for.
            first_date: First date to get observations for.
            last_date: Last date to get observations for.
            return_mask: Whether to return a mask of the observations.
            as_numpy: Whether to return the observations as numpy arrays.
            split: Split to use for the observations.
            copy: Whether to return a copy of the observations.
            
        Returns:
            Dictionary of observations.
        """
        
        valid_stations = self.validate_stations(stations)
        if not valid_stations:
            return (np.empty((0,0,0)), np.empty((0,0,0))) if return_mask else np.empty((0,0,0))

        stations_loc = slice(None) if stations is None else valid_stations
        parameters_loc = slice(None) if parameters is None else ([parameters] if isinstance(parameters, str) else parameters)

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

        if self.observations.empty:
            out_obs = np.empty((0,0,0)) if as_numpy else pd.DataFrame()
            out_mask = np.empty((0,0,0), dtype=bool) if as_numpy else pd.DataFrame()
            return (out_obs, out_mask) if return_mask else out_obs

        obs_slice = self.observations.loc[index_loc, (stations_loc, parameters_loc)]
        mask_slice = self.mask.loc[index_loc, (stations_loc, parameters_loc)]

        obs_out = self._export_df(obs_slice, as_numpy=as_numpy, copy=copy)
        mask_out = self._export_df(mask_slice, as_numpy=as_numpy, copy=copy)

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
    ) -> Windows:
        """
        Optional: Extract daily precipitation totals from CHIRPS/TAMSAT and
        attach them to an existing Windows object as mass conservation targets.

        This method does NOT modify the sequential inputs (station_x, grids_x).
        It populates windows.daily_constraints and windows.daily_masks exclusively.

        Only groups in ["chirps", "tamsat"] are accepted. All others raise ValueError.

        Mathematical justification:
            The mass conservation constraint enforces:
                sum_{t in day} P_hat(t) = P_daily
            where P_hat(t) are the hourly predictions and P_daily is the
            observed daily total from CHIRPS or TAMSAT. This is evaluated
            via Tweedie Deviance Loss in the training loop.

        Args:
            windows: An existing Windows object from get_windows().
            groups: Daily precipitation products to extract. Defaults to ["chirps"].
            years: Years to extract. Defaults to self.years.
            stations: Stations to extract for. Defaults to windows.station_labels.
            first_date: Optional temporal lower bound.
            last_date: Optional temporal upper bound.

        Returns:
            The same Windows object with daily_constraints and daily_masks populated.
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
                "Windows will have empty daily_constraints.",
                UserWarning
            )
            return windows

        # Align daily totals to the hourly window temporal extent
        # Use the hourly observation index from the windows to determine day boundaries
        if windows.index_x is not None and len(windows.index_x) > 0:
            # The last timestamp of each window defines its reference time
            hourly_index = pd.DatetimeIndex(windows.index_x[:, -1])
        else:
            warnings.warn(
                "Cannot align daily constraints: windows.index_x is empty.",
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

            # Resample to daily sums (native resolution for CHIRPS/TAMSAT)
            # For products already at daily resolution, this is a no-op identity
            daily_df = df.resample("1D").sum(min_count=1)
            daily_mask = mask_df.resample("1D").max()  # Missing if ANY hour is missing

            # Map each hourly window reference time to its corresponding day
            ref_dates = hourly_index.normalize()  # Floor to day
            unique_days = pd.DatetimeIndex(sorted(ref_dates.unique()))

            # Find overlapping days between the daily product and the window span
            overlap_days = unique_days.intersection(daily_df.index)
            if len(overlap_days) == 0:
                warnings.warn(
                    f"No temporal overlap between {group} daily data and window span. "
                    f"Skipping constraint for {group}.",
                    UserWarning
                )
                continue

            # Build the day_index mapping: for each window sample, which day does it belong to?
            day_to_idx = {d: i for i, d in enumerate(overlap_days)}
            day_index = np.array([day_to_idx.get(d, -1) for d in ref_dates], dtype=np.int64)

            # Filter windows that have valid day mappings
            valid_mask = day_index >= 0

            # Extract the constraint values as numpy arrays
            # Shape: (n_days, n_stations * n_params)
            constraint_vals = daily_df.loc[overlap_days].values.astype(np.float32)
            constraint_mask_vals = daily_mask.loc[overlap_days].values.astype(np.float32)

            # Replace NaN with 0 in constraint values (masked out by the mask anyway)
            constraint_vals = np.nan_to_num(constraint_vals, nan=0.0)

            windows.daily_constraints[group] = constraint_vals
            windows.daily_masks[group] = constraint_mask_vals

        # Build the day_index from the last processed group (shared temporal axis)
        if windows.index_x is not None and len(windows.index_x) > 0:
            hourly_index = pd.DatetimeIndex(windows.index_x[:, -1])
            ref_dates = hourly_index.normalize()
            unique_days = pd.DatetimeIndex(sorted(ref_dates.unique()))
            day_to_idx = {d: i for i, d in enumerate(unique_days)}
            windows.daily_time_index = np.array(
                [day_to_idx.get(d, -1) for d in ref_dates], dtype=np.int64
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

        from numpy.lib.stride_tricks import sliding_window_view
        index = observations.index
        obs_np = self._export_df(observations, as_numpy=True, copy=False)
        mask_np = self._export_df(mask, as_numpy=True, copy=False, fill_value=False)

        x_slice = slice(None) if (horizon_size + delay == 0) else slice(None, len(obs_np) - horizon_size - delay)
        x = sliding_window_view(obs_np[x_slice], window_size, axis=0)[::stride]
        mask_x = sliding_window_view(mask_np[x_slice], window_size, axis=0)[::stride]
        index_x = sliding_window_view(index[x_slice].values, window_size, axis=0)[::stride]
        
        y_start = window_size + delay
        if horizon_size > 0:
            y = sliding_window_view(obs_np[y_start:], horizon_size, axis=0)[::stride]
            mask_y = sliding_window_view(mask_np[y_start:], horizon_size, axis=0)[::stride]
            index_y = sliding_window_view(index[y_start:].values, horizon_size, axis=0)[::stride]
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

        from numpy.lib.stride_tricks import sliding_window_view
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
        if ml_task in ["correction", "downscaling", "anomaly_detection"]:
            # window_size is total context (e.g. 48h). Y is the midpoint.
            mid = window_size // 2
            
            x_slice = slice(None)
            windows.station_x = sliding_window_view(obs_np[x_slice], window_size, axis=0)[::stride]
            windows.station_mask_x = sliding_window_view(mask_np[x_slice], window_size, axis=0)[::stride]
            windows.index_x = sliding_window_view(observations.index.values[x_slice], window_size, axis=0)[::stride]
            
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
            
        # FORECASTING: Strict Causal Windowing [t-tau, t] -> [t+delay, t+delay+horizon]
        else:
            sample_span = window_size + delay + horizon_size
            if len(observations) < sample_span:
                return windows

            x_slice = slice(None, len(obs_np) - horizon_size - delay)
            windows.station_x = sliding_window_view(obs_np[x_slice], window_size, axis=0)[::stride]
            windows.station_mask_x = sliding_window_view(mask_np[x_slice], window_size, axis=0)[::stride]
            windows.index_x = sliding_window_view(observations.index.values[x_slice], window_size, axis=0)[::stride]
            
            y_start = window_size + delay
            y_slice = slice(y_start, None)
            y_is_single_point = False
            
            windows.station_y = sliding_window_view(obs_np[y_slice], horizon_size, axis=0)[::stride]
            windows.station_mask_y = sliding_window_view(mask_np[y_slice], horizon_size, axis=0)[::stride]
            windows.index_y = sliding_window_view(observations.index.values[y_slice], horizon_size, axis=0)[::stride]
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
                                
                                ds = self.spatial_sel(
                                    ds[valid_vars],
                                    lat_slice=slice(max(0, l_min), min(len(lats), l_max + 1)),
                                    lon_slice=slice(max(0, n_min), min(len(lons), n_max + 1))
                                )
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
                 common_times = observations.index
                 for g_name, g_ds in per_grid_datasets.items():
                      if isinstance(g_ds, pd.DataFrame):
                           common_times = common_times.intersection(g_ds.index)
                      else:
                           common_times = common_times.intersection(pd.DatetimeIndex(g_ds.time.values))
                           
                 obs_mask = np.isin(observations.index, common_times)
                 x_idx_mask = np.isin(windows.index_x[:, -1], common_times)
                 if windows.index_y is not None and len(windows.index_y) > 0:
                     x_idx_mask &= np.isin(windows.index_y[:, -1], common_times)
                     
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
                           # Native Scale Preservation and Alignment
                           if group == "imerg":
                               g_ds = g_ds.resample(time="1h").sum(skipna=False)
                           
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