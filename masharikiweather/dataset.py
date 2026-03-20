import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union, Dict, Callable
import numpy as np
import pandas as pd
import xarray as xr
import fsspec
from tqdm import tqdm

FrameArray = Union[pd.DataFrame, np.ndarray]

@dataclass
class Windows:
    """
    A container for sliding window views of the dataset.
    """
    x: Union[pd.DataFrame, np.ndarray, xr.Dataset]
    mask_x: Union[pd.DataFrame, np.ndarray, xr.Dataset]
    y: Union[pd.DataFrame, np.ndarray, xr.Dataset]
    mask_y: Union[pd.DataFrame, np.ndarray, xr.Dataset]
    index_x: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None
    index_y: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None
    
    # ML Gridded Tensors (input resolution)
    gridded_x: Optional[Union[np.ndarray, xr.Dataset]] = None
    gridded_y: Optional[Union[np.ndarray, xr.Dataset]] = None
    gridded_features: Optional[Sequence[str]] = None

    # Target Grid Tensors (for supervised downscaling at a different resolution)
    target_gridded_y: Optional[np.ndarray] = None
    target_gridded_features: Optional[Sequence[str]] = None
    target_grid_lats: Optional[np.ndarray] = None
    target_grid_lons: Optional[np.ndarray] = None
    target_temporal_freq: Optional[str] = None

    # Static Tensors - Continuous float32
    # When target_gridded_url is set: static_grid = target-res, static_grid_input = input-res
    # When no target grid:           static_grid = input-res,  static_grid_input = None
    static_stations: Optional[Union[np.ndarray, xr.Dataset]] = None
    static_stations_categorical: Optional[np.ndarray] = None
    static_grid: Optional[Union[np.ndarray, xr.Dataset]] = None
    static_grid_categorical: Optional[np.ndarray] = None
    static_grid_input: Optional[np.ndarray] = None
    static_grid_input_categorical: Optional[np.ndarray] = None
    station_mask_2d: Optional[np.ndarray] = None

    # Mathematical Coordinates
    station_labels: Optional[Sequence[str]] = None
    feature_labels: Optional[Sequence[str]] = None
    grid_lats: Optional[np.ndarray] = None
    grid_lons: Optional[np.ndarray] = None

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
    __version__ = "0.1.0"
    
    # Hardcoded Test Set Split Anchor
    test_set_start = pd.Timestamp("2023-01-01", tz="UTC")

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
        tabular_parquet_path: Optional[str] = None
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
        Open a zipped Zarr dataset from Hugging Face.
        """
        full_path = f"hf://datasets/{self.repo_id}/{rel_path}"
        try:
            # Safety Check: Verify file exists and token is valid
            if not self.fs.exists(full_path):
                print(f"Authentication Error: Cannot find {full_path}.")
                print("Check that your HF Token is correct and has access to the private repository.")
                return None
                
            # Explicitly open the remote file object first
            remote_file = self.fs.open(full_path)
            
            # Pass the active file object to the zip mapper
            zip_mapper = fsspec.get_mapper("zip://", fo=remote_file)
            
            chunks = {} if lazy else None
            
            try:
                ds = xr.open_dataset(zip_mapper, engine="zarr", chunks=chunks, backend_kwargs={"consolidated": True}, decode_timedelta=False)
            except Exception:
                ds = xr.open_dataset(zip_mapper, engine="zarr", chunks=chunks, backend_kwargs={"consolidated": False}, decode_timedelta=False)
            return ds
            
        except Exception as e:
            print(f"Error: Could not open {rel_path}. {e}")
            return None

    def _open_gridded_zarr(self, group: str, year: int, lazy: bool = True) -> Optional[xr.Dataset]:
        """
        Dynamically fallback to native local disk C-reads if the Zarr directory was 
        previously comprehensively downloaded, masking 100% of HTTP Latency.
        """
        local_path = f"{self.root_dir}/gridded/{group}/{year}.zarr"
        
        if os.path.exists(local_path):
            chunks = {} if lazy else None
            try:
                return xr.open_dataset(local_path, engine="zarr", chunks=chunks, backend_kwargs={"consolidated": True}, decode_timedelta=False)
            except Exception:
                return xr.open_dataset(local_path, engine="zarr", chunks=chunks, backend_kwargs={"consolidated": False}, decode_timedelta=False)
                
        # If not cached locally, fallback to Hugging Face lazy streaming pipeline natively
        return self._open_zipped_zarr(f"data/{group}/{year}.zarr.zip", lazy)
        
    def download_gridded_datasets(self, groups: List[str], years: Optional[Union[int, List[int]]] = None, force: bool = False):
        """
        Force-downloads the 100+ GB Hugging Face Zarr arrays physically onto the local hard drive 
        exactly mirroring the `mashariki_data/obs/` caching behavior for infinite offline ML loading.
        """
        target_years = [years] if isinstance(years, int) else (years or self.years)
        
        if not force:
            print(f"WARNING: You are about to statically download dense Zarr arrays for {len(groups)} groups across {len(target_years)} years.")
            print("This native backup operation could consume anywhere from 10 GB to well over 100 GB of your local SSD space depending on the array sizes.")
            user_input = input("Are you absolutely sure you want to proceed with the native C-dump? [y/N]: ")
            if user_input.lower() not in ['y', 'yes']:
                print("Download securely aborted by user.")
                return
        
        for group in groups:
            for year in target_years:
                local_path = f"{self.root_dir}/gridded/{group}/{year}.zarr"
                if os.path.exists(local_path):
                    print(f"Skipping {group} {year}: Already cached locally.")
                    continue
                    
                print(f"Downloading dense physics for {group} {year} directly to disk (this may take a while)...")
                ds = self._open_zipped_zarr(f"data/{group}/{year}.zarr.zip", lazy=True)
                if ds is not None:
                    os.makedirs(f"{self.root_dir}/gridded/{group}", exist_ok=True)
                    # Trigger massive bulk native file dump converting Zipped HTTP into Native Folder Zarr hierarchy
                    ds.to_zarr(local_path, consolidated=True, compute=True)
                    print(f"Successfully cached {group} {year} at {local_path}!")

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

        # Cyclical Encoding: Aspect (degrees) -> sin/cos pair
        # This maps the circular variable onto a 2D unit circle so that
        # 359 deg and 1 deg are correctly adjacent in feature space.
        if 'Aspect' in ds.data_vars:
            aspect_rad = ds['Aspect'] * (np.pi / 180.0)
            ds['Aspect_sin'] = np.sin(aspect_rad)
            ds['Aspect_cos'] = np.cos(aspect_rad)
            ds = ds.drop_vars('Aspect')

        return ds

    def interpolate_topography(self) -> None:
        """
        Interpolate topography data to the station locations.

        Args:
            filename (str): Name of the topography file.
        """
        if self._topo_ds is None:
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
        
        # Physically sound defaults for ocean/coastal-masked terrain
        # DEM/Slope/TPI = 0 (sea level, flat), Aspect_cos = 1 (undefined direction defaults
        # to unit vector on cos axis), LandCover = 0 (water/ocean class)
        fill_map = {
            'DEM': 0, 'Slope': 0, 'TPI': 0,
            'Aspect_sin': 0, 'Aspect_cos': 1,
            'LandCover': 0,
        }
        for col in df_topo.columns:
            default_val = fill_map.get(col, 0)
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
        if stations is None:
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
        first_date: Optional[Union[str, pd.Timestamp]] = None,
        last_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> Dict[str, pd.DataFrame]:

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
            first_date (Optional[Union[str, pd.Timestamp]]): First date to extract data for.
            last_date (Optional[Union[str, pd.Timestamp]]): Last date to extract data for.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of gridded weather data for each station.
        """
        
        valid_stations = self.validate_stations(stations)
        if not valid_stations:
            return {}

        target_years = [years] if isinstance(years, int) else (years or self.years)
        
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
                    df['parameter_level'] = df.apply(
                        lambda row: f"_{int(row['level'])}" if pd.notnull(row['level']) else "", axis=1
                    )
                    
                    # Dynamically capture variables that were renamed during patch_flat (e.g. _px0, _px1)
                    actual_val_vars = []
                    for col in df.columns:
                        if col in valid_vars:
                            actual_val_vars.append(col)
                        else:
                            for base_var in valid_vars:
                                if col.startswith(f"{base_var}_px"):
                                    actual_val_vars.append(col)
                                    break
                                    
                    # Melt and reconstruct to combine variable name with level
                    id_vars = [c for c in df.columns if c not in actual_val_vars and c != 'parameter_level']
                    df_melt = df.melt(id_vars=id_vars + ['parameter_level'], value_vars=actual_val_vars, var_name='base_param')
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
            
            extracted_dfs[group] = pivoted_df
            
        return extracted_dfs

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
        target_years = [years] if isinstance(years, int) else (years or self.years)
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

        if len(observations) < window_size + horizon_size:
            empty_arr = np.array([])
            return Windows(
                x=empty_arr, mask_x=empty_arr,
                y=empty_arr, mask_y=empty_arr
            )

        index = observations.index
        obs_np = self._export_df(observations, as_numpy=True, copy=False)
        mask_np = self._export_df(mask, as_numpy=True, copy=False, fill_value=False)

        x = sliding_window_view(obs_np[:-horizon_size], window_size)
        mask_x = sliding_window_view(mask_np[:-horizon_size], window_size)
        index_x = sliding_window_view(index[:-horizon_size], window_size)
        
        y = sliding_window_view(obs_np[window_size:], horizon_size)
        mask_y = sliding_window_view(mask_np[window_size:], horizon_size)
        index_y = sliding_window_view(index[window_size:], horizon_size)

        windows = Windows(
            x=x, mask_x=mask_x, y=y, mask_y=mask_y, index_x=index_x, index_y=index_y
        )

        if not as_xarray:
            return windows

        return self.np_windows_as_xr(windows, stations, parameters)

    def get_windows(
        self,
        window_size: int,
        horizon_size: int,
        stations: Optional[Union[str, List[str]]] = None,
        parameters: Optional[Union[str, List[str]]] = None,
        first_date: Optional[Union[str, pd.Timestamp]] = None,
        last_date: Optional[Union[str, pd.Timestamp]] = None,
        split: Optional[Literal["train", "test"]] = None,
        gridded_url: Optional[Union[str, List[str]]] = None,
        gridded_variables: Optional[List[str]] = None,
        method_overrides: Optional[Dict[str, str]] = None,
        patch_radius: int = 1,
        ml_task: Literal["downscaling", "forecasting", "correction"] = "downscaling",
        spatial_mode: Literal["stations", "grid"] = "stations",
        target_gridded_url: Optional[Union[str, List[str]]] = None,
        target_gridded_variables: Optional[List[str]] = None,
        target_temporal_freq: Optional[str] = None,
        as_xarray: bool = False,
    ) -> Windows:
        """
        Get perfectly aligned ML windows combining ground observations and physically corrected gridded data.

        Args:
            window_size (int): Number of past time steps to use as input (x).
            horizon_size (int): Number of future time steps to predict (y).
            stations (Optional[Union[str, List[str]]]): Station(s) to use.
            parameters (Optional[Union[str, List[str]]]): Variables to use.
            first_date (Optional[Union[str, pd.Timestamp]]): Start date for the data.
            last_date (Optional[Union[str, pd.Timestamp]]): End date for the data.
            split (Optional[Literal["train", "test"]]): Split the data into training and testing sets, or None for no split. 
                If split is "train", the data will be split into training and validation sets using the `get_spatial_split` method.
            gridded_url (Optional[Union[str, List[str]]]): HF group name(s) for the input gridded data (e.g. "era5").
            gridded_variables (Optional[List[str]]): Variables to use from the input gridded data.
            method_overrides (Optional[Dict[str, str]]): Method overrides for specific variables.
            patch_radius (int): Radius to use for patch extraction.
            ml_task (Literal["downscaling", "forecasting", "correction"]): Defines the tensor mapping.
                - "downscaling": gridded_x = past input grid, gridded_y = past input grid (same-res).
                - "forecasting": gridded_x = past input grid, gridded_y = None (blocked).
                - "correction": gridded_x = full window (past+future), station obs as sparse anchors.
            spatial_mode (Literal["stations", "grid"]): The spatial array extraction logic.
            target_gridded_url (Optional[Union[str, List[str]]]): HF group name(s) for the target grid
                at a different resolution (e.g. "chirps", "tamsat", "imerg").
            target_gridded_variables (Optional[List[str]]): Variables to extract from the target grid.
            target_temporal_freq (Optional[str]): Temporal frequency of the target grid if different
                from self.freq (e.g. "D" for daily CHIRPS vs hourly ERA5).
            as_xarray (bool): Whether to return the data as xarray.
        """
        if split is not None and split not in ["train", "test"]:
            raise ValueError(f"Invalid split value: {split}. Expected 'train' or 'test'.")

        if ml_task not in ["downscaling", "forecasting", "correction"]:
            raise ValueError("ml_task must be 'downscaling', 'forecasting', or 'correction'.")

        if spatial_mode not in ["stations", "grid"]:
            raise ValueError("spatial_mode must be either 'stations' or 'grid'.")

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

        gridded_df = None
        gridded_2d_np = None
        grid_lats = None
        grid_lons = None
        gridded_2d_features = None

        if gridded_url:
            groups = [gridded_url] if isinstance(gridded_url, str) else gridded_url
            valid_stations = self.validate_stations(stations) if stations else self.stations_table.index.tolist()
            target_years = list(set(observations.index.year)) if not observations.empty else self.years
            
            if spatial_mode == "stations":
                extracted_dfs = self.get_gridded_for_stations(
                    groups=groups, 
                    stations=valid_stations, 
                    years=target_years,
                    variables=gridded_variables,
                    method_overrides=method_overrides,
                    patch_radius=patch_radius,
                    first_date=first_date,
                    last_date=last_date
                )
                
                if extracted_dfs:
                    gridded_df = pd.concat(extracted_dfs.values(), axis=1)
                    
                    # TEMPORAL ALIGNMENT: Inner join ensures we only keep hours where BOTH exist
                    observations, gridded_df = observations.align(gridded_df, join='inner', axis=0)
                    mask, _ = mask.align(gridded_df, join='inner', axis=0)
                    
            elif spatial_mode == "grid":
                coords = self.stations_table.loc[valid_stations].dropna(subset=['latitude', 'longitude'])
                if not coords.empty:
                    min_lat, max_lat = coords['latitude'].min(), coords['latitude'].max()
                    min_lon, max_lon = coords['longitude'].min(), coords['longitude'].max()
                    
                    group_datasets = []
                    for group in groups:
                        yearly_ds_list = []
                        for year in target_years:
                            ds = self._open_gridded_zarr(group, year, lazy=True)
                            if ds is None: continue
                            if first_date or last_date:
                                f_dt = pd.to_datetime(first_date, utc=True).tz_localize(None) if first_date else None
                                l_dt = pd.to_datetime(last_date, utc=True).tz_localize(None) if last_date else None
                                ds = ds.sel(time=slice(f_dt, l_dt))
                            if len(ds.time) == 0: continue
                            
                            requested_vars = gridded_variables if gridded_variables else list(ds.data_vars)
                            valid_vars = [v for v in requested_vars if v in ds.data_vars]
                            if not valid_vars: continue

                            lat_dim = 'lat' if 'lat' in ds.coords else 'latitude'
                            lon_dim = 'lon' if 'lon' in ds.coords else 'longitude'
                            
                            lats = ds[lat_dim].values
                            lons = ds[lon_dim].values
                            lat_idx_1 = np.abs(lats - min_lat).argmin()
                            lat_idx_2 = np.abs(lats - max_lat).argmin()
                            lon_idx_1 = np.abs(lons - min_lon).argmin()
                            lon_idx_2 = np.abs(lons - max_lon).argmin()
                            
                            l_min, l_max = min(lat_idx_1, lat_idx_2), max(lat_idx_1, lat_idx_2)
                            ln_min, ln_max = min(lon_idx_1, lon_idx_2), max(lon_idx_1, lon_idx_2)
                            
                            lat_slice = slice(max(0, l_min - patch_radius), min(len(lats), l_max + patch_radius + 1))
                            lon_slice = slice(max(0, ln_min - patch_radius), min(len(lons), ln_max + patch_radius + 1))
                            
                            ds_sliced = ds[valid_vars].isel({lat_dim: lat_slice, lon_dim: lon_slice}).load()
                            yearly_ds_list.append(ds_sliced)
                            
                        if yearly_ds_list:
                            concat_ds = xr.concat(yearly_ds_list, dim="time", compat="override", coords="minimal")
                            group_datasets.append(concat_ds)
                    
                    if group_datasets:
                        merged_grid = xr.merge(group_datasets, compat="override")
                        
                        common_times = observations.index.intersection(merged_grid.time.values)
                        observations = observations.loc[common_times]
                        mask = mask.loc[common_times]
                        merged_grid = merged_grid.sel(time=common_times)
                        
                        gridded_2d_features = list(merged_grid.data_vars)
                        arr_list = []
                        for var in gridded_2d_features:
                            var_data = merged_grid[var].values
                            if np.issubdtype(var_data.dtype, np.floating):
                                var_data = var_data.astype(np.float32)
                            if var_data.ndim == 3:
                                arr_list.append(var_data[:, np.newaxis, ...])
                        if arr_list:
                            # Concatenate channel features mapping to exact index positions (Time, Channels, Lat, Lon)
                            gridded_2d_np = np.concatenate(arr_list, axis=1)
                            
                        lat_dim = 'lat' if 'lat' in merged_grid.coords else 'latitude'
                        lon_dim = 'lon' if 'lon' in merged_grid.coords else 'longitude'
                        grid_lats = merged_grid[lat_dim].values
                        grid_lons = merged_grid[lon_dim].values

        if len(observations) < window_size + horizon_size:
            empty_arr = np.array([])
            return Windows(x=empty_arr, mask_x=empty_arr, y=empty_arr, mask_y=empty_arr)

        index = observations.index
        obs_np = self._export_df(observations, as_numpy=True, copy=False)
        if np.issubdtype(obs_np.dtype, np.floating): obs_np = obs_np.astype(np.float32, copy=False)
        mask_np = self._export_df(mask, as_numpy=True, copy=False, fill_value=False)

        x = sliding_window_view(obs_np[:-horizon_size], window_size)
        mask_x = sliding_window_view(mask_np[:-horizon_size], window_size)
        index_x = sliding_window_view(index[:-horizon_size], window_size)
        
        y = sliding_window_view(obs_np[window_size:], horizon_size)
        mask_y = sliding_window_view(mask_np[window_size:], horizon_size)
        index_y = sliding_window_view(index[window_size:], horizon_size)

        windows = Windows(
            x=x, mask_x=mask_x, y=y, mask_y=mask_y, index_x=index_x, index_y=index_y
        )
        
        # Tensor Segregation: separate categorical (int64) from continuous (float32)
        cat_cols = ['LandCover']
        valid_stations_subset = self.validate_stations(stations) if stations else self.stations_table.index.tolist()
        all_static = [c for c in self.stations_table.columns if c not in ['latitude', 'longitude', 'network']]
        valid_cat = [c for c in cat_cols if c in all_static]
        valid_cont = [c for c in all_static if c not in valid_cat]

        if spatial_mode == "stations":
            # Continuous static features as float32 (DEM, Slope, Aspect_sin, Aspect_cos, TPI, ...)
            if valid_cont:
                cont_array = self.stations_table.loc[valid_stations_subset, valid_cont].values
                windows.static_stations = cont_array.astype(np.float32, copy=False)
            # Categorical static features as int64 (LandCover)
            if valid_cat:
                cat_array = self.stations_table.loc[valid_stations_subset, valid_cat].values
                windows.static_stations_categorical = cat_array.astype(np.int64, copy=False)

            if gridded_df is not None:
                # Temporal alignment was already enforced by inner join above (line 1281).
                # No reindex needed here; both DataFrames share the same index.
                
                grid_np = self._export_df(gridded_df, as_numpy=True, copy=False)
                if np.issubdtype(grid_np.dtype, np.floating): 
                    grid_np = grid_np.astype(np.float32, copy=False)
                
                # 2. Generate Sliding Windows (These will temporarily contain NaNs)
                windows.gridded_x = sliding_window_view(grid_np[:-horizon_size], window_size)
                windows.gridded_features = gridded_df.columns.unique(level=1).tolist()
                
                if ml_task == "downscaling":
                    windows.gridded_y = sliding_window_view(grid_np[window_size:], horizon_size)
                elif ml_task == "correction":
                    # Correction: gridded_x gets the FULL temporal window (past + future)
                    # so the model can see what ERA5 predicted and learn to correct it
                    windows.gridded_y = sliding_window_view(grid_np[window_size:], horizon_size)
                elif ml_task == "forecasting":
                    windows.gridded_y = None

                # 3. PeakWeather Scientific Compliance: Window-Level Rejection
                # We locate any batch index where a NaN breached the gridded tensors.
                valid_batches = ~np.isnan(windows.gridded_x).any(axis=(1, 2, 3))
                
                if windows.gridded_y is not None:
                    valid_batches &= ~np.isnan(windows.gridded_y).any(axis=(1, 2, 3))
                    
                # 4. Purge Corrupted Sequences
                # Slicing the numpy arrays via the boolean mask guarantees O(1) internal temporal spacing
                windows.x = windows.x[valid_batches]
                windows.y = windows.y[valid_batches]
                windows.mask_x = windows.mask_x[valid_batches]
                windows.mask_y = windows.mask_y[valid_batches]
                windows.index_x = windows.index_x[valid_batches]
                windows.index_y = windows.index_y[valid_batches]
                windows.gridded_x = windows.gridded_x[valid_batches]
                if windows.gridded_y is not None:
                    windows.gridded_y = windows.gridded_y[valid_batches]
                    
        elif spatial_mode == "grid":
            if gridded_2d_np is not None:
                windows.gridded_features = gridded_2d_features
                windows.grid_lats = grid_lats
                windows.grid_lons = grid_lons
                windows.gridded_x = sliding_window_view(gridded_2d_np[:-horizon_size], window_size)
                
                if ml_task == "downscaling":
                    windows.gridded_y = sliding_window_view(gridded_2d_np[window_size:], horizon_size)
                elif ml_task == "correction":
                    windows.gridded_y = sliding_window_view(gridded_2d_np[window_size:], horizon_size)
                elif ml_task == "forecasting":
                    windows.gridded_y = None
                    
                valid_batches = ~np.isnan(windows.gridded_x).any(axis=(1, 2, 3, 4))
                if windows.gridded_y is not None:
                    valid_batches &= ~np.isnan(windows.gridded_y).any(axis=(1, 2, 3, 4))
                    
                windows.x = windows.x[valid_batches]
                windows.y = windows.y[valid_batches]
                windows.mask_x = windows.mask_x[valid_batches]
                windows.mask_y = windows.mask_y[valid_batches]
                windows.index_x = windows.index_x[valid_batches]
                windows.index_y = windows.index_y[valid_batches]
                windows.gridded_x = windows.gridded_x[valid_batches]
                if windows.gridded_y is not None:
                    windows.gridded_y = windows.gridded_y[valid_batches]

                if self._topo_ds is not None and grid_lats is not None and grid_lons is not None:
                    topo_vars = list(self._topo_ds.data_vars)
                    t_lat_dim = 'lat' if 'lat' in self._topo_ds.coords else 'latitude'
                    t_lon_dim = 'lon' if 'lon' in self._topo_ds.coords else 'longitude'
                    
                    glat_xr = xr.DataArray(grid_lats, dims="lat")
                    glon_xr = xr.DataArray(grid_lons, dims="lon")
                    
                    topo_sliced = self._topo_ds.interp({t_lat_dim: glat_xr, t_lon_dim: glon_xr}, method="nearest").load()
                    
                    # Segregate continuous float32 from categorical int64
                    grid_cat = [v for v in topo_vars if v in cat_cols]
                    grid_cont = [v for v in topo_vars if v not in cat_cols]
                    
                    # Continuous static grid (DEM, Slope, Aspect_sin, Aspect_cos, TPI, ...)
                    c_arr_list = []
                    for tv in grid_cont:
                        c_data = topo_sliced[tv].values.astype(np.float32)
                        c_arr_list.append(c_data[np.newaxis, ...])
                    if c_arr_list:
                        windows.static_grid = np.concatenate(c_arr_list, axis=0)  # (Cont_Features, Lat, Lon)
                    
                    # Categorical static grid (LandCover)
                    cat_arr_list = []
                    for tv in grid_cat:
                        cat_data = topo_sliced[tv].values.astype(np.int64)
                        cat_arr_list.append(cat_data[np.newaxis, ...])
                    if cat_arr_list:
                        windows.static_grid_categorical = np.concatenate(cat_arr_list, axis=0)  # (Cat_Features, Lat, Lon)
                
                if grid_lats is not None and grid_lons is not None:
                    mask_2d = np.zeros((len(grid_lats), len(grid_lons)), dtype=bool)
                    coords = self.stations_table.loc[valid_stations_subset].dropna(subset=['latitude', 'longitude'])
                    for _, row in coords.iterrows():
                        lat_idx = np.abs(grid_lats - row['latitude']).argmin()
                        lon_idx = np.abs(grid_lons - row['longitude']).argmin()
                        mask_2d[lat_idx, lon_idx] = True
                    windows.station_mask_2d = mask_2d

            # --- TARGET GRID LOADING (Supervised Downscaling at Different Resolution) ---
            if target_gridded_url and spatial_mode == "grid" and grid_lats is not None:
                target_groups = [target_gridded_url] if isinstance(target_gridded_url, str) else target_gridded_url
                min_lat, max_lat = grid_lats.min(), grid_lats.max()
                min_lon, max_lon = grid_lons.min(), grid_lons.max()
                
                target_ds_list = []
                for tgroup in target_groups:
                    for year in target_years:
                        tds = self._open_gridded_zarr(tgroup, year, lazy=True)
                        if tds is None:
                            continue
                        if first_date or last_date:
                            f_dt = pd.to_datetime(first_date, utc=True).tz_localize(None) if first_date else None
                            l_dt = pd.to_datetime(last_date, utc=True).tz_localize(None) if last_date else None
                            tds = tds.sel(time=slice(f_dt, l_dt))
                        if len(tds.time) == 0:
                            continue
                        
                        t_vars = target_gridded_variables if target_gridded_variables else list(tds.data_vars)
                        valid_tvars = [v for v in t_vars if v in tds.data_vars]
                        if not valid_tvars:
                            continue
                        
                        t_lat_dim = 'lat' if 'lat' in tds.coords else 'latitude'
                        t_lon_dim = 'lon' if 'lon' in tds.coords else 'longitude'
                        t_lats = tds[t_lat_dim].values
                        t_lons = tds[t_lon_dim].values
                        
                        # Bounding box at target resolution
                        t_lat1 = np.abs(t_lats - min_lat).argmin()
                        t_lat2 = np.abs(t_lats - max_lat).argmin()
                        t_lon1 = np.abs(t_lons - min_lon).argmin()
                        t_lon2 = np.abs(t_lons - max_lon).argmin()
                        tl_min, tl_max = min(t_lat1, t_lat2), max(t_lat1, t_lat2)
                        tln_min, tln_max = min(t_lon1, t_lon2), max(t_lon1, t_lon2)
                        
                        t_lat_slice = slice(max(0, tl_min), min(len(t_lats), tl_max + 1))
                        t_lon_slice = slice(max(0, tln_min), min(len(t_lons), tln_max + 1))
                        
                        tds_sliced = tds[valid_tvars].isel({t_lat_dim: t_lat_slice, t_lon_dim: t_lon_slice}).load()
                        target_ds_list.append(tds_sliced)
                
                if target_ds_list:
                    target_merged = xr.concat(target_ds_list, dim="time", compat="override", coords="minimal")
                    
                    # Temporal alignment between target grid and observation index
                    if target_temporal_freq and target_temporal_freq != self.freq:
                        # Different temporal resolution (e.g. daily CHIRPS vs hourly ERA5)
                        # Align by matching each target timestamp to the observation day
                        target_times = pd.DatetimeIndex(target_merged.time.values)
                        obs_dates = observations.index.normalize().unique()
                        common_dates = target_times.normalize().intersection(obs_dates)
                        target_merged = target_merged.sel(time=target_merged.time.dt.normalize().isin(common_dates))
                        windows.target_temporal_freq = target_temporal_freq
                    else:
                        # Same temporal resolution: standard intersection
                        common_target_times = observations.index.intersection(target_merged.time.values)
                        target_merged = target_merged.sel(time=common_target_times)
                    
                    t_lat_dim = 'lat' if 'lat' in target_merged.coords else 'latitude'
                    t_lon_dim = 'lon' if 'lon' in target_merged.coords else 'longitude'
                    windows.target_grid_lats = target_merged[t_lat_dim].values
                    windows.target_grid_lons = target_merged[t_lon_dim].values
                    windows.target_gridded_features = list(target_merged.data_vars)
                    
                    # Pack target grid as numpy: (Time, Channels, Lat_hi, Lon_hi)
                    t_arr_list = []
                    for tvar in windows.target_gridded_features:
                        tvar_data = target_merged[tvar].values
                        if np.issubdtype(tvar_data.dtype, np.floating):
                            tvar_data = tvar_data.astype(np.float32)
                        if tvar_data.ndim == 3:
                            t_arr_list.append(tvar_data[:, np.newaxis, ...])
                    
                    if t_arr_list:
                        target_2d_np = np.concatenate(t_arr_list, axis=1)
                        
                        if target_temporal_freq and target_temporal_freq != self.freq:
                            # For daily targets with hourly inputs: create windows by day index
                            # Shape: (B, 1, C, Lat_hi, Lon_hi) -- one daily target per window
                            n_target = len(target_2d_np)
                            if n_target >= window_size + horizon_size:
                                windows.target_gridded_y = sliding_window_view(
                                    target_2d_np[window_size:], horizon_size
                                )
                            else:
                                windows.target_gridded_y = target_2d_np[np.newaxis, ...]
                        else:
                            # Same freq: standard sliding window on horizon
                            if len(target_2d_np) >= window_size + horizon_size:
                                windows.target_gridded_y = sliding_window_view(
                                    target_2d_np[window_size:], horizon_size
                                )
                        
                        # NaN rejection on target grid
                        if windows.target_gridded_y is not None:
                            n_batches = min(len(windows.x), len(windows.target_gridded_y))
                            target_valid = ~np.isnan(windows.target_gridded_y[:n_batches]).any(axis=(1, 2, 3, 4))
                            
                            # Intersect with existing valid_batches if applicable
                            if 'valid_batches' in dir():
                                combined = np.ones(n_batches, dtype=bool)
                                combined[:len(valid_batches)] &= valid_batches[:n_batches]
                                combined &= target_valid
                            else:
                                combined = target_valid
                            
                            windows.x = windows.x[:n_batches][combined]
                            windows.y = windows.y[:n_batches][combined]
                            windows.mask_x = windows.mask_x[:n_batches][combined]
                            windows.mask_y = windows.mask_y[:n_batches][combined]
                            windows.index_x = windows.index_x[:n_batches][combined]
                            windows.index_y = windows.index_y[:n_batches][combined]
                            if windows.gridded_x is not None:
                                windows.gridded_x = windows.gridded_x[:n_batches][combined]
                            if windows.gridded_y is not None:
                                windows.gridded_y = windows.gridded_y[:n_batches][combined]
                            windows.target_gridded_y = windows.target_gridded_y[:n_batches][combined]
                    
                    # Dual-Resolution Topo Priors: interpolate to target (high-res) grid
                    # Move existing static_grid to static_grid_input (encoder-res)
                    # and create new static_grid at target-res (decoder-res)
                    if self._topo_ds is not None:
                        windows.static_grid_input = windows.static_grid
                        windows.static_grid_input_categorical = windows.static_grid_categorical
                        
                        topo_vars = list(self._topo_ds.data_vars)
                        tp_lat_dim = 'lat' if 'lat' in self._topo_ds.coords else 'latitude'
                        tp_lon_dim = 'lon' if 'lon' in self._topo_ds.coords else 'longitude'
                        
                        tg_lat_xr = xr.DataArray(windows.target_grid_lats, dims="lat")
                        tg_lon_xr = xr.DataArray(windows.target_grid_lons, dims="lon")
                        
                        topo_target = self._topo_ds.interp(
                            {tp_lat_dim: tg_lat_xr, tp_lon_dim: tg_lon_xr}, method="nearest"
                        ).load()
                        
                        tg_cat = [v for v in topo_vars if v in cat_cols]
                        tg_cont = [v for v in topo_vars if v not in cat_cols]
                        
                        tc_arr = []
                        for tv in tg_cont:
                            tc_arr.append(topo_target[tv].values.astype(np.float32)[np.newaxis, ...])
                        if tc_arr:
                            windows.static_grid = np.concatenate(tc_arr, axis=0)
                        
                        tcat_arr = []
                        for tv in tg_cat:
                            tcat_arr.append(topo_target[tv].values.astype(np.int64)[np.newaxis, ...])
                        if tcat_arr:
                            windows.static_grid_categorical = np.concatenate(tcat_arr, axis=0)

        windows.station_labels = valid_stations_subset
        windows.feature_labels = [parameters] if isinstance(parameters, str) else list(parameters or self.parameters)

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
        
        single.x = single.x[batch_idx:batch_idx+1]
        single.mask_x = single.mask_x[batch_idx:batch_idx+1]
        single.y = single.y[batch_idx:batch_idx+1]
        single.mask_y = single.mask_y[batch_idx:batch_idx+1]
        single.index_x = single.index_x[batch_idx:batch_idx+1]
        single.index_y = single.index_y[batch_idx:batch_idx+1]
        
        if single.gridded_x is not None:
            single.gridded_x = single.gridded_x[batch_idx:batch_idx+1]
        if single.gridded_y is not None:
            single.gridded_y = single.gridded_y[batch_idx:batch_idx+1]
            
        use_stations = stations if stations is not None else windows.station_labels
        use_parameters = parameters if parameters is not None else windows.feature_labels
            
        return self.np_windows_as_xr(single, use_stations, use_parameters)

    def _get_windows_coordinates(self, windows: Windows, stations=None, parameters=None) -> dict:
        reftime = pd.DatetimeIndex(windows.index_x[:, -1]).tz_localize(None)
        window_size = windows.x.shape[1]
        horizon_size = windows.y.shape[1]
        
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
        windows.x = xr.Dataset(get_data_vars(windows.x, feature, "x", is_grid=False), coords=coords_x)
        windows.mask_x = xr.Dataset(get_data_vars(windows.mask_x, feature, "x", is_grid=False), coords=coords_x)
        windows.y = xr.Dataset(get_data_vars(windows.y, feature, "y", is_grid=False), coords=coords_y)
        windows.mask_y = xr.Dataset(get_data_vars(windows.mask_y, feature, "y", is_grid=False), coords=coords_y)
        
        # Pack Gridded Tensors if they exist
        if windows.gridded_x is not None:
            is_grid_mode = windows.gridded_x.ndim == 5
            
            if is_grid_mode:
                # Build spatial coordinate dictionaries
                grid_coords_x = {
                    "reftime": coords_x["reftime"],
                    "lag": coords_x["lag"],
                    "lat": ("lat", windows.grid_lats),
                    "lon": ("lon", windows.grid_lons)
                }
                grid_coords_y = {
                    "reftime": coords_y["reftime"],
                    "lead": coords_y["lead"],
                    "lat": ("lat", windows.grid_lats),
                    "lon": ("lon", windows.grid_lons)
                }
                windows.gridded_x = xr.Dataset(get_data_vars(windows.gridded_x, windows.gridded_features, "x", is_grid=True), coords=grid_coords_x)
                if windows.gridded_y is not None:
                    windows.gridded_y = xr.Dataset(get_data_vars(windows.gridded_y, windows.gridded_features, "y", is_grid=True), coords=grid_coords_y)
            else:
                windows.gridded_x = xr.Dataset(get_data_vars(windows.gridded_x, windows.gridded_features, "x", is_grid=False), coords=coords_x)
                if windows.gridded_y is not None:
                    windows.gridded_y = xr.Dataset(get_data_vars(windows.gridded_y, windows.gridded_features, "y", is_grid=False), coords=coords_y)

        return windows