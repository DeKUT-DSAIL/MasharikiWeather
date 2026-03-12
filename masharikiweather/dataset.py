import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union, Dict
import numpy as np
import pandas as pd
import xarray as xr
import fsspec
from tqdm import tqdm

FrameArray = Union[pd.DataFrame, np.ndarray]

@dataclass
class Windows:
    x: Union[pd.DataFrame, np.ndarray, xr.Dataset]
    mask_x: Union[pd.DataFrame, np.ndarray, xr.Dataset]
    y: Union[pd.DataFrame, np.ndarray, xr.Dataset]
    mask_y: Union[pd.DataFrame, np.ndarray, xr.Dataset]
    index_x: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None
    index_y: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None
    gridded: Optional[Union[np.ndarray, xr.Dataset]] = None
    gridded_to_y: Optional[Sequence[str]] = None

def sliding_window_view(a: np.ndarray, window_shape: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view as np_swv
    if hasattr(a, "values"):
        windows = np_swv(a.values, window_shape, axis=0)
    else:
        windows = np_swv(a, window_shape, axis=0)
    return np.moveaxis(windows, -1, 1)

def df_add_missing_columns(df: pd.DataFrame, col0, col1, fill_value=np.nan) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([col0, col1], names=['station_id', 'parameter'])
    missing = idx.difference(df.columns)
    if not missing.empty:
        df_missing = pd.DataFrame(fill_value, index=df.index, columns=missing)
        df = pd.concat([df, df_missing], axis=1)
        df = df.sort_index(axis=1)
    return df

class MasharikiWeatherDataset:
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
        imputation_method: Literal["locf", "zero", None] = "zero",
        pad_missing_values: bool = True,
        aggregation_methods: Optional[Dict[str, str]] = None,
        tabular_parquet_path: Optional[str] = None
    ):
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
        
        try:
            self.freq = pd.tseries.frequencies.to_offset(freq).freqstr
        except ValueError:
            raise ValueError(f"Value '{freq}' is not a valid pandas frequency string.")
        
        self.fs = fsspec.filesystem("hf", token=token)
        
        # Instantiate remote connections for all requested networks
        self._obs_zarr = {}
        for src in self.source_obs:
            os.makedirs(f"{self.root_dir}/obs/{src}", exist_ok=True)
            ds = self._open_zipped_zarr(f"data/obs/{src}.zarr.zip", lazy=True)
            if ds is None:
                raise ConnectionError(f"Failed to connect to the observation dataset: {src}")
            self._obs_zarr[src] = ds

        self._topo_ds = None
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
        full_path = f"hf://datasets/{self.repo_id}/{rel_path}"
        try:
            remote_file = self.fs.open(full_path)
            zip_mapper = fsspec.get_mapper("zip://", fo=remote_file)
            
            chunks = {} if lazy else None
            
            try:
                ds = xr.open_dataset(zip_mapper, engine="zarr", chunks=chunks, backend_kwargs={"consolidated": True})
            except Exception:
                ds = xr.open_dataset(zip_mapper, engine="zarr", chunks=chunks, backend_kwargs={"consolidated": False})
            return ds
        except Exception as e:
            print(f"Error: Could not open {rel_path}. {e}")
            return None

    def _load_registries(self):
        # Merge coordinates and parameter meta across all networks
        st_dfs = []
        pr_dfs = []
        
        for src, ds in self._obs_zarr.items():
            df_s = pd.DataFrame({
                "latitude": ds.lat.values if 'lat' in ds.coords else np.nan,
                "longitude": ds.lon.values if 'lon' in ds.coords else np.nan,
                "network": src  # Tag the nodes for graph analytics
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

        self.mask = ~obs_df.isna()
        
        if self.imputation_method == "locf":
            obs_df = obs_df.ffill()
        elif self.imputation_method == "zero":
            obs_df = obs_df.fillna(0)
        elif self.imputation_method is not None:
            raise ValueError(f"Invalid imputation method: {self.imputation_method}.")
        
        self.observations = obs_df

    def load_topography(self, filename: str = "east_africa_static_priors") -> Optional[xr.Dataset]:
        path = f"hf://datasets/{self.repo_id}/data/topography/{filename}.nc"
        try:
            with self.fs.open(path, mode="rb") as f:
                return xr.open_dataset(f, engine="h5netcdf", chunks={}).load()
        except Exception as e:
            print(f"Warning: Could not load topography data. {e}")
            return None

    def interpolate_topography(self, method: Literal["nearest", "linear"] = "nearest") -> None:
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

        try:
            sampled = self._topo_ds.sel(lat=lats, lon=lons, method=method) if method == "nearest" else self._topo_ds.interp(lat=lats, lon=lons, method=method)
        except KeyError:
            sampled = self._topo_ds.sel(latitude=lats, longitude=lons, method=method) if method == "nearest" else self._topo_ds.interp(latitude=lats, longitude=lons, method=method)

        df_topo = sampled.to_dataframe()
        cols_to_drop = [c for c in ['lat', 'lon', 'latitude', 'longitude'] if c in df_topo.columns]
        df_topo = df_topo.drop(columns=cols_to_drop, errors='ignore')

        self.stations_table = self.stations_table.join(df_topo)

    def validate_stations(self, stations: Union[str, List[str]]) -> List[str]:
        if isinstance(stations, str):
            stations = [stations]
            
        available = self.stations_table.index
        valid = [s for s in stations if s in available]
        
        if len(valid) < len(stations):
            missing = set(stations) - set(valid)
            print(f"Warning: {len(missing)} requested stations are not in the data source.")
            
        return valid

    def get_stations_metadata(self, stations: Union[str, List[str]]) -> pd.DataFrame:
        valid_stations = self.validate_stations(stations)
        return self.stations_table.loc[valid_stations]

    def get_gridded_for_stations(
        self, 
        groups: List[str], 
        stations: List[str], 
        years: Optional[Union[int, List[int]]] = None,
        variables: Optional[List[str]] = None,
        method: Literal["nearest", "linear"] = "nearest",
    ) -> Dict[str, pd.DataFrame]:
        
        valid_stations = self.validate_stations(stations)
        if not valid_stations:
            return {}

        target_years = [years] if isinstance(years, int) else (years or self.years)
        coords = self.stations_table.loc[valid_stations].dropna(subset=['latitude', 'longitude'])

        extracted_dfs = {}
        for group in groups:
            yearly_dfs = []
            
            for year in tqdm(target_years, desc=f"Extracting {group}"):
                ds = self._open_zipped_zarr(f"data/{group}/{year}.zarr.zip", lazy=True)
                if ds is None:
                    continue
                
                if variables:
                    valid_vars = [v for v in variables if v in ds.data_vars]
                    if valid_vars:
                        ds = ds[valid_vars]

                lat_dim = 'lat' if 'lat' in ds.coords else 'latitude'
                lon_dim = 'lon' if 'lon' in ds.coords else 'longitude'

                grid_lats = ds[lat_dim].values
                grid_lons = ds[lon_dim].values

                if method == "nearest":
                    lat_idx = [np.abs(grid_lats - lat).argmin() for lat in coords['latitude']]
                    lon_idx = [np.abs(grid_lons - lon).argmin() for lon in coords['longitude']]
                    
                    lat_idx_da = xr.DataArray(lat_idx, dims="station")
                    lon_idx_da = xr.DataArray(lon_idx, dims="station")
                    
                    sampled = ds.isel({lat_dim: lat_idx_da, lon_dim: lon_idx_da})
                    sampled = sampled.assign_coords(station=("station", coords.index.values))
                    sampled = sampled.load()
                
                # bilinear interpolation
                else:
                    lat_idx_1 = np.abs(grid_lats - coords['latitude'].min()).argmin()
                    lat_idx_2 = np.abs(grid_lats - coords['latitude'].max()).argmin()
                    lon_idx_1 = np.abs(grid_lons - coords['longitude'].min()).argmin()
                    lon_idx_2 = np.abs(grid_lons - coords['longitude'].max()).argmin()
                    
                    l_min, l_max = min(lat_idx_1, lat_idx_2), max(lat_idx_1, lat_idx_2)
                    ln_min, ln_max = min(lon_idx_1, lon_idx_2), max(lon_idx_1, lon_idx_2)
                    
                    lat_slice = slice(max(0, l_min - 2), min(len(grid_lats), l_max + 3))
                    lon_slice = slice(max(0, ln_min - 2), min(len(grid_lons), ln_max + 3))
                    
                    local_ds = ds.isel({lat_dim: lat_slice, lon_dim: lon_slice}).load()
                    
                    lats = xr.DataArray(coords.latitude.values, dims="station", coords={"station": coords.index.values})
                    lons = xr.DataArray(coords.longitude.values, dims="station", coords={"station": coords.index.values})
                    sampled = local_ds.interp({lat_dim: lats, lon_dim: lons}, method="linear")

                df = sampled.to_dataframe().reset_index()
                
                nuisance_coords = [lat_dim, lon_dim, 'latitude', 'longitude', 'step', 'number', 'surface', 'valid_time', 'height']
                cols_to_drop = [c for c in nuisance_coords if c in df.columns]
                df = df.drop(columns=cols_to_drop, errors='ignore')
                
                if variables:
                    allowed_cols = set(['time', 'station', 'station_id'] + variables)
                    keep_cols = [c for c in df.columns if c in allowed_cols]
                    df = df[keep_cols]
                    
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
        
        valid_stations = self.validate_stations(stations)
        if not valid_stations:
            return (np.empty((0,0,0)), np.empty((0,0,0))) if return_mask else np.empty((0,0,0))

        stations_loc = valid_stations
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
        target_years = [years] if isinstance(years, int) else (years or self.years)
        gridded_dict = {}
        
        for group in groups:
            datasets = []
            for year in target_years:
                ds = self._open_zipped_zarr(f"data/{group}/{year}.zarr.zip", lazy=True)
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

    def np_windows_as_xr(
        self,
        windows: Windows,
        stations: Optional[Union[str, List[str]]],
        parameters: Optional[Union[str, List[str]]],
    ) -> Windows:
        coord_info = self._get_windows_coordinates(windows, stations, parameters)
        return self._pack_windows_as_xarray(
            windows,
            feature=coord_info["features"],
            reftime=coord_info["reftime"],
            lagtime=coord_info["lagtime"],
            leadtime=coord_info["leadtime"],
            stations=coord_info["station"],
        )

    def _get_windows_coordinates(self, windows: Windows, stations=None, parameters=None):
        reftime = pd.DatetimeIndex(windows.index_x[:, -1]).tz_localize(None)
        window_size = windows.x.shape[1]
        horizon_size = windows.y.shape[1]
        
        if self.observations.empty:
            freq = pd.Timedelta("1h")
        else:
            freq = self.observations.index.freq if self.observations.index.freq else pd.Timedelta("1h")
            
        leadtime = pd.timedelta_range(start=1 * freq, periods=horizon_size, freq=freq)
        lagtime = pd.timedelta_range(start=-(window_size - 1) * freq, end=0, freq=freq)

        if isinstance(stations, str):
            stations = [stations]
        if isinstance(parameters, str):
            parameters = [parameters]

        stations_arr = self.stations if stations is None else pd.Index(stations)
        features_arr = self.parameters if parameters is None else pd.Index(parameters)

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
        feature: Union[str, pd.Index, np.ndarray],
        reftime: Union[List, np.ndarray, pd.DatetimeIndex],
        lagtime: Sequence[pd.Timedelta],
        leadtime: Sequence[pd.Timedelta],
        stations: Union[pd.Index, np.ndarray],
    ) -> Windows:
        
        def get_data_vars(data, feature, x_or_y: str):
            if data.ndim == 3:
                feature = [feature] if isinstance(feature, str) else feature
                data = np.expand_dims(data, axis=-1)
            second_dim = "lag" if x_or_y == "x" else "lead"
            return {
                str(f): (["reftime", second_dim, "station"], data[..., i])
                for i, f in enumerate(feature)
            }

        station_vals = stations.values if hasattr(stations, 'values') else np.array(stations)
        reftime_vals = reftime.values if hasattr(reftime, 'values') else np.array(reftime)
        lagtime_vals = lagtime.values if hasattr(lagtime, 'values') else np.array(lagtime)
        leadtime_vals = leadtime.values if hasattr(leadtime, 'values') else np.array(leadtime)

        coords_x = {
            "reftime": ("reftime", reftime_vals),
            "lag": ("lag", lagtime_vals),
            "station": ("station", station_vals)
        }
        coords_y = {
            "reftime": ("reftime", reftime_vals),
            "lead": ("lead", leadtime_vals),
            "station": ("station", station_vals)
        }

        x = xr.Dataset(get_data_vars(windows.x, feature, "x"), coords=coords_x)
        mask_x = xr.Dataset(get_data_vars(windows.mask_x, feature, "x"), coords=coords_x)
        y = xr.Dataset(get_data_vars(windows.y, feature, "y"), coords=coords_y)
        mask_y = xr.Dataset(get_data_vars(windows.mask_y, feature, "y"), coords=coords_y)

        return Windows(
            x=x, mask_x=mask_x, y=y, mask_y=mask_y,
            index_x=windows.index_x, index_y=windows.index_y
        )

    def align_windows(
        self, obs: Windows, gridded_data: "xr.Dataset", drop_extra_y_pars: bool, as_xarray: bool
    ) -> Tuple[Windows, Union["xr.Dataset", np.ndarray]]:
        
        gridded_pars = list(gridded_data.data_vars)
        if drop_extra_y_pars:
            valid_y_pars = [p for p in gridded_pars if p in obs.y.data_vars]
            obs.y = obs.y[valid_y_pars]
            obs.mask_y = obs.mask_y[valid_y_pars]
        else:
            obs.gridded_to_y = gridded_pars

        obs.y, gridded_aligned = xr.align(obs.y, gridded_data, join="inner")
        obs.x, _ = xr.align(obs.x, obs.y, join="inner")
        obs.mask_x, _ = xr.align(obs.mask_x, obs.x, join="inner")
        obs.mask_y, _ = xr.align(obs.mask_y, obs.y, join="inner")

        obs.index_x = pd.DatetimeIndex(obs.x.coords["reftime"].values)
        obs.index_y = pd.DatetimeIndex(obs.y.coords["reftime"].values)

        if as_xarray:
            return obs, gridded_aligned

        def xr_to_np(ds):
            return np.stack([ds[v].values for v in ds.data_vars], axis=-1)

        gridded_np = xr_to_np(gridded_aligned)
        obs.x = xr_to_np(obs.x)
        obs.mask_x = xr_to_np(obs.mask_x)
        obs.y = xr_to_np(obs.y)
        obs.mask_y = xr_to_np(obs.mask_y)
        return obs, gridded_np

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
        drop_extra_y_pars: bool = True,
        as_xarray: bool = False,
    ) -> Windows:
        windows = self.get_observation_windows(
            window_size=window_size,
            horizon_size=horizon_size,
            stations=stations,
            parameters=parameters,
            first_date=first_date,
            last_date=last_date,
            split=split,
            as_xarray=bool(gridded_url) or as_xarray,
        )

        if gridded_url:
            gridded_data = self.load_gridded_data(gridded_url)
            windows, gridded_aligned = self.align_windows(
                windows, gridded_data, drop_extra_y_pars, as_xarray
            )
            windows.gridded = gridded_aligned

        return windows