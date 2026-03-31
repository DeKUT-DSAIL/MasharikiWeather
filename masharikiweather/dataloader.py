"""
PyTorch Geometric DataLoader mappings for Mashariki Weather Dataset.
Implementing the EarthSystemAdapter abstracting targeted TSL
Data handling architectures natively evaluating N-Grids dynamically.

Graph Construction:
    Station-to-Station edges use K-Nearest Neighbors with anisotropic
    edge_attr encoding inverse haversine distance, elevation gradients,
    slope differences, and aspect alignment (PeakWeather methodology).

    Grid-to-Target bipartite edges spatially map coarse macroscopic grid
    nodes to target nodes (stations or finer grids) using KNN (K=4 default
    to capture bounding cell geometry).

    All edge tensors are constructed once in __init__ via _build_hetero_edges()
    and cached for O(1) attachment in __getitem__.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Optional, Literal, Dict, Tuple, Sequence

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader
from tsl.data.preprocessing import StandardScaler
from .dataset import Windows


# ---------------------------------------------------------------------------
# Scientific Scaling Engine
# ---------------------------------------------------------------------------
class ScalingEngine:
    """
    Orchestrates modality-specific scaling to satisfy scientific correctness.
    - PHYSICAL: Standard z-score normalization (Gaussian assumption).
    - PRECIPITATION: Raw intensity preservation for Tweedie/ZIG optimization (Rule 3).
    - STATIC: Native resolution preservation.
    """
    def __init__(
        self, 
        stats: Dict[str, Dict[str, np.ndarray]], 
        variable_types: Dict[str, str]
    ):
        self.stats = stats
        self.variable_types = variable_types
        
    def transform(self, data: torch.Tensor, variables: Sequence[str]) -> torch.Tensor:
        """
        Selectively transforms variables based on their scientific classification.
        """
        transformed = data.clone()
        for i, var in enumerate(variables):
            v_type = self.variable_types.get(var, "PHYSICAL")
            
            if v_type == "PHYSICAL" and self.stats:
                # Apply Gaussian scaling
                v_mean = self.stats.get('mean', {}).get(var, 0.0)
                v_std = self.stats.get('std', {}).get(var, 1.0)
                # Handle zero-variance safety
                v_std = v_std if v_std > 1e-6 else 1.0
                transformed[..., i] = (data[..., i] - v_mean) / v_std
                
            elif v_type == "PRECIPITATION":
                # Pass raw (Rule 3)
                transformed[..., i] = data[..., i]
                
        return transformed


# ---------------------------------------------------------------------------
# Utility: Cyclical temporal encoding
# ---------------------------------------------------------------------------
def datetime_encoded(index: pd.DatetimeIndex) -> np.ndarray:
    """Cyclical temporal encoding for Earth observation data."""
    hour = index.hour.values / 24.0
    dow = index.dayofweek.values / 7.0
    doy = index.dayofyear.values / 365.25

    encoded = np.stack([
        np.sin(2 * np.pi * hour), np.cos(2 * np.pi * hour),
        np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow),
        np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)
    ], axis=-1)
    return encoded.astype(np.float32)


# ---------------------------------------------------------------------------
# Utility: Haversine distance
# ---------------------------------------------------------------------------
def _haversine_distances(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise haversine distances (km) between two coordinate arrays.

    Mathematical formulation:
        d = 2R * arcsin(sqrt(sin^2((lat2-lat1)/2)
            + cos(lat1)*cos(lat2)*sin^2((lon2-lon1)/2)))

    Args:
        coords_a: (N, 2) array of [lat, lon] in degrees.
        coords_b: (M, 2) array of [lat, lon] in degrees.

    Returns:
        (N, M) distance matrix in kilometers.
    """
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(coords_a[:, 0])[:, None]
    lon1 = np.radians(coords_a[:, 1])[:, None]
    lat2 = np.radians(coords_b[:, 0])[None, :]
    lon2 = np.radians(coords_b[:, 1])[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class EarthSystemAdapter(Dataset):
    """
    A PyTorch Dataset rigidly following Torch Spatiotemporal (TSL) and PyTorch Geometric
    HeteroData specifications mapping completely Heterogeneous N-Grid boundaries.

    Architectural Mandate: Uses Composition over Inheritance.
    Inherits ONLY from torch.utils.data.Dataset.
    """
    def __init__(
        self,
        windows: Windows,
        ml_task: Literal["downscaling", "forecasting", "correction"],
        grid_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        k_stations: int = 8,
        k_grid: int = 4,
    ):
        self.windows = windows
        self.ml_task = ml_task

        # 1. Pre-cast Point Station Variables
        self.station_x = torch.as_tensor(
            np.ascontiguousarray(windows.station_x), dtype=torch.float32
        )
        self.station_mask_x = torch.as_tensor(
            np.ascontiguousarray(windows.station_mask_x), dtype=torch.bool
        )

        self.station_y = None
        self.station_mask_y = None
        if windows.station_y is not None and len(windows.station_y) > 0:
            self.station_y = torch.as_tensor(
                np.ascontiguousarray(windows.station_y), dtype=torch.float32
            )
            self.station_mask_y = torch.as_tensor(
                np.ascontiguousarray(windows.station_mask_y), dtype=torch.bool
            )

        self.window_size = self.station_x.shape[1]
        self.horizon_size = self.station_y.shape[1] if self.station_y is not None else 0

        # Station-graph geometry: n_stations and n_params for PyG reshape.
        # station_x shape from sliding_window_view: (n_windows, window_size, n_stations * n_params)
        # PeakWeather standard: per-sample node features are (n_stations, window_size, n_params)
        station_labels = getattr(windows, 'station_labels', None)
        feature_labels = getattr(windows, 'feature_labels', None)
        self.n_stations = len(station_labels) if station_labels else 0
        self.n_params = len(feature_labels) if feature_labels else 0

        # 2. Pre-cast Topographic Priors: Stations
        self.station_static = None
        if windows.station_static is not None:
            self.station_static = torch.as_tensor(
                windows.station_static, dtype=torch.float32
            )

        # 3. Pre-cast Topographic Priors: Native High-Resolution Bipartite DEM Nodes
        self.native_topography = None
        self.native_topography_categorical = None

        if getattr(windows, 'native_topography', None) is not None:
            raw_topo = windows.native_topography
            if hasattr(raw_topo, 'to_array'):
                raw_topo = raw_topo.to_array().values
            self.native_topography = torch.as_tensor(raw_topo, dtype=torch.float32)

        if getattr(windows, 'native_topography_categorical', None) is not None:
            raw_cat = windows.native_topography_categorical
            self.native_topography_categorical = torch.as_tensor(raw_cat, dtype=torch.int64)

        # 4. Initialize Scientific Scaling Engine
        self.scaling_engine = ScalingEngine(
            stats=grid_stats if grid_stats else {},
            variable_types=getattr(windows, 'variable_types', {})
        )

        # 5. Build static graph topology (Anisotropic 3D connectivity)
        self._cached_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._build_hetero_edges(k_stations=k_stations, k_grid=k_grid)

    # ------------------------------------------------------------------
    # Graph Construction (Executed Once)
    # ------------------------------------------------------------------
    def _build_hetero_edges(self, k_stations: int = 8, k_grid: int = 4) -> None:
        """
        Construct a fully coupled, bidirectional GNN topology.
        Implements 3D Anisotropic Distance: 
          d = sqrt( d_geo^2 + lambda * d_alt^2 )
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Override K for scientific density alignment (k=12 provides better multi-modal convergence)
        k_grid = max(k_grid, 12)

        # 1. Discover all spatial sources
        sources = {} 
        
        # Station Coords (Lat, Lon, Alt)
        st_coords = getattr(self.windows, 'station_coords', None)
        if st_coords is not None:
            # Assuming station_static contains Altitude at index 0 (Standard in Mashariki dataset)
            # If not, we fall back to 2D
            st_alt = self.windows.station_static[:, :1] if self.windows.station_static is not None else np.zeros((len(st_coords), 1))
            sources['tahmo'] = np.concatenate([st_coords, st_alt], axis=1)

        # Grid Coords (Lat, Lon, Interpolated Alt)
        for g_name in self.windows.grids_x.keys():
            lats = self.windows.grids_lats.get(g_name)
            lons = self.windows.grids_lons.get(g_name)
            if lats is not None and lons is not None:
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                coords_2d = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1)
                # Placeholder for interpolated altitude (will be expanded if DEM is visible)
                alt = np.zeros((len(coords_2d), 1))
                sources[g_name] = np.concatenate([coords_2d, alt], axis=1)

        # C. Native Topography (As a Node Type)
        if (self.windows.native_topography is not None and 
            self.windows.native_topography_lats is not None and 
            self.windows.native_topography_lons is not None):
            
            t_lats = self.windows.native_topography_lats
            t_lons = self.windows.native_topography_lons
            t_lon_grid, t_lat_grid = np.meshgrid(t_lons, t_lats)
            t_coords = np.stack([t_lat_grid.ravel(), t_lon_grid.ravel()], axis=1)
            
            # Extract only the elevation channel (index 0) for KNN altitude.
            # native_topography shape: (n_vars, n_lat, n_lon) from dataset.py L1675-1676.
            # Flattening all n_vars would produce n_vars*n_lat*n_lon elements vs
            # n_lat*n_lon coords -- a fatal dimension mismatch.
            if isinstance(self.native_topography, torch.Tensor) and self.native_topography.dim() == 3:
                topo_elev = self.native_topography[0]  # (n_lat, n_lon)
            elif isinstance(self.native_topography, np.ndarray) and self.native_topography.ndim == 3:
                topo_elev = self.native_topography[0]
            else:
                topo_elev = self.native_topography
            t_alt = topo_elev.reshape(-1, 1).numpy() if isinstance(topo_elev, torch.Tensor) else topo_elev.reshape(-1, 1)
            sources['topography'] = np.concatenate([t_coords, t_alt], axis=1)

        if not sources:
            return

        # 2. Coupling Loop (Bidirectional)
        all_node_names = list(sources.keys())
        for i, src_name in enumerate(all_node_names):
            for j, dst_name in enumerate(all_node_names):
                src_coords = sources[src_name]
                dst_coords = sources[dst_name]
                
                # Weight altitude difference by 0.1 to prevent vertical dominance over horizontal distance
                # (100m altitude diff ~ 1km horizontal similarity in meteorological space)
                dist_weight = np.array([1.0, 1.0, 0.01]) 
                
                knn_src = src_coords * dist_weight
                knn_dst = dst_coords * dist_weight
                
                k = min(k_stations if dst_name == 'tahmo' else k_grid, len(src_coords))
                if k < 1: continue
                
                nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(knn_src)
                distances, indices = nbrs.kneighbors(knn_dst)
                
                # Edge Index: [src_idx, dst_idx]
                edge_dst = np.repeat(np.arange(len(dst_coords)), k)
                edge_src = indices.ravel()
                
                index_tensor = torch.tensor(np.stack([edge_src, edge_dst]), dtype=torch.long)
                attr_tensor = torch.tensor(distances.ravel(), dtype=torch.float32).unsqueeze(-1)
                
                # Semantic Verb Resolver for PyG-compliant (src, relation, dst) tuple keys
                if src_name == dst_name:
                    verb = 'observes'
                elif src_name == 'tahmo':
                    verb = 'corrects'
                elif src_name == 'topography':
                    verb = 'conditions'
                else:
                    target_node = self.windows.target_node or 'tahmo'
                    if dst_name == target_node:
                        verb = 'downscales_to'
                    else:
                        verb = 'informs'
                
                edge_key = (src_name, verb, dst_name)
                self._cached_edges[edge_key] = (index_tensor, attr_tensor)



    # ------------------------------------------------------------------
    # Dataset Interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        if self.windows.valid_batch_indices is not None:
            return len(self.windows.valid_batch_indices)
        return len(self.station_x)

    def _safe_flatten_grid(self, raw_numpy: np.ndarray) -> np.ndarray:
        """
        Dynamically flattens N-dimensional atmospheric grids based on their shape.
        e.g., 5D (Channels, Time, Pressure, Lat, Lon) -> 4D (Time, Channels*Pressure, Lat, Lon)
        """
        if raw_numpy.ndim == 4:
            # (Channels, Time, Lat, Lon) -> (Time, Channels, Lat, Lon)
            return np.transpose(raw_numpy, (1, 0, 2, 3))
        elif raw_numpy.ndim == 5:
            # (Channels, Time, Pressure, Lat, Lon) -> (Time, Channels*Pressure, Lat, Lon)
            raw_numpy = np.transpose(raw_numpy, (1, 0, 2, 3, 4))
            time, c, p, lat, lon = raw_numpy.shape
            return raw_numpy.reshape(time, c * p, lat, lon)
        return raw_numpy

    @staticmethod
    def _reshape_grid_for_pyg(tensor: torch.Tensor) -> torch.Tensor:
        """
        Reshape grid tensor from (Time, Channels, Lat, Lon) to PyG node-feature
        format (Lat*Lon, Time*Channels) where each spatial pixel is a graph node.

        Scientific justification (Aurora, Building ML LAMs, GraphCast):
        Per-variable normalization must occur BEFORE this reshape while the
        channel dimension is cleanly separated, preventing channel bleed.
        This method is called AFTER scaling.
        """
        if tensor.dim() == 4:
            T, C, H, W = tensor.shape
            # (T, C, H, W) -> (H, W, T, C) -> (H*W, T*C)
            return tensor.permute(2, 3, 0, 1).reshape(H * W, T * C)
        elif tensor.dim() == 2:
            return tensor  # Already (n_nodes, n_features)
        return tensor.reshape(-1, tensor.shape[-1]) if tensor.dim() > 2 else tensor

    def __getitem__(self, idx: int) -> HeteroData:
        data = HeteroData()

        # Real time-index lookup (JIT Execution)
        start_time = (
            self.windows.valid_batch_indices[idx]
            if self.windows.valid_batch_indices is not None
            else idx
        )
        end_time = start_time + self.window_size

        # 1. Dynamic Grid Loop (Lazy Materialization)
        for g_name, grid_ds in self.windows.grids_x.items():
            grid_slice = grid_ds.isel(time=slice(start_time, end_time))
            # Explicitly triggering safe Zarr read into worker RAM
            raw_numpy = grid_slice.compute().to_array().values
            flat_numpy = self._safe_flatten_grid(raw_numpy)

            tensor_x = torch.as_tensor(flat_numpy, dtype=torch.float32)

            # Step 1: Apply variable-specific scaling in grid-space BEFORE flatten.
            # Per Aurora, GraphCast, Building ML LAMs: normalization in native
            # (Time, Channels, Lat, Lon) prevents channel bleed indexing errors.
            grid_vars = list(grid_ds.data_vars)
            if grid_vars and self.scaling_engine.stats:
                tensor_x = self.scaling_engine.transform(tensor_x, grid_vars)

            # Step 2: Reshape to PyG node-feature format AFTER scaling.
            # (Time, Channels, Lat, Lon) -> (Lat*Lon, Time*Channels)
            tensor_x = self._reshape_grid_for_pyg(tensor_x)

            data[g_name].x = tensor_x

        # 2. Universal Target Routing Modality
        target_node = self.windows.target_node

        if target_node == "tahmo" or target_node is None:
            if self.station_y is not None:
                raw_y = self.station_y[idx]       # (horizon_size, n_stations * n_params)
                raw_mask_y = self.station_mask_y[idx]
                if self.n_stations > 0 and self.n_params > 0 and self.horizon_size > 0:
                    data["tahmo"].y = raw_y.reshape(
                        self.horizon_size, self.n_stations, self.n_params
                    ).permute(1, 0, 2)  # (n_stations, horizon_size, n_params)
                    data["tahmo"].mask_y = raw_mask_y.reshape(
                        self.horizon_size, self.n_stations, self.n_params
                    ).permute(1, 0, 2)
                else:
                    data["tahmo"].y = raw_y
                    data["tahmo"].mask_y = raw_mask_y
        elif target_node in self.windows.grids_y:
            delay = getattr(self.windows, 'delay', 0)
            target_start = start_time + self.window_size + delay
            target_end = target_start + self.horizon_size

            target_grid_slice = self.windows.grids_y[target_node].isel(
                time=slice(target_start, target_end)
            )
            # Explicit JIT compute
            target_numpy = target_grid_slice.compute().to_array().values
            flat_target = self._safe_flatten_grid(target_numpy)

            data[target_node].y = torch.as_tensor(flat_target, dtype=torch.float32)
            # Reshape grid target to PyG node-feature format
            data[target_node].y = self._reshape_grid_for_pyg(data[target_node].y)

        # 3. Global Coordinates Mapping (Temporal Encodings)
        if len(self.windows.grids_x) > 0:
            # Scenario A: Grids exist. Extract the sequence directly from the physical Xarray slice.
            first_grid_key = next(iter(self.windows.grids_x))
            actual_time_slice = self.windows.grids_x[first_grid_key].isel(time=slice(start_time, end_time))
            time_seq = pd.DatetimeIndex(actual_time_slice.time.values)
        else:
            # Scenario B: Station-only. Slice a full sequence of length `window_size` 
            # using the physically safe `start_time`, NOT the raw batch `idx`.
            time_seq = pd.DatetimeIndex(
                self.windows.index_x[start_time : start_time + self.window_size]
            )
            
        encoded_time = datetime_encoded(time_seq)
        data['global'].u = torch.as_tensor(encoded_time, dtype=torch.float32)

        # 4. Point-Station Nodes
        # PeakWeather standard: X_t \in R^{N x T x D_x} (stations, time, params)
        # WeatherGNN: R^{N x F_e x T} (stations, params, time)
        # sliding_window_view produces: (window_size, n_stations * n_params)
        # Reshape to: (n_stations, window_size, n_params) -- PeakWeather convention.
        raw_x = self.station_x[idx]      # (window_size, n_stations * n_params)
        raw_mask = self.station_mask_x[idx]
        if self.n_stations > 0 and self.n_params > 0:
            # Unflatten the joint station-parameter axis, then permute to (N, T, D)
            data['tahmo'].x = raw_x.reshape(
                self.window_size, self.n_stations, self.n_params
            ).permute(1, 0, 2)  # (n_stations, window_size, n_params)
            data['tahmo'].mask = raw_mask.reshape(
                self.window_size, self.n_stations, self.n_params
            ).permute(1, 0, 2)
        else:
            # Fallback: cannot determine decomposition
            data['tahmo'].x = raw_x
            data['tahmo'].mask = raw_mask
        if self.station_static is not None:
            data['tahmo'].static_u = self.station_static

        # 5. Native Topography Nodes
        # Reshape from (n_vars, n_lat, n_lon) -> (n_lat*n_lon, n_vars)
        # Each spatial pixel becomes a graph node with n_vars features.
        if self.native_topography is not None:
            if self.native_topography.dim() == 3:
                n_v, n_h, n_w = self.native_topography.shape
                data['topography'].x = self.native_topography.reshape(n_v, -1).T
            else:
                data['topography'].x = self.native_topography.reshape(-1, 1)
        if self.native_topography_categorical is not None:
            if self.native_topography_categorical.dim() == 3:
                data['topography'].x_categorical = self.native_topography_categorical.reshape(
                    self.native_topography_categorical.shape[0], -1
                ).T
            else:
                data['topography'].x_categorical = self.native_topography_categorical.reshape(-1, 1)

        # 6. Attach Cached Graph Edges
        for edge_key, (edge_index, edge_attr) in self._cached_edges.items():
            data[edge_key].edge_index = edge_index
            data[edge_key].edge_attr = edge_attr

        return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_dataloader(
    windows: Windows,
    ml_task: Literal["downscaling", "forecasting", "correction"],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    grid_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    k_stations: int = 8,
    k_grid: int = 4,
) -> PyGDataLoader:
    dataset = EarthSystemAdapter(
        windows, ml_task, grid_stats=grid_stats,
        k_stations=k_stations, k_grid=k_grid,
    )
    return PyGDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
    )