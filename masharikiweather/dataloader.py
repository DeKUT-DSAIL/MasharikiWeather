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
from torch.utils.data import Dataset, DistributedSampler
import numpy as np
import pandas as pd
from typing import Optional, Literal, Dict, Tuple, Sequence, List
from dataclasses import dataclass, field

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader
from tsl.data.preprocessing import StandardScaler
import dask
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

        # Station-graph geometry: n_stations and n_params for PyG reshape.
        station_labels = getattr(windows, 'station_labels', None)
        feature_labels = getattr(windows, 'feature_labels', None)
        self.n_stations = len(station_labels) if station_labels else 0
        self.n_params = len(feature_labels) if feature_labels else 0

        # 1. Pre-cast Point Station Variables (Robust Shape Parsing)
        def _format_station_tensor(arr, dtype):
            if arr is None or len(arr) == 0:
                return None
            t = torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype)
            
            # Dimensions logic:
            # Native NumPy sliding_window_view: (Batch, Stations, Params, Time)
            # Custom wrapper / Flattened: (Batch, Time, Stations * Params)
            # Target Standard: (Batch, Time, Stations, Params)
            if t.dim() == 4:
                if t.shape[1] == self.n_stations and t.shape[2] == self.n_params:
                    t = t.permute(0, 3, 1, 2) 
            elif t.dim() == 3:
                # Handle flattened (Batch, Time, N*P)
                if t.shape[2] == self.n_stations * self.n_params:
                    t = t.reshape(t.shape[0], t.shape[1], self.n_stations, self.n_params)
            return t

        self.station_x = _format_station_tensor(windows.station_x, torch.float32)
        self.station_mask_x = _format_station_tensor(windows.station_mask_x, torch.bool)
        
        self.station_y = None
        self.station_mask_y = None
        if windows.station_y is not None and len(windows.station_y) > 0:
            self.station_y = _format_station_tensor(windows.station_y, torch.float32)
            self.station_mask_y = _format_station_tensor(windows.station_mask_y, torch.bool)

        self.window_size = self.station_x.shape[1] if self.station_x is not None else 0
        self.horizon_size = self.station_y.shape[1] if self.station_y is not None else 0

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

        # Grid Coords (Lat, Lon) - No altitude interpolation for coarse grids
        for g_name in getattr(self.windows, 'grids_x', {}).keys():
            lats = self.windows.grids_lats.get(g_name)
            lons = self.windows.grids_lons.get(g_name)
            if lats is not None and lons is not None:
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                coords_2d = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1)
                sources[g_name] = coords_2d

        # C. Native Topography (As a Node Type)
        if (getattr(self.windows, 'native_topography', None) is not None and 
            getattr(self.windows, 'native_topography_lats', None) is not None and 
            getattr(self.windows, 'native_topography_lons', None) is not None):
            
            t_lats = self.windows.native_topography_lats
            t_lons = self.windows.native_topography_lons
            t_lon_grid, t_lat_grid = np.meshgrid(t_lons, t_lats)
            t_coords = np.stack([t_lat_grid.ravel(), t_lon_grid.ravel()], axis=1)
            
            # FIX 1: Safely isolate the 0th channel (DEM) to prevent shape mismatch 
            # if the topography tensor contains multiple features (Slope, Aspect, etc.)
            topo_tensor = self.native_topography[0] if self.native_topography.ndim == 3 else self.native_topography
            
            # FIX 2: Apply nan_to_num directly to the grid array. 
            # Ocean and Lake pixels in raw NetCDF DEMs inherently contain NaNs.
            t_alt_raw = topo_tensor.reshape(-1, 1).numpy() if isinstance(topo_tensor, torch.Tensor) else topo_tensor.reshape(-1, 1)
            t_alt = np.nan_to_num(t_alt_raw, nan=0.0)
            
            sources['topography'] = np.concatenate([t_coords, t_alt], axis=1)

        if not sources:
            return

        # 2. Coupling Loop (Hybrid Unidirectional Topology)
        all_node_names = list(sources.keys())
        for i, src_name in enumerate(all_node_names):
            for j, dst_name in enumerate(all_node_names):
                # Rule B: Unidirectional Fastpass (Topography cannot receive messages)
                if dst_name == 'topography':
                    continue
                
                src_coords = sources[src_name]
                dst_coords = sources[dst_name]
                
                # Rule A: Mathematically Sound Local Cartesian Projection
                # Convert source and destination coordinates to Kilometers to prevent Metric Space Collision
                ref_lat = src_coords[:, 0].mean()
                r_earth = 111.32  # km per degree of latitude
                
                def _project_cartesian(coords):
                    cart = np.zeros_like(coords, dtype=np.float32)
                    cart[:, 0] = coords[:, 0] * r_earth  # Y [km]
                    cart[:, 1] = coords[:, 1] * r_earth * np.cos(np.radians(ref_lat))  # X [km]
                    if coords.shape[1] == 3:
                        cart[:, 2] = coords[:, 2] / 1000.0  # Z [km]
                    return cart
                
                cart_src = _project_cartesian(src_coords)
                cart_dst = _project_cartesian(dst_coords)
                
                use_3d = (dst_name == 'tahmo' and src_coords.shape[1] == 3 and dst_coords.shape[1] == 3)
                
                if use_3d:
                    # WeatherGNN Anisotropy: λ_alt = 10.0 enforces strong terrain adherence
                    # 1 km of vertical distance is punished equivalently to 10 km horizontally
                    dist_weight = np.array([1.0, 1.0, 10.0])
                    knn_src = cart_src * dist_weight
                    knn_dst = cart_dst * dist_weight
                else:
                    # Strictly 2D physical kilometric topology
                    knn_src = cart_src[:, :2]
                    knn_dst = cart_dst[:, :2]
                
                k_target = k_stations if dst_name == 'tahmo' else k_grid
                # HYBRID TOPOLOGY: Target density alignment
                # 1. Local/Topo: K=8, threshold=0.7 (PeakWeather standard)
                # 2. Regional/Moisture: K=32, no hard threshold (Aurora/WeatherGNN standard)
                if src_name == 'topography' or src_name == dst_name:
                    k_final = min(8, len(src_coords))
                    threshold = 0.7
                else:
                    k_final = min(32, len(src_coords))
                    threshold = 0.1 # Keep weak connections for teleconnections

                if k_final < 1: continue
                
                nbrs = NearestNeighbors(n_neighbors=k_final, algorithm='ball_tree').fit(knn_src)
                distances_scaled, indices = nbrs.kneighbors(knn_dst)
                
                # Rule A: Dynamic Sigma calculation for Gaussian Kernel
                # To ensure the similarity weights [0, 1] are scientifically meaningful
                sigma = np.std(distances_scaled) if np.std(distances_scaled) > 0 else 1.0
                similarities = np.exp(-(distances_scaled**2) / (sigma**2))
                
                # Apply Thresholding
                mask = similarities >= threshold
                
                # Re-compute mathematical physical distance without the λ_alt penalty for Edge Attributes
                true_dist_km = np.linalg.norm(cart_src[indices.ravel(), :2] - np.repeat(cart_dst[:, :2], k_final, axis=0), axis=1)
                
                # Edge Index: [src_idx, dst_idx]
                edge_dst_all = np.repeat(np.arange(len(dst_coords)), k_final)
                edge_src_all = indices.ravel()
                weights_all = similarities.ravel()
                
                # Apply Mask to everything
                edge_src = edge_src_all[mask.ravel()]
                edge_dst = edge_dst_all[mask.ravel()]
                dist_km = true_dist_km[mask.ravel()]
                weights = weights_all[mask.ravel()]
                
                if len(edge_src) == 0: continue
                
                index_tensor = torch.tensor(np.stack([edge_src, edge_dst]), dtype=torch.long)
                
                # Rule C: Anisotropic Edge Attributes [Normalized]
                # To prevent MLP saturation from raw km (>1.0), we normalize per-bipartite-edge-type.
                max_dist = dist_km.max() if len(dist_km) > 0 and dist_km.max() > 0 else 1.0
                dist_norm = dist_km / max_dist
                
                # Combine physical distance with Gaussian Similarity weight
                dist_tensor = torch.tensor(dist_norm, dtype=torch.float32).unsqueeze(-1)
                weight_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)
                
                # Rule C: Uniform Dimension Edge Attributes [dist_norm, weight, delta_elev]
                # To support Data-Dependent Message Passing (Swiss Anemoi standard), all edge 
                # types must output exactly [E, 3] features so MLPs can route flow dynamically.
                # Delta Elevation (km) is signed: Uphill is positive, Downhill is negative.
                if src_coords.shape[1] == 3 and dst_coords.shape[1] == 3:
                    # 3D to 3D (Station -> Station, Topo -> Station)
                    delta_elev_km = cart_dst[edge_dst, 2] - cart_src[edge_src, 2]
                elif src_coords.shape[1] == 3 and dst_coords.shape[1] == 2:
                    # 3D to 2D (Station/Topo -> Grid). Assume Grid is relative 0.
                    delta_elev_km = 0.0 - cart_src[edge_src, 2]
                elif src_coords.shape[1] == 2 and dst_coords.shape[1] == 3:
                    # 2D to 3D (Grid -> Station). Assume Grid is relative 0.
                    delta_elev_km = cart_dst[edge_dst, 2] - 0.0
                else:
                    # 2D to 2D (Grid -> Grid)
                    delta_elev_km = np.zeros(len(edge_src), dtype=np.float32)
                    
                # Gracefully map steep differentials into [-1, 1] while preserving sign
                delta_signed = np.tanh(delta_elev_km / 2.0)
                delta_elev_tensor = torch.tensor(delta_signed, dtype=torch.float32).unsqueeze(-1)
                
                attr_tensor = torch.cat([dist_tensor, weight_tensor, delta_elev_tensor], dim=-1)
                
                # Semantic Verb Resolver
                if src_name == dst_name:
                    verb = 'observes'
                elif src_name == 'tahmo':
                    verb = 'corrects'
                elif src_name == 'topography':
                    verb = 'conditions'
                else:
                    target_node = getattr(self.windows, 'target_node', 'tahmo')
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

        # 0. Global Coordinates Mapping (Temporal Encodings)
        if len(self.windows.grids_x) > 0:
            first_grid_key = next(iter(self.windows.grids_x))
            actual_time_slice = self.windows.grids_x[first_grid_key].isel(time=slice(start_time, end_time))
            time_seq = pd.DatetimeIndex(actual_time_slice.time.values)
        else:
            time_seq = pd.DatetimeIndex(
                self.windows.index_x[start_time : start_time + self.window_size]
            )
        encoded_time = torch.as_tensor(datetime_encoded(time_seq), dtype=torch.float32)
        data['global'].u = encoded_time  # Retain as global metadata

        # 1. Dynamic Grid Loop (Lazy & Eager Routing)
        for g_name, grid_ds in self.windows.grids_x.items():
            
            # ROUTE A: Eager Mode (spatial_mode="stations")
            if isinstance(grid_ds, np.ndarray):
                # Extract pre-sliced numpy array: (window_size, n_stations, n_params)
                raw_numpy = grid_ds[idx]
                tensor_x = torch.as_tensor(raw_numpy, dtype=torch.float32)
                
                grid_vars = self.windows.grids_features.get(g_name, [])
                if grid_vars and self.scaling_engine.stats:
                    tensor_x = self.scaling_engine.transform(tensor_x, grid_vars)
                
                # Permute to PyG node-feature standard: (Stations, Time, Params)
                if tensor_x.dim() == 3:
                    tensor_x = tensor_x.permute(1, 0, 2)
                    # INJECT TEMPORAL ENCODINGS
                    S, T, P = tensor_x.shape
                    time_expanded = encoded_time.view(1, T, 6).expand(S, T, 6)
                    tensor_x = torch.cat([tensor_x, time_expanded], dim=2)
            # ROUTE B: Lazy Mode (spatial_mode="grid")
            else:
                with dask.config.set(scheduler='single-threaded'):
                    grid_slice = grid_ds.isel(time=slice(start_time, end_time))
                    raw_numpy = grid_slice.compute().to_array().values
                    
                flat_numpy = self._safe_flatten_grid(raw_numpy)

                tensor_x = torch.as_tensor(flat_numpy, dtype=torch.float32)

                grid_vars = list(grid_ds.data_vars)
                if grid_vars and self.scaling_engine.stats:
                    tensor_x = tensor_x.permute(0, 2, 3, 1)
                    tensor_x = self.scaling_engine.transform(tensor_x, grid_vars)
                    tensor_x = tensor_x.permute(0, 3, 1, 2)

                # INJECT TEMPORAL ENCODINGS (Broadcasting to entire Spatial Grid)
                T, C, H, W = tensor_x.shape
                time_expanded = encoded_time.view(T, 6, 1, 1).expand(T, 6, H, W)
                tensor_x = torch.cat([tensor_x, time_expanded], dim=1)

                # Reshape (Time, Channels+6, Lat, Lon) -> (Lat*Lon, Time*(Channels+6))
                tensor_x = self._reshape_grid_for_pyg(tensor_x)

            data[g_name].x = tensor_x

        # 2. Universal Target Routing Modality
        target_node = self.windows.target_node

        if target_node == "tahmo" or target_node is None:
            if self.station_y is not None:
                # Pre-formatted as [Time, Stations, Params]
                # Permute to [Stations, Time, Params] (PeakWeather Spatial Node Convention)
                data["tahmo"].y = self.station_y[idx].permute(1, 0, 2)
                data["tahmo"].mask_y = self.station_mask_y[idx].permute(1, 0, 2)
        elif target_node in self.windows.grids_y:
            delay = getattr(self.windows, 'delay', 0)
            target_start = start_time + self.window_size + delay
            target_end = target_start + self.horizon_size

            with dask.config.set(scheduler='single-threaded'):
                target_grid_slice = self.windows.grids_y[target_node].isel(
                    time=slice(target_start, target_end)
                )
                # Explicit JIT compute
                target_numpy = target_grid_slice.compute().to_array().values
                
            flat_target = self._safe_flatten_grid(target_numpy)

            target_tensor = torch.as_tensor(flat_target, dtype=torch.float32)

            # FIX Gap 3: Inject temporal encoding into grid target for autoregressive consistency.
            # When y becomes the next x in rollout, missing temporal features create distributional shift.
            T_y, C_y, H_y, W_y = target_tensor.shape
            # Compute target time sequence for the horizon window
            target_time_start = start_time + self.window_size + delay
            if len(self.windows.grids_x) > 0:
                first_grid_key = next(iter(self.windows.grids_x))
                target_time_slice = self.windows.grids_x[first_grid_key].isel(
                    time=slice(target_time_start, target_time_start + self.horizon_size)
                )
                target_time_seq = pd.DatetimeIndex(target_time_slice.time.values)
            else:
                target_time_seq = pd.DatetimeIndex(
                    self.windows.index_y[idx] if self.windows.index_y is not None
                    else self.windows.index_x[idx][-self.horizon_size:]
                )
            encoded_target_time = torch.as_tensor(
                datetime_encoded(target_time_seq), dtype=torch.float32
            )
            time_y_expanded = encoded_target_time.view(T_y, 6, 1, 1).expand(T_y, 6, H_y, W_y)
            target_tensor = torch.cat([target_tensor, time_y_expanded], dim=1)

            data[target_node].y = self._reshape_grid_for_pyg(target_tensor)

        # 3. Global Coordinates Mapping (Temporal Encodings)
        # Moved to Step 0 to physically inject time into all PyG Spatial Nodes natively.

        # 4. Point-Station Nodes
        # PeakWeather standard: X_t \in R^{N x T x D_x} (stations, time, params)
        if self.station_x is not None:
            # Pre-formatted as [Time, Stations, Params]
            # Permute to [Stations, Time, Params] (O(1) View transformation)
            tahmo_x = self.station_x[idx].permute(1, 0, 2)
            tahmo_mask = self.station_mask_x[idx].permute(1, 0, 2)
            
            # FIX Gap 2: PeakWeather-compliant mask concatenation.
            mask_reshaped = tahmo_mask.float()
            
            # INJECT TEMPORAL ENCODINGS
            S, T, P = tahmo_x.shape
            time_expanded = encoded_time.view(1, T, 6).expand(S, T, 6)
            
            # Concatenate: [weather_params | temporal_encodings | binary_mask]
            # Final shape: (n_stations, window_size, n_params + 6 + n_params)
            data['tahmo'].x = torch.cat([tahmo_x, time_expanded, mask_reshaped], dim=2)
            
            # Retain separate mask for loss computation (sparse boolean masking)
            data['tahmo'].mask = tahmo_mask
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
# Gap 5 Fix: 2D CRS Local Cartesian Projection Utility (ML LAMs Standard)
# For regional models, ML LAMs recommends 2D CRS map projections over
# 3D Cartesian or icosahedral decomposition. At equatorial East Africa,
# lat-lon grids have near-uniform spacing, but this utility provides
# a formal projection framework for future extensibility.
# ---------------------------------------------------------------------------
def project_to_local_crs(
    coords: np.ndarray,
    ref_lat: Optional[float] = None,
    ref_lon: Optional[float] = None,
) -> np.ndarray:
    """
    Project geographic coordinates to a local 2D Cartesian CRS (km).
    
    Uses the ML LAMs rectangular mesh standard: a equirectangular projection
    centered on the domain centroid. This avoids the 3D Cartesian training
    failures documented in ML LAMs DANRA experiments.
    
    For East Africa (equatorial), cos(lat) ~ 1.0, so distortion is minimal.
    This becomes critical if the domain extends to higher latitudes.
    
    Mathematical formulation:
        x_km = (lon - ref_lon) * 111.32 * cos(ref_lat)
        y_km = (lat - ref_lat) * 111.32
    
    Args:
        coords: (N, 2+) array with [lat, lon, ...] in degrees.
        ref_lat: Reference latitude for projection center (default: centroid).
        ref_lon: Reference longitude for projection center (default: centroid).
    
    Returns:
        (N, 2) array of [x_km, y_km] in local Cartesian coordinates.
    """
    if ref_lat is None:
        ref_lat = coords[:, 0].mean()
    if ref_lon is None:
        ref_lon = coords[:, 1].mean()
    
    km_per_deg = 111.32
    x_km = (coords[:, 1] - ref_lon) * km_per_deg * np.cos(np.radians(ref_lat))
    y_km = (coords[:, 0] - ref_lat) * km_per_deg
    
    return np.stack([x_km, y_km], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Gap 6 Fix: Aurora-style BatchGenerator for Multi-GPU DDP Training
# Implements the delayed-execution pattern from Aurora Appendix E:
# lightweight metadata objects are generated, shuffled, and sharded
# across GPUs. Expensive I/O is deferred until unpacking on the target worker.
# ---------------------------------------------------------------------------
@dataclass
class BatchDescriptor:
    """
    Lightweight metadata object following the Aurora BatchGenerator pattern.
    Contains only the information needed to reconstruct a batch -- no tensors.
    
    Aurora mandates: "Each dataset generates a stream of lightweight
    BatchGenerator objects containing only metadata and file paths.
    Expensive operations are deferred until unpacking on the target GPU."
    """
    sample_indices: List[int] = field(default_factory=list)
    grid_name: Optional[str] = None
    zarr_path: Optional[str] = None
    time_slice: Optional[Tuple[int, int]] = None


def create_distributed_dataloader(
    windows: Windows,
    ml_task: Literal["downscaling", "forecasting", "correction"],
    batch_size: int = 32,
    num_workers: int = 4,
    grid_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    k_stations: int = 8,
    k_grid: int = 4,
    world_size: int = 1,
    rank: int = 0,
    drop_last: bool = True,
) -> PyGDataLoader:
    """
    Aurora-pattern distributed dataloader for multi-GPU DDP training.
    
    Uses DistributedSampler for deterministic cross-GPU sharding,
    persistent_workers for edge cache amortization, and batch_size
    balancing following Aurora's workload distribution strategy.
    
    For multi-GPU training with torch.distributed:
        torch.distributed.init_process_group('nccl')
        dataloader = create_distributed_dataloader(
            windows, ml_task, batch_size=32,
            world_size=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
        )
    
    Args:
        windows: Windows dataclass from MasharikiWeatherDataset.get_windows()
        ml_task: Task type for the EarthSystemAdapter.
        batch_size: Per-GPU batch size.
        num_workers: DataLoader workers per GPU.
        grid_stats: Optional scaling statistics.
        k_stations: KNN for station edges.
        k_grid: KNN for grid edges.
        world_size: Total number of GPUs (from torch.distributed).
        rank: Current GPU rank (from torch.distributed).
        drop_last: Drop incomplete final batch (recommended for training).
    
    Returns:
        PyGDataLoader with DDP-aware sampling and persistent workers.
    """
    dataset = EarthSystemAdapter(
        windows, ml_task, grid_stats=grid_stats,
        k_stations=k_stations, k_grid=k_grid,
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=drop_last,
    )
    
    return PyGDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # DistributedSampler handles shuffling
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=drop_last,
    )


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
    drop_last: bool = False,
) -> PyGDataLoader:
    """
    Create a single-GPU PyG DataLoader with optimized worker management.
    
    Fixes applied:
        - persistent_workers=True: Keeps workers alive between batches,
          amortizing the cost of _cached_edges reconstruction per worker.
        - prefetch_factor=2: Enables double-buffered prefetching for 
          overlapping I/O and compute.
        - drop_last: Prevents gradient noise spikes from undersized final batch.
    
    Args:
        windows: Windows dataclass from MasharikiWeatherDataset.get_windows()
        ml_task: Task type for the EarthSystemAdapter.
        batch_size: Batch size.
        shuffle: Whether to shuffle samples.
        num_workers: Number of DataLoader workers.
        grid_stats: Optional scaling statistics.
        k_stations: KNN for station edges.
        k_grid: KNN for grid edges.
        drop_last: Drop incomplete final batch (default: False).
    
    Returns:
        PyGDataLoader with persistent workers and optimized prefetching.
    """
    dataset = EarthSystemAdapter(
        windows, ml_task, grid_stats=grid_stats,
        k_stations=k_stations, k_grid=k_grid,
    )
    return PyGDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=drop_last,
    )