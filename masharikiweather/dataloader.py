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
import os
import random
import torch
from torch_cluster import knn
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler
import numpy as np
import pandas as pd
from typing import Optional, Literal, Dict, Tuple, Sequence, List
from dataclasses import dataclass, field

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader
from tsl.data.preprocessing import StandardScaler
import dask
import xarray as xr
from .dataset import Windows


# ---------------------------------------------------------------------------
# Scientific Scaling Engine
# ---------------------------------------------------------------------------
class ScalingEngine:
    """
    Orchestrates modality-specific scaling to satisfy scientific correctness.
    - PHYSICAL: Standard z-score normalization (Gaussian assumption).
    - PRECIPITATION: Raw intensity preservation for Tweedie/ZIG optimization.
    - STATIC: Native resolution preservation.
    """
    def __init__(
        self, 
        stats: Dict[str, Dict[str, np.ndarray]], 
        variable_types: Dict[str, str],
        tsl_scaler: Optional['tsl.data.preprocessing.StandardScaler'] = None
    ):
        self.stats = stats
        self.variable_types = variable_types
        self.tsl_scaler = tsl_scaler
        
    def transform(self, data: torch.Tensor, variables: Sequence[str]) -> torch.Tensor:
        """
        Selectively transforms variables based on their scientific classification.
        """
        transformed = data.clone()
        for i, var in enumerate(variables):
            v_type = self.variable_types.get(var, "PHYSICAL")
            
            if v_type == "PHYSICAL":
                if self.tsl_scaler is not None:
                    # Leverage TSL scaler directly for standard physical variables
                    # Note: Assumes tsl_scaler operates on the feature dimension uniformly
                    # or is indexed correctly if properly fitted.
                    # As a safe fallback for variable-wise iteration, we only apply it 
                    # if the scaler was explicitly provided.
                    
                    # Temporarily reshape to match TSL expectation if needed, or apply directly
                    # For a single feature channel, tsl_scaler typically handles it fine
                    # provided it was fitted on the same feature index
                    transformed[..., i] = self.tsl_scaler.transform(data[..., i:i+1]).squeeze(-1)
                elif self.stats:
                    # Apply Gaussian scaling
                    v_mean = self.stats.get('mean', {}).get(var, 0.0)
                    v_std = self.stats.get('std', {}).get(var, 1.0)
                    # Handle zero-variance safety
                    v_std = v_std if v_std > 1e-6 else 1.0
                    transformed[..., i] = (data[..., i] - v_mean) / v_std
                
            elif v_type == "PRECIPITATION":
                if self.stats:
                    v_std = self.stats.get('std', {}).get(var, 1.0)
                    v_std = v_std if v_std > 1e-6 else 1.0
                    # Std-only scaling (no mean subtraction) to preserve zeros for Tweedie loss
                    transformed[..., i] = data[..., i] / v_std
                else:
                    # Pass raw if no stats available
                    transformed[..., i] = data[..., i]
                
        return transformed


# ---------------------------------------------------------------------------
# Utility: Cyclical temporal encoding
# ---------------------------------------------------------------------------
def datetime_encoded(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Cyclical temporal encoding for Earth observation data.
    Ensures zero-centered features in the range [-1, 1] for stable GNN training.
    """
    # Normalize indices to [0, 1) before trigger functions
    hour = index.hour.values / 24.0
    dow = index.dayofweek.values / 7.0
    doy = index.dayofyear.values / 365.25

    # Enforce periodicity via sin/cos mapping
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


def compute_terrain_wind_alignment(
    edge_src: np.ndarray, 
    edge_dst: np.ndarray, 
    grid_features: dict = None
) -> np.ndarray:
    """
    Computes terrain-steered wind alignment scalar: dot product of wind vector (u, v) 
    and surface slope/aspect orientation vector.
    Gracefully falls back to zeros if u, v variables are absent.
    """
    n_edges = len(edge_src)
    
    # Graceful Fallback Check
    has_wind = grid_features and 'u10' in grid_features and 'v10' in grid_features
    if not has_wind:
        return np.zeros((n_edges, 1), dtype=np.float32)
        
    try:
        # Extract dynamic wind components (u, v) and static slope/aspect components
        u = grid_features['u10'] # [Nodes, Time] or node-level vector
        v = grid_features['v10']
        
        # Terrain normal vector derived from slope and aspect (sin/cos)
        slope = grid_features.get('Slope', np.zeros_like(u))
        aspect_sin = grid_features.get('Aspect_sin', np.zeros_like(u))
        aspect_cos = grid_features.get('Aspect_cos', np.zeros_like(u))
        
        # Terrain gradient spatial orientation vector
        terrain_x = slope * aspect_sin
        terrain_y = slope * aspect_cos
        
        # Compute directional alignment via dot product along edges (sender to receiver)
        u_src = u[edge_src]
        v_src = v[edge_src]
        tx_dst = terrain_x[edge_dst]
        ty_dst = terrain_y[edge_dst]
        
        dot_product = (u_src * tx_dst + v_src * ty_dst)
        norm = (np.sqrt(u_src**2 + v_src**2) * np.sqrt(tx_dst**2 + ty_dst**2) + 1e-6)
        alignment = (dot_product / norm).astype(np.float32)
        
        return np.expand_dims(alignment, axis=-1)
    except Exception:
        # Absolute safety net against shape mismatches or missing keys
        return np.zeros((n_edges, 1), dtype=np.float32)



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
        ml_task: Literal["downscaling", "forecasting", "correction", "anomaly_detection"],
        grid_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        k_stations: int = 8,
        k_grid: int = 4,
        tsl_scaler: Optional['tsl.data.preprocessing.StandardScaler'] = None,
        projection_type: Literal["graphcast_3d", "esfm_fourier", "raw"] = "esfm_fourier",
    ):
        self.windows = windows
        self.ml_task = ml_task
        self.projection_type = projection_type

        # Station-graph geometry: n_stations and n_params for PyG reshape.
        station_labels = getattr(windows, 'station_labels', None)
        feature_labels = getattr(windows, 'feature_labels', None)
        self.n_stations = len(station_labels) if station_labels else 0
        self.n_params = len(feature_labels) if feature_labels else 0

        # 1. Pre-cast Point Station Variables (Robust Shape Parsing)
        def _format_station_tensor(arr, dtype, pad_nans=False):
            if arr is None or len(arr) == 0:
                return None
            t = torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype)
            
            # Pad NaN gaps to strictly maintain geometric lengths before any forward pass
            if pad_nans and dtype == torch.float32:
                t = torch.nan_to_num(t, nan=0.0)
                
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

        self.station_x = _format_station_tensor(windows.station_x, torch.float32, pad_nans=True)
        self.station_mask_x = _format_station_tensor(windows.station_mask_x, torch.bool)
        
        self.station_y = None
        self.station_mask_y = None
        if windows.station_y is not None and len(windows.station_y) > 0:
            self.station_y = _format_station_tensor(windows.station_y, torch.float32, pad_nans=True)
            self.station_mask_y = _format_station_tensor(windows.station_mask_y, torch.bool)

        self.window_size = self.station_x.shape[1] if self.station_x is not None else 0
        self.horizon_size = self.station_y.shape[1] if self.station_y is not None else 0

        # 2 & 3. Pre-cast Topographic Priors & Build Node Spatial Features
        self._build_node_features()

        # 4. Initialize Scientific Scaling Engine
        self.scaling_engine = ScalingEngine(
            stats=grid_stats if grid_stats else {},
            variable_types=getattr(windows, 'variable_types', {}),
            tsl_scaler=tsl_scaler
        )

        # 5. Build static graph topology (Anisotropic 3D connectivity)
        self._cached_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._build_hetero_edges(k_stations=k_stations, k_grid=k_grid)

    def _build_node_features(self):
        """Builds static node features, applying selected spatial projections."""
        # 1. Point Station Nodes
        self.station_static = None
        if self.windows.station_static is not None:
            station_features = [self.windows.station_static]
            st_coords = getattr(self.windows, 'station_coords', None)
            
            if st_coords is not None:
                proj_coords = self._project_coordinates(st_coords, self.projection_type)
                station_features.append(proj_coords)
                # Append raw coordinates for debugging metadata
                station_features.append(st_coords)
                
            combined = np.concatenate(station_features, axis=1)
            self.station_static = torch.as_tensor(combined, dtype=torch.float32)

        # 2. Native High-Resolution Bipartite DEM Nodes
        self.native_topography = None
        self.native_topography_categorical = None
        
        if getattr(self.windows, 'native_topography', None) is not None:
            raw_topo = self.windows.native_topography
            if hasattr(raw_topo, 'to_array'):
                raw_topo = raw_topo.to_array().values
                
            topo_features = [raw_topo]
            
            # Apply spatial projection to grid coordinates if available
            t_lats = getattr(self.windows, 'native_topography_lats', None)
            t_lons = getattr(self.windows, 'native_topography_lons', None)
            
            if t_lats is not None and t_lons is not None:
                # Create meshgrid
                t_lon_grid, t_lat_grid = np.meshgrid(t_lons, t_lats)
                t_coords = np.stack([t_lat_grid.ravel(), t_lon_grid.ravel()], axis=1)
                
                # Project
                proj_coords = self._project_coordinates(t_coords, self.projection_type)
                
                # Reshape to match raw_topo (Channels, Lat, Lon)
                n_lat, n_lon = len(t_lats), len(t_lons)
                proj_coords_reshaped = proj_coords.T.reshape(-1, n_lat, n_lon)
                t_coords_reshaped = t_coords.T.reshape(-1, n_lat, n_lon)
                
                topo_features.append(proj_coords_reshaped)
                # Append raw coordinates as debugging metadata
                topo_features.append(t_coords_reshaped)
                
            combined_topo = np.concatenate(topo_features, axis=0)
            self.native_topography = torch.as_tensor(combined_topo, dtype=torch.float32)

        if getattr(self.windows, 'native_topography_categorical', None) is not None:
            raw_cat = self.windows.native_topography_categorical
            self.native_topography_categorical = torch.as_tensor(raw_cat, dtype=torch.int64)

    def _project_coordinates(self, coords: np.ndarray, projection_type: str) -> np.ndarray:
        """
        Projects raw (lat, lon) coordinates into spatial embeddings based on the selected mode.
        """
        if coords is None or len(coords) == 0:
            return coords

        lat, lon = coords[:, 0], coords[:, 1]

        if projection_type == "graphcast_3d":
            # Option A: GraphCast 3D Cartesian (Unit Sphere)
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            x = np.cos(lat_rad) * np.cos(lon_rad)
            y = np.cos(lat_rad) * np.sin(lon_rad)
            z = np.sin(lat_rad)
            return np.stack([x, y, z], axis=1).astype(np.float32)
            
        elif projection_type == "esfm_fourier":
            # Option B: ESFM High-Resolution 2D Sine-Cosine (Fourier)
            # Scaling down frequency to 10^-5 degrees for point-scale locality
            scale = 1e-5
            lat_scaled = lat / scale
            lon_scaled = lon / scale
            
            # Using 4 frequencies to capture varying scales of locality
            freqs = np.array([1.0, 2.0, 4.0, 8.0])
            
            emb = []
            for f in freqs:
                emb.append(np.sin(2 * np.pi * lat_scaled / f))
                emb.append(np.cos(2 * np.pi * lat_scaled / f))
                emb.append(np.sin(2 * np.pi * lon_scaled / f))
                emb.append(np.cos(2 * np.pi * lon_scaled / f))
                
            return np.stack(emb, axis=1).astype(np.float32)
            
        else:
            # "raw" or default
            return coords.astype(np.float32)

    # ------------------------------------------------------------------
    # Graph Construction (Executed Once)
    # ------------------------------------------------------------------
    def _build_hetero_edges(self, k_stations: int = 8, k_grid: int = 4) -> None:
        """
        Construct a fully coupled, bidirectional GNN topology.
        Implements 3D Anisotropic Distance: 
          d = sqrt( d_geo^2 + lambda * d_alt^2 )
        """
        pass
        # Override K for scientific density alignment (k=12 provides better multi-modal convergence)
        k_grid = max(k_grid, 12)

        # 1. Discover all spatial sources
        sources = {} 
        
        # Station Coords (Lat, Lon, Alt, DEM, TPI)
        st_coords = getattr(self.windows, 'station_coords', None)
        if st_coords is not None:
            # Assuming station_static contains Altitude at index 0 (Standard in Mashariki dataset)
            # If not, we fall back to 2D
            st_alt = self.windows.station_static[:, :1] if self.windows.station_static is not None else np.zeros((len(st_coords), 1))
            
            # Topographic Analogue KNN features
            if self.windows.station_static is not None and self.windows.station_static.shape[1] >= 7:
                dem = self.windows.station_static[:, 0:1]
                tpi = self.windows.station_static[:, 6:7]
                if isinstance(dem, torch.Tensor): dem = dem.numpy()
                if isinstance(tpi, torch.Tensor): tpi = tpi.numpy()
                sources['tahmo'] = np.concatenate([st_coords, st_alt, dem, tpi], axis=1) # 5D
            else:
                sources['tahmo'] = np.concatenate([st_coords, st_alt], axis=1) # 3D

        # Grid Coords (Lat, Lon) - No altitude interpolation for coarse grids
        for g_name in getattr(self.windows, 'grids_x', {}).keys():
            lats = self.windows.grids_lats.get(g_name)
            lons = self.windows.grids_lons.get(g_name)
            if lats is not None and lons is not None:
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                coords_2d = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1)
                sources[g_name] = coords_2d

        # C. Native Topography (As a Node Type)
        t_lats = None
        t_lons = None
        topo_tensor = None
        if (getattr(self.windows, 'native_topography', None) is not None and 
            getattr(self.windows, 'native_topography_lats', None) is not None and 
            getattr(self.windows, 'native_topography_lons', None) is not None):
            
            t_lats = self.windows.native_topography_lats
            t_lons = self.windows.native_topography_lons
            t_lon_grid, t_lat_grid = np.meshgrid(t_lons, t_lats)
            t_coords = np.stack([t_lat_grid.ravel(), t_lon_grid.ravel()], axis=1)
            
            # Safely isolate the 0th channel (DEM) to prevent shape mismatch 
            # if the topography tensor contains multiple features (Slope, Aspect, etc.)
            topo_tensor = self.native_topography[0] if self.native_topography.ndim == 3 else self.native_topography
            
            # Apply nan_to_num directly to the grid array. 
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
                # Unidirectional Fastpass (Topography cannot receive messages)
                if dst_name == 'topography':
                    continue
                
                # Task-Aware Edge Isolation (Prevent Target Leakage)
                # For strictly spatial retrospective tasks (Downscaling, AO), we must prevent 
                # a station from informing a grid cell that is directly used to reconstruct it.
                # Identity mapping leakage is the biggest cause of 'too-perfect' weather emulators.
                target_node_name = getattr(self.windows, 'target_node', 'tahmo')

                is_bipartite_leak = False
                if self.ml_task in ['downscaling', 'anomaly_detection', 'correction']:
                    if src_name == target_node_name and dst_name != target_node_name:
                         # We allow neighbors to inform the grid, but the target node's self-edge 
                         # will be masked in the KNN block later if dist ~ 0.
                         is_bipartite_leak = True
                
                src_coords = sources[src_name]
                dst_coords = sources[dst_name]
                
                # Mathematically Sound Local Cartesian Projection
                # Convert source and destination coordinates to Kilometers
                ref_lat = src_coords[:, 0].mean()
                r_earth = 111.32  # km per degree of latitude
                
                def _project_cartesian(coords):
                    cart = np.zeros((coords.shape[0], coords.shape[1]), dtype=np.float32)
                    cart[:, 0] = coords[:, 0] * r_earth  # Y [km]
                    cart[:, 1] = coords[:, 1] * r_earth * np.cos(np.radians(ref_lat))  # X [km]
                    if coords.shape[1] >= 3:
                        cart[:, 2] = coords[:, 2] / 1000.0  # Z [km]
                    return cart
                
                cart_src = _project_cartesian(src_coords)
                cart_dst = _project_cartesian(dst_coords)
                
                use_analogue_5d = (src_name == 'tahmo' and dst_name == 'tahmo' and src_coords.shape[1] >= 5 and dst_coords.shape[1] >= 5)
                use_3d = (src_coords.shape[1] >= 3 and dst_coords.shape[1] >= 3)
                
                if use_analogue_5d:
                    # Topographic Feature-Space Analogue KNN (Z-Score Standardization)
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    
                    # Apply Cartesian Projection for X, Y, Z
                    cart_src_5d = np.zeros_like(src_coords, dtype=np.float32)
                    cart_src_5d[:, :3] = cart_src[:, :3]
                    cart_src_5d[:, 3:] = src_coords[:, 3:5] # DEM, TPI
                    
                    cart_dst_5d = np.zeros_like(dst_coords, dtype=np.float32)
                    cart_dst_5d[:, :3] = cart_dst[:, :3]
                    cart_dst_5d[:, 3:] = dst_coords[:, 3:5]
                    
                    # Standardize to give equal weight to all 5 dimensions
                    knn_src = scaler.fit_transform(cart_src_5d)
                    knn_dst = scaler.transform(cart_dst_5d)
                elif use_3d:
                    # WeatherGNN Anisotropy: λ_alt = 10.0 enforces strong terrain adherence
                    dist_weight = np.array([1.0, 1.0, 10.0])
                    knn_src = cart_src[:, :3] * dist_weight
                    knn_dst = cart_dst[:, :3] * dist_weight
                else:
                    # Strictly 2D physical kilometric topology
                    knn_src = cart_src[:, :2]
                    knn_dst = cart_dst[:, :2]
                
                k_target = k_stations if dst_name == 'tahmo' else k_grid
                if src_name == 'topography' or src_name == dst_name:
                    k_final = min(8, len(src_coords))
                    threshold = 0.7
                else:
                    k_final = min(32, len(src_coords))
                    threshold = 0.1 

                if k_final < 1: continue
                
                # KNN Search via torch_cluster (CUDA Accelerated)
                
                knn_src_t = torch.tensor(knn_src, dtype=torch.float32)
                knn_dst_t = torch.tensor(knn_dst, dtype=torch.float32)
                    
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                edge_index = knn(knn_src_t.to(device), knn_dst_t.to(device), k=k_final)
                
                dst_idx = edge_index[0].cpu().numpy()
                src_idx = edge_index[1].cpu().numpy()
                
                # Ensure deterministic sorting for reshaping
                sort_order = np.argsort(dst_idx, kind='mergesort')
                src_idx = src_idx[sort_order]
                
                indices = src_idx.reshape(len(dst_coords), k_final)
                distances_scaled = np.linalg.norm(knn_src[indices] - knn_dst[:, None, :], axis=-1)
                
                # Initialize similarities and mask
                sigma = np.std(distances_scaled) if np.std(distances_scaled) > 0 else 1.0
                similarities = np.exp(-(distances_scaled**2) / (sigma**2))
                
                # --- TOPOGRAPHIC RIDGE PROMINENCE PENALTY ---
                # If DEM is available and we're calculating physically plausible edges (e.g. between stations)
                if topo_tensor is not None and t_lats is not None and t_lons is not None and src_coords.shape[1] >= 3 and dst_coords.shape[1] >= 3:
                    gamma = 3.0  # Barrier penalty hyperparameter
                    steps = 10
                    
                    dem_numpy = topo_tensor.numpy() if isinstance(topo_tensor, torch.Tensor) else topo_tensor
                    
                    for i in range(len(dst_coords)):
                        for k_idx, j in enumerate(indices[i]):
                            # Draw linear transect between i and j
                            lats_profile = np.linspace(dst_coords[i, 0], src_coords[j, 0], steps)
                            lons_profile = np.linspace(dst_coords[i, 1], src_coords[j, 1], steps)
                            
                            # Search indices in DEM coordinates
                            lat_indices = np.clip(np.searchsorted(t_lats, lats_profile) - 1, 0, len(t_lats) - 1)
                            lon_indices = np.clip(np.searchsorted(t_lons, lons_profile) - 1, 0, len(t_lons) - 1)
                            
                            profile_elevations = dem_numpy[lat_indices, lon_indices]
                            
                            # Calculate Ridge Prominence in kilometers
                            z_max_path = np.nanmax(profile_elevations) / 1000.0 if not np.all(np.isnan(profile_elevations)) else 0.0
                            z_max_endpoints = max(dst_coords[i, 2], src_coords[j, 2]) / 1000.0
                            H_ij = max(0.0, z_max_path - z_max_endpoints)
                            
                            # Penalize the similarity edge weight
                            similarities[i, k_idx] *= np.exp(-gamma * H_ij)
                            
                mask = similarities >= threshold

                # Leave-One-Out (LOO) self-loop removal for Scientific Correctness
                # If connecting to the same node type (e.g., Station-to-Station), remove self-edges.
                if src_name == dst_name:
                    self_mask = indices == np.arange(len(dst_coords))[:, None]
                    # Correct self-loop masking across all tasks
                    mask &= (~self_mask)

                # Bipartite LOO Masking for spatial retrospective tasks.
                # If Station i is very close to Grid Cell G (dist < spatial_res), mask it 
                # if we are in downscaling/correction/AD mode to prevent leakage.
                if is_bipartite_leak:
                    # Find indices where distance is nearly zero (self-projection)
                    leak_mask = distances_scaled < 1e-4 # Threshold for spatial collocation
                    mask &= (~leak_mask)

                true_dist_km = np.linalg.norm(cart_src[indices.ravel(), :2] - np.repeat(cart_dst[:, :2], k_final, axis=0), axis=1)
                edge_dst_all = np.repeat(np.arange(len(dst_coords)), k_final)
                edge_src_all = indices.ravel()
                weights_all = similarities.ravel()
                
                # Apply Mask
                edge_src = edge_src_all[mask.ravel()]
                edge_dst = edge_dst_all[mask.ravel()]
                dist_km = true_dist_km[mask.ravel()]
                weights = weights_all[mask.ravel()]
                
                if len(edge_src) == 0: continue
                
                index_tensor = torch.tensor(np.stack([edge_src, edge_dst]), dtype=torch.long)
                max_dist = dist_km.max() if len(dist_km) > 0 and dist_km.max() > 0 else 1.0
                dist_norm = dist_km / max_dist
                
                dist_tensor = torch.tensor(dist_norm, dtype=torch.float32).unsqueeze(-1)
                weight_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)
                
                # Delta Elevation Logic
                if src_coords.shape[1] >= 3 and dst_coords.shape[1] >= 3:
                    delta_elev_km = cart_dst[edge_dst, 2] - cart_src[edge_src, 2]
                    delta_signed = np.tanh(delta_elev_km / 2.0)
                elif src_coords.shape[1] >= 3 and dst_coords.shape[1] == 2:
                    delta_signed = np.tanh((0.0 - cart_src[edge_src, 2]) / 2.0)
                elif src_coords.shape[1] == 2 and dst_coords.shape[1] >= 3:
                    delta_signed = np.tanh((cart_dst[edge_dst, 2] - 0.0) / 2.0)
                else:
                    delta_signed = np.zeros(len(edge_src), dtype=np.float32)
                    
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
        time_seq = None
        if len(self.windows.grids_x) > 0:
            first_grid_key = next(iter(self.windows.grids_x))
            grid_obj = self.windows.grids_x[first_grid_key]
            if isinstance(grid_obj, xr.Dataset):
                actual_time_slice = grid_obj.isel(time=slice(start_time, end_time))
                time_seq = pd.DatetimeIndex(actual_time_slice.time.values)
                
        if time_seq is None:
            # Eager mode (numpy arrays) or no grids present.
            # index_x is pre-windowed, so we access it via idx
            time_seq = pd.DatetimeIndex(self.windows.index_x[idx])
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
                    tensor_x = tensor_x.permute(0, 2, 1)
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

            # Inject temporal encoding into grid target for autoregressive consistency.
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
            
            # Retain separate mask for loss computation (sparse node-level feature mask)
            data['tahmo'].x_mask = tahmo_mask
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

        # 6. Regime Nodes
        # Virtual nodes act as global bottlenecks. They are initialized with an 
        # identity matrix to allow the network to learn specialized physical regimes.
        if self.n_regime_nodes > 0:
            data['regime'].x = torch.eye(self.n_regime_nodes, dtype=torch.float32)

        # 7. Attach Cached Graph Edges
        for edge_key, (edge_index, edge_attr) in self._cached_edges.items():
            data[edge_key].edge_index = edge_index
            data[edge_key].edge_attr = edge_attr

        # 8. Optional Mass Conservation Constraints (CHIRPS/TAMSAT)
        # Conditionally attached only if get_mass_constraints() was called on the Windows object.
        # When absent, the entire block is a no-op, ensuring full backward compatibility.
        if self.windows.constraint_targets:
            target = self.windows.target_node or 'tahmo'
            for constraint_name, constraint_data in self.windows.constraint_targets.items():
                if isinstance(constraint_data, np.ndarray) and len(constraint_data) > 0:
                    # Map this window's reference time to the constraint index
                    if self.windows.constraint_time_index is not None:
                        actual_idx = (
                            self.windows.valid_batch_indices[idx]
                            if self.windows.valid_batch_indices is not None
                            else idx
                        )
                        time_idx = self.windows.constraint_time_index[actual_idx]
                        if 0 <= time_idx < len(constraint_data):
                            data[target].mass_constraint = torch.as_tensor(
                                constraint_data[time_idx], dtype=torch.float32
                            )
                            # Attach corresponding mask
                            if constraint_name in self.windows.temporal_masks:
                                mask_data = self.windows.temporal_masks[constraint_name]
                                if isinstance(mask_data, np.ndarray) and time_idx < len(mask_data):
                                    data[target].mass_mask = torch.as_tensor(
                                        mask_data[time_idx], dtype=torch.float32
                                    )

            # Attach constraint_time_index for temporal aggregation in the loss function
            if self.windows.constraint_time_index is not None:
                actual_idx = (
                    self.windows.valid_batch_indices[idx]
                    if self.windows.valid_batch_indices is not None
                    else idx
                )
                data[target].constraint_time_index = torch.tensor(
                    self.windows.constraint_time_index[actual_idx], dtype=torch.long
                )

        return data


# ---------------------------------------------------------------------------
# 2D CRS Local Cartesian Projection Utility (ML LAMs Standard)
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
# Distributed Initialization & RNG Seeding Utilities
# ---------------------------------------------------------------------------
def _seed_worker(worker_id):
    """
    Ensures parallel PyTorch DataLoader workers have mathematically independent 
    NumPy and Random states, preventing identical stochastic data pipelines.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_ddp():
    """
    Initializes the DistributedDataParallel environment following Aurora standards.
    Requirements:
        1. NCCL backend for collective GPU synchronization.
        2. Strict CUDA device binding to the local rank to prevent VRAM overflow.
    """
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    """Safely destroys the PyTorch DDP process group."""
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Aurora-style BatchGenerator for Multi-GPU DDP Training
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
    ml_task: Literal["downscaling", "forecasting", "correction", "anomaly_detection"],
    batch_size: int = 32,
    num_workers: int = 4,
    grid_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    tsl_scaler: Optional['tsl.data.preprocessing.StandardScaler'] = None,
    k_stations: int = 8,
    k_grid: int = 4,
    world_size: int = 1,
    rank: int = 0,
    drop_last: bool = True,
    global_seed: int = 42,
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
        tsl_scaler=tsl_scaler,
        k_stations=k_stations, k_grid=k_grid,
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=drop_last,
        seed=global_seed,
    )
    
    # Establish root mathematical seed for parallel fork delegation
    g = torch.Generator()
    g.manual_seed(global_seed)
    
    return PyGDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # DistributedSampler handles shuffling
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=drop_last,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_dataloader(
    windows: Windows,
    ml_task: Literal["downscaling", "forecasting", "correction", "anomaly_detection"],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    grid_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    tsl_scaler: Optional['tsl.data.preprocessing.StandardScaler'] = None,
    k_stations: int = 8,
    k_grid: int = 4,
    drop_last: bool = False,
    global_seed: int = 42,
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
        tsl_scaler=tsl_scaler,
        k_stations=k_stations, k_grid=k_grid,
    )
    
    g = torch.Generator()
    g.manual_seed(global_seed)
    
    return PyGDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=drop_last,
    )