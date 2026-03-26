"""
PyTorch Geometric DataLoader mappings for Mashariki Weather Dataset.
Implementing the precise EarthSystemAdapter abstracting targeted TSL
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
from typing import Optional, Literal, Dict, Tuple

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader
from tsl.data.preprocessing import StandardScaler
from .dataset import Windows


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

        # 4. Out-of-Core Variable-Specific Scaling Setup
        self.scalers: Dict[str, StandardScaler] = {}
        if grid_stats is not None:
            for g_name, stats in grid_stats.items():
                scaler = StandardScaler()
                if hasattr(scaler, 'bias'):
                    scaler.bias = stats.get('mean', 0.0)
                    scaler.scale = stats.get('std', 1.0)
                else:
                    scaler.mean = stats.get('mean', 0.0)
                    scaler.std = stats.get('std', 1.0)
                self.scalers[g_name] = scaler

        # 5. Build static graph topology (executed exactly once)
        self._cached_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._build_hetero_edges(k_stations=k_stations, k_grid=k_grid)

    # ------------------------------------------------------------------
    # Graph Construction (Executed Once)
    # ------------------------------------------------------------------
    def _build_hetero_edges(self, k_stations: int = 8, k_grid: int = 4) -> None:
        """
        Construct all static heterogeneous graph edges and cache them.

        Implements two classes of edges:
            1. Homogeneous station-to-station with anisotropic edge_attr
               (inverse haversine distance, elevation/slope/aspect gradients).
            2. Bipartite grid-to-target for each grid in grids_x
               (KNN from flattened grid coordinates to target node coordinates).

        Scientific justification (from PeakWeather, WeatherGNN, observation-
        guided interpolation literature):
            Simple Euclidean distance fails to capture micro-climate boundaries.
            Two stations 5 km apart across a mountain ridge exist in entirely
            different precipitation regimes due to orographic lift and rain
            shadow effects. Encoding elevation gradients, slope differences,
            and aspect alignment as edge attributes allows the GNN to learn
            anisotropic message-passing weights that respect these physical
            boundaries.
        """
        from sklearn.neighbors import NearestNeighbors

        station_coords = getattr(self.windows, 'station_coords', None)
        if station_coords is None or len(station_coords) < 2:
            return  # No coordinates available; skip edge construction

        n_stations = len(station_coords)

        # ----- Resolve topographic feature indices dynamically -----
        feat_labels = list(self.windows.static_feature_labels or [])
        topo_idx = {}
        for name in ['DEM', 'Slope', 'Aspect_sin', 'Aspect_cos', 'TPI']:
            if name in feat_labels:
                topo_idx[name] = feat_labels.index(name)

        station_static_np = (
            self.windows.station_static
            if self.windows.station_static is not None
            else None
        )

        # =============================================================
        # A. Station-to-Station Homogeneous Anisotropic Edges
        # =============================================================
        actual_k = min(k_stations, n_stations - 1)
        if actual_k > 0:
            nn = NearestNeighbors(n_neighbors=actual_k + 1, metric='haversine')
            nn.fit(np.radians(station_coords))  # haversine expects radians
            distances_rad, indices = nn.kneighbors(np.radians(station_coords))

            # Convert haversine radians to km
            distances_km = distances_rad * 6371.0

            src_list, dst_list, attr_list = [], [], []
            eps = 1e-6

            for i in range(n_stations):
                for j_pos in range(1, actual_k + 1):  # skip self (index 0)
                    j = indices[i, j_pos]
                    d = distances_km[i, j_pos]

                    edge_feat = [1.0 / (d + eps)]  # inverse distance

                    if station_static_np is not None:
                        if 'DEM' in topo_idx:
                            dem_diff = float(
                                station_static_np[j, topo_idx['DEM']]
                                - station_static_np[i, topo_idx['DEM']]
                            )
                            edge_feat.append(dem_diff)
                        if 'Slope' in topo_idx:
                            slope_diff = float(
                                station_static_np[j, topo_idx['Slope']]
                                - station_static_np[i, topo_idx['Slope']]
                            )
                            edge_feat.append(slope_diff)
                        if 'Aspect_sin' in topo_idx:
                            asp_sin_diff = float(
                                station_static_np[j, topo_idx['Aspect_sin']]
                                - station_static_np[i, topo_idx['Aspect_sin']]
                            )
                            edge_feat.append(asp_sin_diff)
                        if 'Aspect_cos' in topo_idx:
                            asp_cos_diff = float(
                                station_static_np[j, topo_idx['Aspect_cos']]
                                - station_static_np[i, topo_idx['Aspect_cos']]
                            )
                            edge_feat.append(asp_cos_diff)

                    src_list.append(i)
                    dst_list.append(j)
                    attr_list.append(edge_feat)

            station_node_name = 'tahmo'  # derived from target_node default
            edge_key = (station_node_name, 'observes', station_node_name)
            self._cached_edges[edge_key] = (
                torch.tensor([src_list, dst_list], dtype=torch.long),
                torch.tensor(attr_list, dtype=torch.float32),
            )

        # =============================================================
        # B. Bipartite Downscaling Edges (Grid -> Target)
        # =============================================================
        target_node = self.windows.target_node or 'tahmo'

        # Resolve target coordinates
        if target_node == 'tahmo' or target_node not in self.windows.grids_lats:
            target_coords = station_coords  # (N_target, 2)
        else:
            t_lats = self.windows.grids_lats[target_node]
            t_lons = self.windows.grids_lons[target_node]
            t_lon_grid, t_lat_grid = np.meshgrid(t_lons, t_lats)
            target_coords = np.stack(
                [t_lat_grid.ravel(), t_lon_grid.ravel()], axis=1
            ).astype(np.float32)

        target_node_name = target_node if target_node != 'tahmo' else 'tahmo'

        for g_name in self.windows.grids_x.keys():
            g_lats = self.windows.grids_lats.get(g_name)
            g_lons = self.windows.grids_lons.get(g_name)
            if g_lats is None or g_lons is None:
                continue  # no spatial coords for this grid

            g_lon_grid, g_lat_grid = np.meshgrid(g_lons, g_lats)
            grid_coords = np.stack(
                [g_lat_grid.ravel(), g_lon_grid.ravel()], axis=1
            ).astype(np.float32)

            n_grid = len(grid_coords)
            n_target = len(target_coords)
            actual_k_g = min(k_grid, n_grid)
            if actual_k_g < 1:
                continue

            # Find K nearest grid nodes for each target node
            nn_g = NearestNeighbors(n_neighbors=actual_k_g, metric='haversine')
            nn_g.fit(np.radians(grid_coords))
            dists_rad, idxs = nn_g.kneighbors(np.radians(target_coords))
            dists_km = dists_rad * 6371.0

            src_list, dst_list, attr_list = [], [], []
            eps = 1e-6

            for t_i in range(n_target):
                for k_j in range(actual_k_g):
                    g_j = idxs[t_i, k_j]
                    d = dists_km[t_i, k_j]
                    edge_feat = [1.0 / (d + eps)]

                    # If we have DEM data for the target and can approximate
                    # grid elevation from native topography, add height diff.
                    # For station targets, DEM is in station_static.
                    if (
                        target_node_name == 'tahmo'
                        and station_static_np is not None
                        and 'DEM' in topo_idx
                        and t_i < station_static_np.shape[0]
                    ):
                        target_dem = float(station_static_np[t_i, topo_idx['DEM']])
                        edge_feat.append(target_dem)  # absolute target elevation as context

                    src_list.append(g_j)
                    dst_list.append(t_i)
                    attr_list.append(edge_feat)

            edge_key = (g_name, 'downscales_to', target_node_name)
            self._cached_edges[edge_key] = (
                torch.tensor([src_list, dst_list], dtype=torch.long),
                torch.tensor(attr_list, dtype=torch.float32),
            )

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

            # Apply variable-specific scaling if defined
            if g_name in self.scalers:
                try:
                    tensor_x = self.scalers[g_name].transform(tensor_x)
                except AttributeError:
                    bias = getattr(
                        self.scalers[g_name], 'bias',
                        getattr(self.scalers[g_name], 'mean', 0.0)
                    )
                    scale = getattr(
                        self.scalers[g_name], 'scale',
                        getattr(self.scalers[g_name], 'std', 1.0)
                    )
                    tensor_x = (tensor_x - bias) / scale

            data[g_name].x = tensor_x

        # 2. Universal Target Routing Modality
        target_node = self.windows.target_node

        if target_node == "tahmo" or target_node is None:
            if self.station_y is not None:
                data["tahmo"].y = self.station_y[idx]
                data["tahmo"].mask_y = self.station_mask_y[idx]
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

        # 3. Global Coordinates Mapping (Temporal Encodings)
        time_seq = pd.DatetimeIndex(self.windows.index_x[idx])
        encoded_time = datetime_encoded(time_seq)
        data['global'].u = torch.as_tensor(encoded_time, dtype=torch.float32)

        # 4. Point-Station Nodes
        data['tahmo'].x = self.station_x[idx]
        data['tahmo'].mask = self.station_mask_x[idx]
        if self.station_static is not None:
            data['tahmo'].static_u = self.station_static

        # 5. Native Topography Nodes
        if self.native_topography is not None:
            data['topography'].x = self.native_topography
        if self.native_topography_categorical is not None:
            data['topography'].x_categorical = self.native_topography_categorical

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