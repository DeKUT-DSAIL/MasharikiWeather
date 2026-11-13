"""
Unit and integration tests for PyTorch Geometric HeteroData graph construction and DataLoader batching.

Validates node types, tensor feature shapes, static geographic attributes,
bipartite and anisotropic KNN edges, and PyG mini-batch collation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Skip test module gracefully if PyTorch or PyTorch Geometric are not installed
torch = pytest.importorskip("torch")
pyg = pytest.importorskip("torch_geometric")

from masharikiweather.dataloader import create_dataloader, EarthSystemAdapter
from masharikiweather.dataset import MasharikiWeatherDataset, Windows


def test_hetero_data_item_and_batch_creation() -> None:
    """
    Verify PyG HeteroData graph construction, node representations, and batch collation.

    Mathematical docstring:
        HeteroData Node Tensors:
            - Point Station Nodes ('tahmo'):
                x shape: (N_stations, T_in, C_features) where
                C_features = C_params + C_temporal_encodings + C_mask = 2 + 6 + 2 = 10.
                static_u shape: (N_stations, C_static).
                pos shape: (N_stations, 2) [latitude, longitude].
            - Macro Gridded Nodes ('era5', 'imerg'):
                x shape: (N_grid_nodes, T_in, C_grid_params + 6)
                pos shape: (N_grid_nodes, 2)
            - Graph Edge Topology:
                edge_index: (2, E)
                edge_attr: (E, D_edge) where edge features encode inverse distance,
                elevation gradients, and directional orientation.
            - DataLoader Mini-batch:
                Batch size: B = 2.
    """
    # 1. Construct isolated synthetic dataset
    ds = MasharikiWeatherDataset(
        years=[2023],
        source_obs=["tahmo"],
        parameters=["temperature_2m", "total_precipitation"],
        freq="h",
        use_hf=False,
    )

    stations = ["TA00001", "TA00002", "TA00003"]
    ds.stations_table = pd.DataFrame(
        {
            "latitude": [-1.28, -0.42, 0.52],
            "longitude": [36.82, 36.95, 35.28],
            "network": ["tahmo", "tahmo", "tahmo"],
            "dem": [1680.0, 1950.0, 2100.0],
            "Slope": [3.2, 7.8, 11.5],
            "Aspect_sin": [0.35, -0.82, 0.45],
            "Aspect_cos": [0.93, 0.57, -0.89],
            "LandCover": [2, 4, 1],
            "DistanceToWater": [1500.0, 420.0, 2800.0],
            "TPI_2000M": [0.12, -0.34, 0.65],
        },
        index=pd.Index(stations, name="station_id"),
    )
    ds.parameters_table = pd.DataFrame(index=["temperature_2m", "total_precipitation"])

    times = pd.date_range("2023-01-01 00:00:00", periods=120, freq="h")
    cols = pd.MultiIndex.from_product([stations, ["temperature_2m", "total_precipitation"]], names=["station_id", "parameter"])
    rng = np.random.default_rng(999)
    obs_data = rng.standard_normal((len(times), len(cols))).astype(np.float32)
    obs_data[:, 1::2] = np.abs(obs_data[:, 1::2])  # Precipitation non-negativity
    ds.observations = pd.DataFrame(obs_data, index=times, columns=cols)
    ds.mask = ~ds.observations.isna()

    def mock_grid(group: str, year: int, lazy: bool = True) -> xr.Dataset:
        g_times = pd.date_range(f"{year}-01-01", f"{year}-01-08", freq="h")
        lats = np.linspace(-2.0, 1.0, 6, dtype=np.float32)
        lons = np.linspace(35.0, 38.0, 6, dtype=np.float32)
        t_data = rng.standard_normal((len(g_times), len(lats), len(lons))).astype(np.float32)
        p_data = np.abs(rng.standard_normal((len(g_times), len(lats), len(lons)))).astype(np.float32)
        return xr.Dataset(
            data_vars={
                "temperature_2m": (("time", "lat", "lon"), t_data),
                "total_precipitation": (("time", "lat", "lon"), p_data),
            },
            coords={"time": g_times, "lat": lats, "lon": lons},
        )

    ds._open_gridded_zarr = mock_grid

    # 2. Extract Windows
    windows = ds.get_windows(
        window_size=4,
        horizon_size=2,
        first_date="2023-01-01",
        last_date="2023-01-05",
        gridded_url=["era5", "imerg"],
        spatial_mode="grid",
        ml_task="downscaling",
        target_node="tahmo",
        crop_to_stations=True,
    )

    assert isinstance(windows, Windows)
    assert len(windows.station_x) > 0

    # 3. Create PyG DataLoader
    loader = create_dataloader(
        windows,
        ml_task="downscaling",
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    # 4. Validate Single HeteroData Item
    hetero_data = loader.dataset[0]

    assert "tahmo" in hetero_data.node_types
    assert "era5" in hetero_data.node_types
    assert "imerg" in hetero_data.node_types

    # Station node feature shape: (N_stations=3, T_in=4, C=2 + 6 + 2 = 10)
    assert hetero_data["tahmo"].x.dim() == 3
    assert hetero_data["tahmo"].x.shape[0] == 3
    assert hetero_data["tahmo"].x.shape[1] == 4
    assert hetero_data["tahmo"].x.shape[2] == 10

    # Station static and spatial coordinates
    assert hasattr(hetero_data["tahmo"], "static_u")
    assert hetero_data["tahmo"].static_u.shape[0] == 3

    assert hasattr(hetero_data["tahmo"], "pos")
    assert hetero_data["tahmo"].pos.shape == (3, 2)

    # Validate graph edge topology
    assert len(hetero_data.edge_types) > 0
    for edge_type in hetero_data.edge_types:
        edge_index = hetero_data[edge_type].edge_index
        edge_attr = hetero_data[edge_type].edge_attr

        assert edge_index.dim() == 2
        assert edge_index.shape[0] == 2
        assert edge_attr.dim() == 2
        assert edge_attr.shape[0] == edge_index.shape[1]
        assert edge_index.min() >= 0

    # 5. Validate Mini-Batch Collation
    batch = next(iter(loader))
    assert batch.batch_size == 2
    assert "tahmo" in batch.node_types
    assert hasattr(batch["tahmo"], "batch")
