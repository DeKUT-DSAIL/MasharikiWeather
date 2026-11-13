"""
Unit and integration tests for multi-grid atmospheric bias correction task.

Validates Windows dataclass attributes, target node routing for bias correction,
symmetric retrospective windowing, and PyG HeteroData target tensor alignment.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Optional torch / PyG import
try:
    import torch
    from torch_geometric.data import HeteroData
    from masharikiweather.dataloader import create_dataloader

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from masharikiweather.dataset import MasharikiWeatherDataset, Windows


def test_bias_correction_windows_and_hetero_data() -> None:
    """
    Validate multi-grid bias correction scenario (ERA5 + IMERG + TAHMO -> correct ERA5).

    Mathematical docstring:
        Time range: 7 days hourly (145 timestamps).
        Stations: N_stations = 2.
        Parameters: C_params = 5:
            ['total_precipitation', 'temperature_2m', 'u_component_of_wind_10m',
             'v_component_of_wind_10m', 'total_column_water_vapour'].
        Window size: W = 24.
        Horizon size: H = 0.
        Expected batches: B = 145 - 24 + 1 = 121.
        Station input tensor: (121, 24, 2, 5).
        Target node: 'era5'.
    """
    params: List[str] = [
        "total_precipitation",
        "temperature_2m",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "total_column_water_vapour",
    ]

    ds = MasharikiWeatherDataset(
        token=os.getenv("HF_TOKEN"),
        years=[2023],
        source_obs=["tahmo"],
        parameters=params,
        freq="h",
        use_hf=False,
    )

    test_stations = ["TA00001", "TA00002"]
    ds.stations_table = pd.DataFrame(
        {
            "latitude": [-1.0, -1.5],
            "longitude": [36.8, 37.0],
            "network": ["tahmo", "tahmo"],
            "dem": [1650.0, 1820.0],
            "Slope": [4.0, 8.5],
            "Aspect_sin": [0.2, -0.7],
            "Aspect_cos": [0.97, 0.71],
            "LandCover": [2, 5],
            "DistanceToWater": [1200.0, 950.0],
            "TPI_2000M": [0.1, -0.2],
        },
        index=pd.Index(test_stations, name="station_id"),
    )
    ds.parameters_table = pd.DataFrame(index=params)
    times = pd.date_range("2023-05-01", "2023-05-07 23:00:00", freq="h")
    cols = pd.MultiIndex.from_product([test_stations, params], names=["station_id", "parameter"])
    rng = np.random.default_rng(42)
    obs_data = rng.standard_normal((len(times), len(cols))).astype(np.float32)
    obs_data[:, 0::5] = np.abs(obs_data[:, 0::5])  # Precipitation non-negativity
    ds.observations = pd.DataFrame(obs_data, index=times, columns=cols)
    ds.mask = ~ds.observations.isna()

    def mock_grid(group: str, year: int, lazy: bool = True) -> xr.Dataset:
        g_times = pd.date_range(f"{year}-05-01", f"{year}-05-08", freq="h")
        lats = np.linspace(-3.0, 1.0, 10, dtype=np.float32)
        lons = np.linspace(35.0, 39.0, 10, dtype=np.float32)
        data = {
            p: (("time", "lat", "lon"), rng.standard_normal((len(g_times), len(lats), len(lons))).astype(np.float32))
            for p in params
        }
        data["total_precipitation"] = (
            ("time", "lat", "lon"),
            np.abs(rng.standard_normal((len(g_times), len(lats), len(lons)))).astype(np.float32),
        )
        return xr.Dataset(data, coords={"time": g_times, "lat": lats, "lon": lons})

    ds._open_gridded_zarr = mock_grid

    first_date = "2023-05-01"
    last_date = "2023-05-07"

    # Extract Windows under bias correction configuration
    win = ds.get_windows(
        window_size=24,
        horizon_size=0,
        first_date=first_date,
        last_date=last_date,
        gridded_url=["era5", "imerg"],
        spatial_mode="grid",
        ml_task="correction",
        target_node="era5",
        crop_to_stations=True,
    )

    # 1. Assertions on Windows dataclass
    assert isinstance(win, Windows)
    assert win.target_node == "era5"
    assert set(win.grids_x.keys()) == {"era5", "imerg"}

    # Grid feature channels
    assert win.grids_features["era5"] == params
    assert win.grids_features["imerg"] == params

    # Tensor shape checks: 145 hours total -> 145 - 24 + 1 = 121 batches
    expected_b = len(win.index_x)
    assert expected_b == 121
    assert win.station_x.ndim == 4
    assert win.station_x.shape == (expected_b, 24, 2, 5)
    assert win.station_mask_x.shape == win.station_x.shape
    assert win.delay == 0
    assert len(win.valid_batch_indices) == expected_b

    # Station coordinates for bipartite graph coupling
    assert win.station_coords is not None
    assert win.station_coords.shape == (2, 2)

    # 2. PyTorch Geometric HeteroData Assertions (Executed if Torch/PyG available)
    if not HAS_TORCH:
        pytest.skip("Torch or PyG not installed. Out-of-core Windows extraction validated.")

    loader = create_dataloader(win, ml_task="correction", batch_size=1, k_stations=8, k_grid=4)
    batch = next(iter(loader))

    assert "tahmo" in batch.node_types
    assert "era5" in batch.node_types
    assert "imerg" in batch.node_types

    assert len(batch.edge_types) > 0
    for edge_type in batch.edge_types:
        ei = batch[edge_type].edge_index
        ea = batch[edge_type].edge_attr
        assert ei.dim() == 2 and ei.shape[0] == 2
        assert ea.dim() == 2 and ea.shape[0] == ei.shape[1]

    # Verify that the target node 'era5' contains the correction target tensor 'y'
    assert hasattr(batch["era5"], "y")
    assert batch["era5"].y is not None
    assert batch["era5"].y.shape[0] > 0


def test_correction_symmetric_windowing() -> None:
    """
    Validate symmetric retrospective windowing for bias correction when horizon_size=1.

    Mathematical docstring:
        Context window: W = 24. Midpoint: mid = 12.
        The target Y is the midpoint observation:
            Y[b] == X[b, 12].
        Station Y shape: (B, 1, N_stations, C_params).
    """
    ds = MasharikiWeatherDataset(
        years=[2023],
        source_obs=["tahmo"],
        parameters=["total_precipitation"],
        freq="h",
        use_hf=False,
    )

    test_stations = ["TA00001", "TA00002"]
    ds.stations_table = pd.DataFrame(
        {"latitude": [-1.0, -1.5], "longitude": [36.8, 37.0]},
        index=pd.Index(test_stations, name="station_id"),
    )
    ds.parameters_table = pd.DataFrame(index=["total_precipitation"])
    times = pd.date_range("2023-05-01 00:00:00", periods=72, freq="h")
    cols = pd.MultiIndex.from_product([test_stations, ["total_precipitation"]], names=["station_id", "parameter"])
    obs_data = np.ones((len(times), len(cols)), dtype=np.float32)
    ds.observations = pd.DataFrame(obs_data, index=times, columns=cols)
    ds.mask = ~ds.observations.isna()

    win = ds.get_windows(
        window_size=24,
        horizon_size=1,
        ml_task="correction",
        spatial_mode="stations",
        target_node="tahmo",
    )

    expected_batches = 72 - 24 + 1
    assert expected_batches == 49
    assert win.station_x.shape == (expected_batches, 24, 2, 1)
    assert win.station_y.shape == (expected_batches, 1, 2, 1)
    assert win.index_y.shape == (expected_batches, 1)

    # Verify that Y timestamp corresponds to the window midpoint (index 12)
    for b in (0, 10, expected_batches - 1):
        mid_x_time = pd.to_datetime(win.index_x[b][12])
        y_time = pd.to_datetime(win.index_y[b][0])
        assert mid_x_time == y_time
