"""
Unit tests for Torch Spatiotemporal (TSL) mathematical sliding window logic.

Validates causal temporal windowing, lead-time delay shifting, stride stepping,
temporal index alignment, and lazy versus eager gridded extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Add project root to sys.path if not present
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from masharikiweather.dataset import MasharikiWeatherDataset, Windows


def create_mock_grid(year: int, freq: str = "1D") -> xr.Dataset:
    """Construct synthetic gridded xarray Dataset with spatial lat/lon coordinates."""
    times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq=freq)
    lats = np.linspace(-5, 5, 10, dtype=np.float32)
    lons = np.linspace(30, 40, 10, dtype=np.float32)
    data = np.random.rand(len(times), len(lats), len(lons)).astype(np.float32)
    return xr.Dataset(
        data_vars={"precipitation": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )


def create_mock_observations(
    stations: Sequence[str],
    start_date: str = "2023-01-01",
    end_date: str = "2023-01-15",
    freq: str = "D",
) -> pd.DataFrame:
    """Construct synthetic MultiIndex DataFrame for station point observations."""
    times = pd.date_range(start_date, end_date, freq=freq)
    cols = pd.MultiIndex.from_product([stations, ["total_precipitation"]], names=["station_id", "parameter"])
    data = np.random.rand(len(times), len(cols)).astype(np.float32)
    return pd.DataFrame(data, index=times, columns=cols)


def test_tsl_logic() -> None:
    """
    Validate causal sliding window indexing, delay shifting, and stride stepping.

    Mathematical docstring:
        Contiguous observations: N = 9 timesteps (2023-01-01 to 2023-01-10 daily).
        Stations: N_stations = 3.
        Features: C = 1 ('total_precipitation').

        - Case 1 (Standard):
            W = 3, H = 2, D = 0, S = 1.
            Expected batches: B = N - W - D - H + 1 = 9 - 3 - 0 - 2 + 1 = 5.
            station_x: (5, 3, 3, 1)
            station_y: (5, 2, 3, 1)

        - Case 2 (Delay):
            W = 3, H = 2, D = 2, S = 1.
            Expected batches: B = N - W - D - H + 1 = 9 - 3 - 2 - 2 + 1 = 3.
            station_x: (3, 3, 3, 1)
            station_y: (3, 2, 3, 1)
            Index alignment: Y_start - X_end = (D + 1) * delta_t = 3 days.

        - Case 3 (Stride):
            W = 3, H = 2, D = 0, S = 2.
            Expected batches: B = len(arange(0, 9 - 5 + 1, 2)) = 3.
            station_x: (3, 3, 3, 1)
            station_y: (3, 2, 3, 1)
            Stride step: X_start[b+1] - X_start[b] = S * delta_t = 2 days.
    """
    ds = MasharikiWeatherDataset(
        years=[2023],
        source_obs=["tahmo"],
        parameters=["total_precipitation"],
        freq="D",
        use_hf=False,
    )

    test_stations = ["TA00001", "TA00002", "TA00003"]
    ds.stations_table = pd.DataFrame(
        {"latitude": [-1.0, 0.0, 1.0], "longitude": [35.0, 36.0, 37.0]},
        index=pd.Index(test_stations, name="station_id"),
    )
    ds.parameters_table = pd.DataFrame(index=["total_precipitation"])
    ds.observations = create_mock_observations(test_stations, "2023-01-01", "2023-01-15", freq="D")
    ds.mask = ~ds.observations.isna()

    first_date = "2023-01-01"
    last_date = "2023-01-10"

    obs, mask = ds.get_observations(
        stations=test_stations,
        first_date=first_date,
        last_date=last_date,
        as_numpy=False,
        return_mask=True,
    )
    n_timesteps = len(obs)
    assert n_timesteps == 9

    # Case 1: Standard (W=3, H=2, delay=0, stride=1)
    w1 = ds.get_windows(
        window_size=3,
        horizon_size=2,
        delay=0,
        stride=1,
        lazy=True,
        stations=test_stations,
        first_date=first_date,
        last_date=last_date,
    )

    expected_b1 = n_timesteps - 3 - 0 - 2 + 1
    assert expected_b1 == 5
    assert w1.x.shape == (5, 3, 3, 1)
    assert w1.y.shape == (5, 2, 3, 1)
    assert w1.station_mask_x.shape == (5, 3, 3, 1)
    assert w1.station_mask_y.shape == (5, 2, 3, 1)
    assert len(w1.valid_batch_indices) == 5
    assert np.array_equal(w1.valid_batch_indices, np.arange(5))

    # Verify temporal index alignment for all batches in Case 1
    assert w1.index_x.shape == (5, 3)
    assert w1.index_y.shape == (5, 2)
    for b in range(5):
        t_x_end = pd.to_datetime(w1.index_x[b][-1])
        t_y_start = pd.to_datetime(w1.index_y[b][0])
        assert t_y_start - t_x_end == pd.Timedelta(days=1)

    # Case 2: Delay (W=3, H=2, delay=2, stride=1)
    w2 = ds.get_windows(
        window_size=3,
        horizon_size=2,
        delay=2,
        stride=1,
        lazy=True,
        stations=test_stations,
        first_date=first_date,
        last_date=last_date,
    )

    expected_b2 = n_timesteps - 3 - 2 - 2 + 1
    assert expected_b2 == 3
    assert w2.x.shape == (3, 3, 3, 1)
    assert w2.y.shape == (3, 2, 3, 1)
    assert len(w2.valid_batch_indices) == 3
    assert np.array_equal(w2.valid_batch_indices, np.arange(3))

    # Verify lead-time delay: Day 4 and Day 5 are skipped; Day 6 starts target sequence
    t2_x_end = pd.to_datetime(w2.index_x[0][-1])
    t2_y_start = pd.to_datetime(w2.index_y[0][0])
    assert t2_y_start - t2_x_end == pd.Timedelta(days=3)
    assert t2_x_end == pd.Timestamp("2023-01-03")
    assert t2_y_start == pd.Timestamp("2023-01-06")

    # Case 3: Stride (W=3, H=2, delay=0, stride=2)
    w3 = ds.get_windows(
        window_size=3,
        horizon_size=2,
        delay=0,
        stride=2,
        lazy=True,
        stations=test_stations,
        first_date=first_date,
        last_date=last_date,
    )

    expected_b3 = len(np.arange(0, n_timesteps - 3 - 0 - 2 + 1, 2))
    assert expected_b3 == 3
    assert w3.x.shape == (3, 3, 3, 1)
    assert w3.y.shape == (3, 2, 3, 1)
    assert len(w3.valid_batch_indices) == 3
    assert np.array_equal(w3.valid_batch_indices, np.array([0, 2, 4]))

    # Verify stride gap between consecutive mini-batches
    t3_x_0 = pd.to_datetime(w3.index_x[0][0])
    t3_x_1 = pd.to_datetime(w3.index_x[1][0])
    t3_x_2 = pd.to_datetime(w3.index_x[2][0])
    assert t3_x_1 - t3_x_0 == pd.Timedelta(days=2)
    assert t3_x_2 - t3_x_1 == pd.Timedelta(days=2)


def test_gridded_tsl_logic() -> None:
    """
    Validate N-Grid spatiotemporal extraction with lazy and eager configurations.

    Mathematical docstring:
        Grid dimensions:
            Time: 9 timesteps.
            Lat: 10 grid points.
            Lon: 10 grid points.
        Parameters:
            W = 3, H = 2, Delay D = 2, Stride S = 2.
            Expected batch indices: [0, 2] (length 2).
    """
    ds = MasharikiWeatherDataset(
        years=[2023],
        source_obs=["tahmo"],
        parameters=["total_precipitation"],
        freq="D",
        use_hf=False,
    )

    test_stations = ["TA00001", "TA00002", "TA00003"]
    ds.stations_table = pd.DataFrame(
        {"latitude": [-1.0, 0.0, 1.0], "longitude": [35.0, 36.0, 37.0]},
        index=pd.Index(test_stations, name="station_id"),
    )
    ds.parameters_table = pd.DataFrame(index=["total_precipitation"])
    ds.observations = create_mock_observations(test_stations, "2023-01-01", "2023-01-15", freq="D")
    ds.mask = ~ds.observations.isna()

    # Mock the zarr loader completely to dynamically prevent network access
    ds._open_gridded_zarr = lambda group, year, lazy=True: create_mock_grid(year, "1D")

    first_date = "2023-01-01"
    last_date = "2023-01-10"

    obs, mask = ds.get_observations(
        stations=test_stations,
        first_date=first_date,
        last_date=last_date,
        as_numpy=False,
        return_mask=True,
    )
    n_timesteps = len(obs)
    assert n_timesteps == 9

    # Lazy Grid
    w_lazy = ds.get_windows(
        window_size=3,
        horizon_size=2,
        delay=2,
        stride=2,
        lazy=True,
        stations=test_stations,
        first_date=first_date,
        last_date=last_date,
        gridded_url="era5",
        spatial_mode="grid",
        ml_task="downscaling",
    )

    expected_stride_batches = len(np.arange(0, n_timesteps - 3 - 2 - 2 + 1, 2))
    assert expected_stride_batches == 2
    assert len(w_lazy.valid_batch_indices) == 2
    assert np.array_equal(w_lazy.valid_batch_indices, np.array([0, 2]))
    assert isinstance(w_lazy.gridded_x, xr.Dataset)
    assert w_lazy.gridded_features == ["precipitation"]
    assert dict(w_lazy.gridded_x.sizes)["time"] == 9
    assert dict(w_lazy.gridded_x.sizes)["lat"] == 10
    assert dict(w_lazy.gridded_x.sizes)["lon"] == 10

    # Full Eager Grid Extraction (automatically mapped lazily in modern N-grid architecture)
    w_eager = ds.get_windows(
        window_size=3,
        horizon_size=2,
        delay=2,
        stride=2,
        lazy=False,
        stations=test_stations,
        first_date=first_date,
        last_date=last_date,
        gridded_url="era5",
        spatial_mode="grid",
        ml_task="downscaling",
    )

    assert isinstance(w_eager.gridded_x, xr.Dataset)
    assert dict(w_eager.gridded_x.sizes)["time"] == 9
    assert dict(w_eager.gridded_x.sizes)["lat"] == 10
    assert dict(w_eager.gridded_x.sizes)["lon"] == 10
    assert dict(w_eager.gridded_y.sizes)["time"] == 9
