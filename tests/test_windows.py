"""
Unit and integration tests for MasharikiWeather sliding windows and the Windows dataclass.

Validates the N-Grid heterogeneous architecture, mathematical window slicing,
temporal delays, stride skip steps, target node routing, and geostatistical priors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from masharikiweather.dataset import MasharikiWeatherDataset, Windows


# ==============================================================================
# Synthetic Fixtures (Zero Network Calls, Completely Offline)
# ==============================================================================

@pytest.fixture
def synthetic_observations() -> pd.DataFrame:
    """
    Construct a contiguous hourly observation DataFrame for 3 stations and 2 parameters.

    Mathematical docstring:
        T_steps = 240 (10 contiguous days at hourly frequency).
        N_stations = 3 ('TA00001', 'TA00002', 'TA00003').
        C_params = 2 ('temperature_2m', 'total_precipitation').
        Output matrix shape: (240, 6) with MultiIndex columns (station_id, parameter).
    """
    times = pd.date_range("2023-01-01 00:00:00", periods=240, freq="h")
    stations = ["TA00001", "TA00002", "TA00003"]
    parameters = ["temperature_2m", "total_precipitation"]
    cols = pd.MultiIndex.from_product([stations, parameters], names=["station_id", "parameter"])

    rng = np.random.default_rng(12345)
    data = rng.standard_normal((len(times), len(cols))).astype(np.float32)
    # Ensure physical non-negativity for precipitation channels (indices 1, 3, 5)
    data[:, 1] = np.abs(data[:, 1])
    data[:, 3] = np.abs(data[:, 3])
    data[:, 5] = np.abs(data[:, 5])

    return pd.DataFrame(data, index=times, columns=cols)


@pytest.fixture
def synthetic_stations_metadata() -> pd.DataFrame:
    """Construct station metadata table with topographic priors and geographic coordinates."""
    stations = ["TA00001", "TA00002", "TA00003"]
    return pd.DataFrame(
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


@pytest.fixture
def synthetic_mock_grid_factory():
    """Factory creating synthetic CF-compliant xarray Gridded Datasets."""
    def _create_grid(group: str, year: int, lazy: bool = True) -> xr.Dataset:
        times = pd.date_range(f"{year}-01-01", f"{year}-01-15 23:00:00", freq="h")
        lats = np.linspace(-2.0, 1.0, 8, dtype=np.float32)
        lons = np.linspace(35.0, 38.0, 8, dtype=np.float32)

        rng = np.random.default_rng(42)
        temp_data = rng.standard_normal((len(times), len(lats), len(lons))).astype(np.float32)
        precip_data = np.abs(rng.standard_normal((len(times), len(lats), len(lons)))).astype(np.float32)

        return xr.Dataset(
            data_vars={
                "temperature_2m": (("time", "lat", "lon"), temp_data),
                "total_precipitation": (("time", "lat", "lon"), precip_data),
            },
            coords={"time": times, "lat": lats, "lon": lons},
        )
    return _create_grid


@pytest.fixture
def mock_dataset(synthetic_observations, synthetic_stations_metadata, synthetic_mock_grid_factory) -> MasharikiWeatherDataset:
    """Construct a fully isolated MasharikiWeatherDataset instance."""
    ds = MasharikiWeatherDataset(
        years=[2023],
        source_obs=["tahmo"],
        parameters=["temperature_2m", "total_precipitation"],
        freq="h",
        use_hf=False,
    )
    ds.stations_table = synthetic_stations_metadata.copy()
    ds.parameters_table = pd.DataFrame(index=["temperature_2m", "total_precipitation"])
    ds.observations = synthetic_observations.copy()
    ds.mask = ~synthetic_observations.isna()
    ds._open_gridded_zarr = synthetic_mock_grid_factory

    # Attach synthetic topography prior dataset
    topo_lats = np.linspace(-2.5, 1.5, 16, dtype=np.float32)
    topo_lons = np.linspace(34.5, 38.5, 16, dtype=np.float32)
    dem_grid = np.linspace(1000.0, 3000.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    slope_grid = np.full((16, 16), 5.0, dtype=np.float32)
    landcover_grid = np.full((16, 16), 3, dtype=np.int64)

    ds._topo_ds = xr.Dataset(
        data_vars={
            "dem": (("lat", "lon"), dem_grid),
            "Slope": (("lat", "lon"), slope_grid),
            "LandCover": (("lat", "lon"), landcover_grid),
        },
        coords={"lat": topo_lats, "lon": topo_lons},
    )
    return ds


# ==============================================================================
# TestWindows Test Suite
# ==============================================================================

class TestWindows:
    """Test suite verifying the Windows container and geostatistical extraction math."""

    def test_windows_dataclass_initialization(self) -> None:
        """
        Verify Windows container initialization, field assignments, and property aliases.

        Mathematical docstring:
            Input tensor shapes:
                station_x: (B=10, T_in=24, N_stations=3, C=2)
                station_y: (B=10, T_out=6, N_stations=3, C=2)
                native_topography: (C_static=2, H=16, W=16)
        """
        b, t_in, t_out, n_st, c_feat = 10, 24, 6, 3, 2
        sx = np.zeros((b, t_in, n_st, c_feat), dtype=np.float32)
        smx = np.ones((b, t_in, n_st, c_feat), dtype=bool)
        sy = np.ones((b, t_out, n_st, c_feat), dtype=np.float32)
        smy = np.ones((b, t_out, n_st, c_feat), dtype=bool)

        gx = {"era5": np.zeros((b, t_in, 10, 10), dtype=np.float32)}
        gy = {"era5": np.zeros((b, t_out, 10, 10), dtype=np.float32)}
        topo = np.zeros((2, 16, 16), dtype=np.float32)

        win = Windows(
            station_x=sx,
            station_mask_x=smx,
            station_y=sy,
            station_mask_y=smy,
            grids_x=gx,
            grids_y=gy,
            target_node="era5",
            native_topography=topo,
            delay=0,
        )

        # Direct attribute checks
        assert win.station_x.shape == (b, t_in, n_st, c_feat)
        assert win.station_mask_x.shape == (b, t_in, n_st, c_feat)
        assert win.station_y.shape == (b, t_out, n_st, c_feat)
        assert win.station_mask_y.shape == (b, t_out, n_st, c_feat)
        assert "era5" in win.grids_x
        assert "era5" in win.grids_y
        assert win.target_node == "era5"
        assert win.native_topography.shape == (2, 16, 16)

        # Property accessors
        assert np.array_equal(win.x, sx)
        assert np.array_equal(win.y, sy)
        assert np.array_equal(win.mask_x, smx)
        assert np.array_equal(win.mask_y, smy)
        assert win.gridded_x is not None

        # Property setters
        new_sx = np.ones((5, 12, 2, 1), dtype=np.float32)
        win.x = new_sx
        assert np.array_equal(win.station_x, new_sx)

        new_sy = np.ones((5, 4, 2, 1), dtype=np.float32)
        win.y = new_sy
        assert np.array_equal(win.station_y, new_sy)

    def test_sliding_window_mathematical_slicing(self, mock_dataset: MasharikiWeatherDataset) -> None:
        """
        Validate causal sliding window slicing dimensions and temporal alignment.

        Mathematical docstring:
            Total sequence length: N = 240 timesteps.
            Input window size: W = 24.
            Forecast horizon: H = 6.
            Delay: D = 0.
            Stride: S = 1.
            Expected batch count:
                B = N - W - D - H + 1 = 240 - 24 - 0 - 6 + 1 = 211.
            Shapes:
                station_x: (211, 24, 3, 2)
                station_y: (211, 6, 3, 2)
        """
        win = mock_dataset.get_windows(
            window_size=24,
            horizon_size=6,
            delay=0,
            stride=1,
            ml_task="forecasting",
            spatial_mode="stations",
            target_node="tahmo",
        )

        n_total = 240
        w, h, d, s = 24, 6, 0, 1
        expected_b = n_total - w - d - h + 1
        assert expected_b == 211

        assert win.station_x.shape == (expected_b, w, 3, 2)
        assert win.station_y.shape == (expected_b, h, 3, 2)
        assert win.station_mask_x.shape == (expected_b, w, 3, 2)
        assert win.station_mask_y.shape == (expected_b, h, 3, 2)
        assert win.index_x.shape == (expected_b, w)
        assert win.index_y.shape == (expected_b, h)
        assert len(win.valid_batch_indices) == expected_b
        assert np.array_equal(win.valid_batch_indices, np.arange(expected_b))

        # Check temporal continuity between input sequence end and target sequence start
        for b_idx in (0, 50, 100, expected_b - 1):
            last_x_time = pd.to_datetime(win.index_x[b_idx][-1])
            first_y_time = pd.to_datetime(win.index_y[b_idx][0])
            assert first_y_time - last_x_time == pd.Timedelta(hours=1)

    def test_sliding_window_delay_and_stride_math(self, mock_dataset: MasharikiWeatherDataset) -> None:
        """
        Validate causal window slicing with lead-time delay and strided stepping.

        Mathematical docstring:
            N = 240 timesteps.
            W = 24, H = 6, Delay D = 12, Stride S = 6.
            Expected batch indices:
                indices = np.arange(0, N - (W + D + H) + 1, S)
                max_start = 240 - 42 + 1 = 199 -> len(np.arange(0, 199, 6)) = 34.
            Temporal alignment check:
                Y_start - X_end = (D + 1) * delta_t = 13 hours.
                X_start[b+1] - X_start[b] = S * delta_t = 6 hours.
        """
        win = mock_dataset.get_windows(
            window_size=24,
            horizon_size=6,
            delay=12,
            stride=6,
            ml_task="forecasting",
            spatial_mode="stations",
            target_node="tahmo",
        )

        expected_batches = len(np.arange(0, 240 - 24 - 12 - 6 + 1, 6))
        assert expected_batches == 34

        assert win.station_x.shape == (expected_batches, 24, 3, 2)
        assert win.station_y.shape == (expected_batches, 6, 3, 2)
        assert len(win.valid_batch_indices) == expected_batches

        # Verify delay offset: day gap between X end and Y start
        t_x_last = pd.to_datetime(win.index_x[0][-1])
        t_y_first = pd.to_datetime(win.index_y[0][0])
        assert t_y_first - t_x_last == pd.Timedelta(hours=13)

        # Verify stride step between batch 0 and batch 1
        t_x_0 = pd.to_datetime(win.index_x[0][0])
        t_x_1 = pd.to_datetime(win.index_x[1][0])
        assert t_x_1 - t_x_0 == pd.Timedelta(hours=6)

    def test_edge_case_horizon_size_zero(self, mock_dataset: MasharikiWeatherDataset) -> None:
        """
        Test horizon_size=0 edge case used in feature extraction and bias correction.

        Mathematical docstring:
            N = 240, W = 24, H = 0, D = 0, S = 1.
            Expected batches: B = 240 - 24 + 1 = 217.
            Target array station_y must be empty.
        """
        win = mock_dataset.get_windows(
            window_size=24,
            horizon_size=0,
            delay=0,
            stride=1,
            ml_task="forecasting",
            spatial_mode="stations",
            target_node="tahmo",
        )

        expected_b = 240 - 24 + 1
        assert expected_b == 217
        assert win.station_x.shape == (expected_b, 24, 3, 2)
        assert len(win.station_y) == 0 or win.station_y.size == 0
        assert len(win.valid_batch_indices) == expected_b

    def test_edge_case_insufficient_timesteps(self, mock_dataset: MasharikiWeatherDataset) -> None:
        """Verify that requesting a window span larger than the dataset returns an empty Windows object."""
        win = mock_dataset.get_windows(
            window_size=200,
            horizon_size=50,
            delay=10,
            ml_task="forecasting",
            spatial_mode="stations",
        )
        # sample_span = 200 + 50 + 10 = 260 > 240 available timesteps
        assert len(win.station_x) == 0
        assert len(win.station_mask_x) == 0

    def test_multiple_grids_and_target_node_routing(self, mock_dataset: MasharikiWeatherDataset) -> None:
        """
        Verify multi-grid heterogeneous mapping and target node routing.

        Mathematical docstring:
            Gridded products: ['era5', 'imerg'].
            Spatial mode: 'grid' (native 2D field retention).
            Target routing: target_node='era5'.
        """
        win = mock_dataset.get_windows(
            window_size=12,
            horizon_size=4,
            first_date="2023-01-01",
            last_date="2023-01-08",
            gridded_url=["era5", "imerg"],
            spatial_mode="grid",
            ml_task="downscaling",
            target_node="era5",
        )

        assert set(win.grids_x.keys()) == {"era5", "imerg"}
        assert win.target_node == "era5"
        assert "temperature_2m" in win.grids_features["era5"]
        assert "total_precipitation" in win.grids_features["imerg"]
        assert "era5" in win.grids_lats and "era5" in win.grids_lons
        assert len(win.grids_lats["era5"]) == 8
        assert len(win.grids_lons["era5"]) == 8

    def test_topography_priors_extraction(self, mock_dataset: MasharikiWeatherDataset) -> None:
        """Verify continuous and categorical static topography priors are extracted."""
        win = mock_dataset.get_windows(
            window_size=12,
            horizon_size=4,
            gridded_url="era5",
            spatial_mode="grid",
            ml_task="downscaling",
            crop_to_stations=True,
        )

        assert win.native_topography is not None
        assert win.native_topography.ndim == 3  # (C_continuous, Lat, Lon)
        assert win.native_topography_categorical is not None
        assert win.native_topography_categorical.ndim == 3  # (C_categorical, Lat, Lon)
        assert win.station_static is not None
        assert win.station_coords is not None
        assert win.station_coords.shape == (3, 2)

    def test_bounding_box_filtering(self, mock_dataset: MasharikiWeatherDataset) -> None:
        """Verify spatial bounding box filtering restricts stations to coordinates within bbox."""
        # Stations latitudes: TA00001: -1.28, TA00002: -0.42, TA00003: 0.52
        # Target bbox covering only Southern Kenya: [-1.5, 36.5, -1.0, 37.0]
        bbox = (-1.5, 36.5, -1.0, 37.0)
        win = mock_dataset.get_windows(
            window_size=12,
            horizon_size=4,
            bbox=bbox,
            ml_task="forecasting",
            spatial_mode="stations",
        )

        # Only TA00001 (-1.28, 36.82) falls strictly within this box
        assert win.station_labels == ["TA00001"]
        assert win.station_x.shape[2] == 1  # Exactly 1 station retained
