import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from masharikiweather.dataset import MasharikiWeatherDataset


@pytest.fixture
def synthetic_topography_zarr(tmp_path: Path) -> Path:
    """
    Create a synthetic regular-grid Zarr dataset modeling the 17 African
    topography variables at high spatial resolution.

    Mathematical docstring:
        Grid dimensions: lat=10, lon=10 covering equatorial East Africa [-1.0, 1.0] lat, [35.0, 37.0] lon.
        Data variables: dem, Slope, Aspect, LandCover, DistanceToWater, TPI_2000M, STD_2000M, etc.
    """
    lats = np.linspace(1.0, -1.0, 10)
    lons = np.linspace(35.0, 37.0, 10)

    data_vars = {
        "dem": (["lat", "lon"], (1500.0 + 300.0 * np.outer(np.sin(lats), np.cos(lons))).astype(np.float32)),
        "Slope": (["lat", "lon"], np.abs(5.0 * np.outer(np.cos(lats), np.sin(lons))).astype(np.float32)),
        "Aspect": (["lat", "lon"], (180.0 + 170.0 * np.outer(np.sin(lats), np.sin(lons))).astype(np.float32)),
        "LandCover": (["lat", "lon"], np.random.choice([10, 20, 30, 40, 50], size=(10, 10)).astype(np.float32)),
        "DistanceToWater": (["lat", "lon"], np.full((10, 10), 12500.0, dtype=np.float32)),
        "ASPECT_2000M_SIGRATIO1": (["lat", "lon"], (120.0 + 50.0 * np.outer(np.cos(lats), np.cos(lons))).astype(np.float32)),
        "ASPECT_10000M_SIGRATIO1": (["lat", "lon"], (130.0 + 40.0 * np.outer(np.cos(lats), np.cos(lons))).astype(np.float32)),
        "SLOPE_2000M_SIGRATIO1": (["lat", "lon"], np.full((10, 10), 2.5, dtype=np.float32)),
        "SLOPE_10000M_SIGRATIO1": (["lat", "lon"], np.full((10, 10), 1.8, dtype=np.float32)),
        "SN_DERIVATIVE_2000M_SIGRATIO1": (["lat", "lon"], np.full((10, 10), -0.015, dtype=np.float32)),
        "SN_DERIVATIVE_10000M_SIGRATIO1": (["lat", "lon"], np.full((10, 10), -0.010, dtype=np.float32)),
        "WE_DERIVATIVE_2000M_SIGRATIO1": (["lat", "lon"], np.full((10, 10), 0.020, dtype=np.float32)),
        "WE_DERIVATIVE_10000M_SIGRATIO1": (["lat", "lon"], np.full((10, 10), 0.012, dtype=np.float32)),
        "STD_2000M": (["lat", "lon"], np.full((10, 10), 45.0, dtype=np.float32)),
        "STD_10000M": (["lat", "lon"], np.full((10, 10), 110.0, dtype=np.float32)),
        "TPI_2000M": (["lat", "lon"], np.full((10, 10), 12.0, dtype=np.float32)),
        "TPI_10000M": (["lat", "lon"], np.full((10, 10), -25.0, dtype=np.float32)),
    }

    ds = xr.Dataset(data_vars=data_vars, coords={"lat": lats, "lon": lons})
    topo_dir = tmp_path / "topography"
    topo_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = topo_dir / "africa_topography.zarr"
    ds.to_zarr(zarr_path, consolidated=True)
    return zarr_path


@pytest.fixture
def mock_dataset_with_stations(tmp_path: Path) -> MasharikiWeatherDataset:
    """Construct a minimal dataset instance with stations inventory."""
    obs_dir = tmp_path / "obs" / "tahmo"
    meta_dir = tmp_path / "metadata"
    obs_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.date_range("2023-01-01", periods=10, freq="h", tz="UTC")
    cols = pd.MultiIndex.from_tuples([
        ("TA00001", "2m_temperature"),
        ("TA00001", "total_precipitation"),
        ("TA00002", "2m_temperature"),
        ("TA00002", "total_precipitation"),
    ])
    df = pd.DataFrame(np.ones((10, 4), dtype=np.float32), index=idx, columns=cols)
    flat_df = df.stack(level=0, future_stack=True).reset_index().rename(columns={"level_1": "station_id"})
    flat_df.to_parquet(obs_dir / "2023.parquet")

    inst_df = pd.DataFrame(
        {
            "latitude": [-0.42, 0.05],
            "longitude": [36.05, 35.80],
            "network": ["tahmo", "tahmo"],
            "station_name": ["Nyeri Station", "Rift Valley Station"],
        },
        index=pd.Index(["TA00001", "TA00002"], name="station_id"),
    )
    inst_df.to_parquet(meta_dir / "installation.parquet")

    param_df = pd.DataFrame(
        {"standard_name": ["air_temperature", "precipitation_amount"], "units": ["degC", "mm"]},
        index=pd.Index(["2m_temperature", "total_precipitation"], name="parameter"),
    )
    param_df.to_parquet(meta_dir / "parameters.parquet")

    return MasharikiWeatherDataset(
        root=tmp_path,
        years=[2023],
        source_obs="tahmo",
        use_hf=False,
    )


class TestTopographyOOPInterface:
    """Verification of topography attributes and accessors."""

    def test_available_topography_property(self, mock_dataset_with_stations: MasharikiWeatherDataset) -> None:
        """Verify available_topography contains all 17 multi-scale terrain descriptor variables."""
        ds = mock_dataset_with_stations
        avail = ds.available_topography

        assert isinstance(avail, set)
        expected_channels = {
            "dem",
            "Slope",
            "Aspect",
            "LandCover",
            "DistanceToWater",
            "ASPECT_2000M_SIGRATIO1",
            "ASPECT_10000M_SIGRATIO1",
            "SLOPE_2000M_SIGRATIO1",
            "SLOPE_10000M_SIGRATIO1",
            "SN_DERIVATIVE_2000M_SIGRATIO1",
            "SN_DERIVATIVE_10000M_SIGRATIO1",
            "WE_DERIVATIVE_2000M_SIGRATIO1",
            "WE_DERIVATIVE_10000M_SIGRATIO1",
            "STD_2000M",
            "STD_10000M",
            "TPI_2000M",
            "TPI_10000M",
        }
        for ch in expected_channels:
            assert ch in avail, f"Expected channel {ch} missing from available_topography."

    def test_load_local_topography_zarr(
        self,
        mock_dataset_with_stations: MasharikiWeatherDataset,
        synthetic_topography_zarr: Path,
    ) -> None:
        """Verify loading local Zarr topography applies cyclical aspect encoding and preserves fields."""
        ds = mock_dataset_with_stations
        topo_ds = ds.load_topography()

        assert topo_ds is not None
        assert isinstance(topo_ds, xr.Dataset)
        assert "dem" in topo_ds.data_vars
        assert "Slope" in topo_ds.data_vars
        assert "TPI_2000M" in topo_ds.data_vars

        # Cyclical encoding verification
        assert "Aspect" in topo_ds.data_vars
        assert "Aspect_sin" in topo_ds.data_vars
        assert "Aspect_cos" in topo_ds.data_vars
        assert "ASPECT_2000M_SIGRATIO1_sin" in topo_ds.data_vars
        assert "ASPECT_2000M_SIGRATIO1_cos" in topo_ds.data_vars

        # Mathematical sanity: sin^2 + cos^2 == 1
        sin_val = topo_ds["Aspect_sin"].values
        cos_val = topo_ds["Aspect_cos"].values
        np.testing.assert_allclose(sin_val**2 + cos_val**2, 1.0, rtol=1e-5)

    def test_interpolate_topography_onto_stations(
        self,
        mock_dataset_with_stations: MasharikiWeatherDataset,
        synthetic_topography_zarr: Path,
    ) -> None:
        """Verify interpolation populates station metadata table with terrain descriptors."""
        ds = mock_dataset_with_stations
        ds.load_topography(merge_stations=True)

        st_df = ds.stations_table
        assert isinstance(st_df, pd.DataFrame)
        assert "TA00001" in st_df.index
        assert "TA00002" in st_df.index

        # Verify key topography columns exist in station table
        assert "dem" in st_df.columns
        assert "Slope" in st_df.columns
        assert "Aspect" in st_df.columns
        assert "Aspect_sin" in st_df.columns
        assert "Aspect_cos" in st_df.columns
        assert "LandCover" in st_df.columns
        assert "TPI_2000M" in st_df.columns
        assert "DistanceToWater" in st_df.columns

        # Verify elevation and dem are synchronized
        assert "elevation" in st_df.columns
        np.testing.assert_allclose(st_df["elevation"].values, st_df["dem"].values, rtol=1e-5)

        # Installation property mirrors stations_table
        inst_df = ds.installation
        assert "dem" in inst_df.columns
        assert "TPI_2000M" in inst_df.columns

    def test_get_topography_data_method(
        self,
        mock_dataset_with_stations: MasharikiWeatherDataset,
        synthetic_topography_zarr: Path,
    ) -> None:
        """Verify get_topography_data queries variables and spatial bounding boxes."""
        ds = mock_dataset_with_stations

        # Query specific variable subset
        sub_ds = ds.get_topography_data(variables=["dem", "Slope"])
        assert isinstance(sub_ds, xr.Dataset)
        assert "dem" in sub_ds.data_vars
        assert "Slope" in sub_ds.data_vars
        assert "DistanceToWater" not in sub_ds.data_vars

        # Query bounding box slice
        bbox = [35.5, -0.5, 36.5, 0.5]
        crop_ds = ds.get_topography_data(variables=["dem"], bbox=bbox)
        lat_vals = crop_ds["lat"].values
        lon_vals = crop_ds["lon"].values
        assert lon_vals.min() >= 35.0
        assert lon_vals.max() <= 37.0
        assert len(lat_vals) > 0

    def test_extended_topo_vars_constructor_filtering(
        self,
        tmp_path: Path,
        synthetic_topography_zarr: Path,
    ) -> None:
        """Verify constructor parameter extended_topo_vars controls variable selection."""
        # Case 1: Specific list of variables
        ds_subset = MasharikiWeatherDataset(
            root=tmp_path,
            years=[2023],
            source_obs="tahmo",
            use_hf=False,
            extended_topo_vars=["dem", "TPI_2000M"],
        )
        ds_subset.load_topography()
        assert "dem" in ds_subset.topography.data_vars
        assert "TPI_2000M" in ds_subset.topography.data_vars
        assert "DistanceToWater" not in ds_subset.topography.data_vars

        # Case 2: 'all' preserves all channels
        ds_all = MasharikiWeatherDataset(
            root=tmp_path,
            years=[2023],
            source_obs="tahmo",
            use_hf=False,
            extended_topo_vars="all",
        )
        ds_all.load_topography()
        assert "DistanceToWater" in ds_all.topography.data_vars
        assert "SLOPE_10000M_SIGRATIO1" in ds_all.topography.data_vars

    def test_get_windows_incorporates_topo_static_features(
        self,
        mock_dataset_with_stations: MasharikiWeatherDataset,
        synthetic_topography_zarr: Path,
    ) -> None:
        """Verify window extraction automatically maps static topography into static_features."""
        ds = mock_dataset_with_stations
        ds.load_topography(merge_stations=True)

        windows = ds.get_windows(
            window_size=3,
            horizon_size=2,
            stations=["TA00001", "TA00002"],
            parameters=["2m_temperature"],
        )

        assert hasattr(windows, "static_features")
        assert hasattr(windows, "static_feature_labels")
        assert "dem" in windows.static_feature_labels
        assert "Slope" in windows.static_feature_labels
        assert "TPI_2000M" in windows.static_feature_labels
        assert windows.static_features.shape[0] == 2  # 2 stations
        assert windows.static_features.shape[1] == len(windows.static_feature_labels)


class TestRemoteHuggingFaceTopographyStreaming:
    """Verification of remote Hugging Face streaming for africa_topography.zarr."""

    def test_remote_huggingface_zarr_connection(self) -> None:
        """Verify lazy opening and chunk sampling of africa_topography.zarr from Hugging Face."""
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            pytest.skip("HF_TOKEN not set; skipping live remote streaming test.")

        try:
            import fsspec
            fs = fsspec.filesystem("hf", token=token)
            hf_path = "datasets/DeKUT-DSAIL/weather-data/data/topography/africa_topography.zarr"
            mapper = fs.get_mapper(hf_path)
            ds = xr.open_zarr(mapper, consolidated=False)
        except Exception as e:
            pytest.skip(f"Remote Hugging Face connection failed or unauthenticated: {e}")

        # Check coordinates and continental scope
        assert "lat" in ds.coords
        assert "lon" in ds.coords
        assert ds["lat"].values.min() <= -34.0  # South Africa
        assert ds["lat"].values.max() >= 35.0   # North Africa
        assert ds["lon"].values.min() <= -17.0  # West Africa (Dakar)
        assert ds["lon"].values.max() >= 50.0   # East Africa (Somalia)

        # Check multi-scale channels exist
        assert "dem" in ds.data_vars
        assert "Slope" in ds.data_vars
        assert "Aspect" in ds.data_vars
        assert "TPI_2000M" in ds.data_vars
        assert "ASPECT_2000M_SIGRATIO1" in ds.data_vars

        # Sample a specific station location in Kenya (Nairobi: lat -1.286, lon 36.817)
        pt_lat = xr.DataArray([-1.286389], dims="station")
        pt_lon = xr.DataArray([36.817223], dims="station")
        sampled = ds[["dem", "Slope"]].interp(lat=pt_lat, lon=pt_lon, method="linear").compute()
        elevation = float(sampled["dem"].values[0])

        # Physical sanity: Nairobi elevation must be between 1600m and 1800m
        assert 1600.0 <= elevation <= 1800.0, f"Unphysical Nairobi elevation: {elevation} m"
