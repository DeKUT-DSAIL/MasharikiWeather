"""
Unit tests for MasharikiWeather unified data extraction modules.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from masharikiweather.extract.apis.base import (
    AFRICA_BBOX,
    KENYA_BBOX,
    SpatialBoundingBox,
    TemporalRange,
)
from masharikiweather.extract.apis.cds_insitu import (
    CDSInSituExtractor,
    canonicalize_cds_insitu_variable,
    cds_insitu_canonical_to_raw,
    parse_cds_insitu_dataframe,
    parse_cds_insitu_dataset,
)
from masharikiweather.extract.apis.credentials import (
    CredentialsError,
    CredentialsManager,
    _parse_cdsapirc,
    _parse_env_file,
    get_cds_api_key,
    get_credentials,
    parse_date,
    parse_date_range,
)
from masharikiweather.extract.apis.era5 import (
    CopernicusCDSError,
    ERA5Extractor,
    canonicalize_era5_variable,
    check_grib_dependencies,
    era5_canonical_to_short,
    flatten_accum_ds,
    generate_n_days_dict,
    harmonize_era5_dataset,
    open_era5_grib,
    open_era5_netcdf,
    resample_era5,
)
from masharikiweather.extract.apis.ghcnd import GHCNExtractor
from masharikiweather.extract.apis.imerg import (
    IMERGExtractor,
    IMERGResult,
    IMERG_LATE_RUN_NAME,
    IMERG_TRANSITION_DATE,
    _read_hdf5_granule,
    canonicalize_imerg_variable,
    harmonize_imerg_dataset,
    imerg_canonical_to_raw,
)
from masharikiweather.extract.apis.reanalysis import (
    ReanalysisExtractor,
    WMO_WEATHER_CODES,
)
from masharikiweather.extract.apis.satellite import SatelliteExtractor
from masharikiweather.extract.apis.tahmo import (
    DEFAULT_TAHMO_PARAMETERS,
    TAHMOExtractor,
    canonicalize_tahmo_variable,
    tahmo_canonical_to_short,
)


# ==============================================================================
# Credentials Tests
# ==============================================================================

class TestCredentialsManager:
    """Test BYOK credential discovery and error handling."""

    def test_parse_env_file(self, tmp_path: Path):
        env_file = tmp_path / "test.env"
        env_file.write_text("TAHMO_API_KEY=test_key_123\nTAHMO_API_SECRET=\"secret_456\"\n# Comment\nINVALID\n")
        parsed = _parse_env_file(env_file)
        assert parsed["TAHMO_API_KEY"] == "test_key_123"
        assert parsed["TAHMO_API_SECRET"] == "secret_456"

    def test_parse_cdsapirc(self, tmp_path: Path):
        rc_file = tmp_path / ".cdsapirc"
        rc_file.write_text("url: https://cds.climate.copernicus.eu/api\nkey: 12345:abcdef\n")
        parsed = _parse_cdsapirc(rc_file)
        assert parsed["url"] == "https://cds.climate.copernicus.eu/api"
        assert parsed["key"] == "12345:abcdef"

    def test_explicit_arguments_take_precedence(self) -> None:
        creds = CredentialsManager.get_tahmo_credentials(api_key="explicit_k", api_secret="explicit_s")
        assert creds == {"api_key": "explicit_k", "api_secret": "explicit_s"}

    def test_missing_credentials_raise_informative_error(self, monkeypatch, tmp_path: Path):
        monkeypatch.delenv("TAHMO_API_KEY", raising=False)
        monkeypatch.delenv("TAHMO_API_SECRET", raising=False)
        empty_env = tmp_path / "empty.env"
        empty_env.write_text("")

        with patch.object(CredentialsManager, "_get_candidate_env_paths", return_value=[empty_env]):
            with pytest.raises(CredentialsError) as exc_info:
                CredentialsManager.get_tahmo_credentials(env_file=empty_env)
            assert "Missing API key or API secret" in str(exc_info.value)
            assert len(exc_info.value.resolution_hints) > 0

    def test_unified_get_credentials_entrypoint(self) -> None:
        creds = get_credentials("tahmo", api_key="k", api_secret="s")
        assert creds["api_key"] == "k"

    def test_tahmo_env_variables(self, monkeypatch) -> None:
        monkeypatch.setenv("TAHMO_API_KEY", "env_key")
        monkeypatch.setenv("TAHMO_API_SECRET", "env_secret")
        creds = CredentialsManager.get_tahmo_credentials()
        assert creds == {"api_key": "env_key", "api_secret": "env_secret"}

    def test_cds_explicit_and_aliases(self) -> None:
        creds = CredentialsManager.get_cds_credentials(key="my_cds_key")
        assert creds["key"] == "my_cds_key"
        assert creds["url"] == "https://cds.climate.copernicus.eu/api"

        creds_alias = CredentialsManager.get_cds_credentials(cds_key="alias_key", cds_url="https://custom.cds.org/api")
        assert creds_alias["key"] == "alias_key"
        assert creds_alias["url"] == "https://custom.cds.org/api"

    def test_cds_none_fallback_for_native_discovery(self, monkeypatch) -> None:
        monkeypatch.delenv("CDSAPI_KEY", raising=False)
        monkeypatch.delenv("CDSAPI_URL", raising=False)
        creds = CredentialsManager.get_cds_credentials()
        assert creds["key"] is None

    def test_earthdata_credentials(self, monkeypatch) -> None:
        monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
        monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
        monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)

        creds = CredentialsManager.get_earthdata_credentials(username="my_user", password="my_password")
        assert creds["username"] == "my_user"
        assert creds["password"] == "my_password"

        creds_tok = CredentialsManager.get_earthdata_credentials(token="bearer_123")
        assert creds_tok["token"] == "bearer_123"

    def test_unified_get_credentials_aliases(self) -> None:
        c_tahmo = get_credentials("tahmo", api_key="k1", api_secret="s1")
        assert c_tahmo == {"api_key": "k1", "api_secret": "s1"}

        c_cds = get_credentials("era5", cds_key="k2")
        assert c_cds["key"] == "k2"
        assert c_cds["url"] == "https://cds.climate.copernicus.eu/api"

        c_imerg = get_credentials("imerg", username="u3", password="p3")
        assert c_imerg["username"] == "u3"
        assert c_imerg["password"] == "p3"


# ==============================================================================
# Spatial and Temporal Boundary Tests
# ==============================================================================

class TestSpatialBoundingBox:
    """Test spatial coordinate sanity and transformations."""

    def test_valid_box(self) -> None:
        box = SpatialBoundingBox(lat_min=-4.0, lat_max=4.0, lon_min=34.0, lon_max=42.0)
        assert box.to_area_list() == [4.0, 34.0, -4.0, 42.0]  # North, West, South, East
        assert box.to_bbox_tuple() == (34.0, -4.0, 42.0, 4.0)  # West, South, East, North

    def test_invalid_lat_bounds(self) -> None:
        with pytest.raises(ValueError, match="lat_min"):
            SpatialBoundingBox(lat_min=-95.0, lat_max=10.0, lon_min=30.0, lon_max=40.0)

        with pytest.raises(ValueError, match="cannot exceed"):
            SpatialBoundingBox(lat_min=10.0, lat_max=5.0, lon_min=30.0, lon_max=40.0)

    def test_contains(self) -> None:
        box = SpatialBoundingBox(lat_min=-1.0, lat_max=1.0, lon_min=35.0, lon_max=37.0)
        assert box.contains(0.0, 36.0) is True
        assert box.contains(2.0, 36.0) is False


class TestTemporalRange:
    """Test date interval validation and splitting."""

    def test_valid_range(self) -> None:
        t = TemporalRange.from_dates("2024-01-01", "2024-03-31")
        assert len(t.years()) == 1
        windows = t.split_by_month()
        assert len(windows) == 3

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            TemporalRange.from_dates("2024-05-01", "2024-01-01")

    def test_split_by_days(self) -> None:
        t = TemporalRange.from_dates("2024-01-01", "2024-01-15")
        chunks = t.split_by_days(5)
        assert len(chunks) == 3


# ==============================================================================
# ERA5 Extraction Tests
# ==============================================================================

class TestERA5Extractor:
    """Test ERA5 chunking, ARCO query building, and CDS error interception."""

    def test_generate_n_days_dict(self) -> None:
        d = generate_n_days_dict("2023-01-01", "2023-01-20", n_days=5)
        assert 2023 in d
        assert len(d[2023]) == 4  # 4 x 5-day chunks

    def test_cds_limit_exceeded_error_interception(self, tmp_path: Path):
        extractor = ERA5Extractor(output_dir=tmp_path, prefer_arco=False)
        mock_client = MagicMock()
        mock_client.retrieve.side_effect = Exception("Request is too large. Limit exceeded.")

        with pytest.raises(CopernicusCDSError, match="Copernicus CDS limit exceeded"):
            extractor._execute_cds_request(
                client=mock_client,
                dataset="reanalysis-era5-single-levels",
                request={},
                target_path=tmp_path / "test.grib",
            )

    def test_flatten_accum_ds_monotonicity_and_native_cadence(self) -> None:
        """Verify that stacking time and step coordinates preserves monotonic 1h cadence."""
        t_base = pd.to_datetime(["2024-01-01T06:00:00", "2024-01-01T18:00:00"])
        steps = [pd.Timedelta(hours=i) for i in range(1, 13)]
        lat = np.array([0.0, 1.0])
        lon = np.array([36.0, 37.0])

        tp_data = np.ones((2, 12, 2, 2), dtype=np.float32) * 1.5
        ds = xr.Dataset(
            data_vars={"tp": (("time", "step", "lat", "lon"), tp_data)},
            coords={"time": t_base, "step": steps, "lat": lat, "lon": lon},
        )

        flat = flatten_accum_ds(ds)
        assert "valid_time" in flat.dims
        assert len(flat["valid_time"]) == 24
        # Check strictly monotonic chronological order
        v_times = pd.to_datetime(flat["valid_time"].values)
        assert (v_times[1:] > v_times[:-1]).all()
        # Check exact 1-hour delta between consecutive timestamps
        deltas = v_times[1:] - v_times[:-1]
        assert all(d == pd.Timedelta(hours=1) for d in deltas)

    def test_resample_era5_flux_sum_and_state_mean(self) -> None:
        """Verify that fluxes are aggregated by SUM and intensive states by MEAN."""
        times = pd.date_range("2024-01-01T00:00:00", periods=6, freq="1h")
        lat = np.array([0.0, 1.0])
        lon = np.array([36.0, 37.0])

        ds = xr.Dataset(
            data_vars={
                "total_precipitation": (("time", "lat", "lon"), np.ones((6, 2, 2), dtype=np.float32) * 2.0),
                "surface_solar_radiation_downwards": (("time", "lat", "lon"), np.ones((6, 2, 2), dtype=np.float32) * 100.0),
                "2m_temperature": (("time", "lat", "lon"), np.ones((6, 2, 2), dtype=np.float32) * 300.0),
                "10m_u_component_of_wind": (("time", "lat", "lon"), np.ones((6, 2, 2), dtype=np.float32) * 5.0),
                "convective_available_potential_energy": (("time", "lat", "lon"), np.ones((6, 2, 2), dtype=np.float32) * 1500.0),
                "land_sea_mask": (("lat", "lon"), np.ones((2, 2), dtype=np.float32)),
            },
            coords={"time": times, "lat": lat, "lon": lon},
        )

        # 1. 1h native cadence returns unchanged
        native = resample_era5(ds, freq="1h")
        assert native.sizes["time"] == 6
        assert np.isclose(native["total_precipitation"].values[0, 0, 0], 2.0)

        # 2. 6h coarsening: flux by SUM, state by MEAN
        resampled = resample_era5(ds, freq="6h")
        assert resampled.sizes["time"] == 1
        # Fluxes: sum of 6 hours * 2.0 = 12.0
        assert np.isclose(resampled["total_precipitation"].values[0, 0, 0], 12.0)
        # Solar radiation: sum of 6 hours * 100.0 = 600.0
        assert np.isclose(resampled["surface_solar_radiation_downwards"].values[0, 0, 0], 600.0)
        # Temperature: mean = 300.0 (NOT 1800.0)
        assert np.isclose(resampled["2m_temperature"].values[0, 0, 0], 300.0)
        # Wind: mean = 5.0 (NOT 30.0)
        assert np.isclose(resampled["10m_u_component_of_wind"].values[0, 0, 0], 5.0)
        # CAPE: mean = 1500.0 (intensive stability metric, NOT 9000.0)
        assert np.isclose(resampled["convective_available_potential_energy"].values[0, 0, 0], 1500.0)
        # Static invariant preserved
        assert "land_sea_mask" in resampled.data_vars

    def test_infer_variable_types_cape_cin_physical_sanity(self) -> None:
        """Verify MasharikiWeatherDataset infer_variable_types classifies CAPE/CIN as PHYSICAL."""
        from masharikiweather.dataset import MasharikiWeatherDataset
        types = MasharikiWeatherDataset.infer_variable_types([
            "convective_available_potential_energy",
            "convective_inhibition",
            "cape",
            "cin",
            "convective_precipitation",
            "total_precipitation",
            "surface_solar_radiation_downwards",
            "2m_temperature",
        ])
        assert types["convective_available_potential_energy"] == "PHYSICAL"
        assert types["convective_inhibition"] == "PHYSICAL"
        assert types["cape"] == "PHYSICAL"
        assert types["cin"] == "PHYSICAL"
        assert types["convective_precipitation"] == "PRECIPITATION"
        assert types["total_precipitation"] == "PRECIPITATION"
        assert types["surface_solar_radiation_downwards"] == "FLUX"
        assert types["2m_temperature"] == "PHYSICAL"

    def test_check_grib_dependencies_docker_guidance_and_no_conda(self, caplog):
        with patch.dict("sys.modules", {"cfgrib": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'cfgrib'")):
                with caplog.at_level("WARNING"):
                    available = check_grib_dependencies()
                    assert available is False
                    assert "docker build -t masharikiweather ." in caplog.text
                    assert "conda" not in caplog.text.lower()

    def test_era5_variable_harmonization(self):
        assert canonicalize_era5_variable("2t") == "2m_temperature"
        assert canonicalize_era5_variable("tp") == "total_precipitation"
        assert canonicalize_era5_variable("sp") == "surface_pressure"
        assert era5_canonical_to_short("2m_temperature") == "2t"
        assert era5_canonical_to_short("total_precipitation") == "tp"

        ds = xr.Dataset({
            "2t": (("time",), [295.15]),
            "tp": (("time",), [0.005]),
        })
        canonical_ds = harmonize_era5_dataset(ds, to_canonical=True)
        assert "2m_temperature" in canonical_ds.data_vars
        assert "total_precipitation" in canonical_ds.data_vars

        short_ds = harmonize_era5_dataset(canonical_ds, to_canonical=False)
        assert "2t" in short_ds.data_vars
        assert "tp" in short_ds.data_vars

    def test_open_era5_netcdf(self, tmp_path: Path) -> None:
        """Verify opening NetCDF files directly bypasses cfgrib and eccodes dependencies."""
        times = pd.date_range("2024-01-01 00:00:00", periods=4, freq="1h")
        ds1 = xr.Dataset(
            data_vars={
                "2t": (("time", "latitude", "longitude"), np.ones((4, 2, 2), dtype=np.float32) * 295.0),
                "tp": (("time", "latitude", "longitude"), np.zeros((4, 2, 2), dtype=np.float32)),
            },
            coords={"time": times, "latitude": [10.0, 9.75], "longitude": [35.0, 35.25]},
        )
        nc_file1 = tmp_path / "era5_chunk1.nc"
        ds1.to_netcdf(nc_file1)

        combined = open_era5_netcdf(nc_file1, harmonize=True)
        assert "lat" in combined.coords
        assert "lon" in combined.coords
        assert "2m_temperature" in combined.data_vars
        assert "total_precipitation" in combined.data_vars
        assert combined.sizes["time"] == 4

    def test_era5_adaptive_chunking_on_cds_error(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """
        Verify the CRITICAL ADAPTIVE CHUNKING RULE:
        Starts with 5-day interval. When CDS returns request-too-large, logs
        'Request size too large, reducing chunk size to 4 days...' and retries
        with smaller chunk sizes (4 days, 3 days, etc.).
        """
        from masharikiweather.extract.apis.base import SpatialBoundingBox
        extractor = ERA5Extractor(output_dir=tmp_path, use_legacy=True, data_format="netcdf4")

        mock_client = MagicMock()
        attempts = []

        def side_effect_retrieve(dataset, request, target_path):
            num_days = len(request.get("day", []))
            attempts.append(num_days)
            if num_days == 5:
                raise Exception("Request is too large. Cost limit exceeded.")
            elif num_days == 4:
                raise Exception("Request size too large. Time out.")
            # At 3 days, succeed and create dummy file
            Path(target_path).touch()

        mock_client.retrieve.side_effect = side_effect_retrieve

        with patch("cdsapi.Client", return_value=mock_client):
            with caplog.at_level(logging.INFO):
                box = SpatialBoundingBox(lat_min=0.0, lat_max=1.0, lon_min=35.0, lon_max=36.0)
                downloaded = extractor.extract_cds_slice(
                    bbox=box,
                    start_date="2023-01-01",
                    end_date="2023-01-05",
                    n_days_per_chunk=5,
                    data_format="netcdf4",
                    data_levels="single",
                )

        assert 5 in attempts
        assert 4 in attempts
        assert 3 in attempts
        assert "Request size too large, reducing chunk size to 4 days..." in caplog.text
        assert "Request size too large, reducing chunk size to 3 days..." in caplog.text
        assert len(downloaded) > 0

    def test_cds_arco_loading_with_obstore(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify CDS ARCO backend uses obstore HTTPStore + ObjectStore and retrieves CDSAPI_KEY."""
        from masharikiweather.extract.apis.base import SpatialBoundingBox
        monkeypatch.setenv("CDSAPI_KEY", "test-cds-token-12345")

        mock_ds = xr.Dataset(
            data_vars={"2m_temperature": (("time", "lat", "lon"), np.ones((2, 2, 2), dtype=np.float32) * 290.0)},
            coords={
                "time": pd.date_range("2023-01-01 00:00:00", periods=2, freq="1h"),
                "lat": [1.0, 0.0],
                "lon": [36.0, 37.0],
            },
        )

        with patch("obstore.store.HTTPStore") as mock_http_store, \
             patch("zarr.storage.ObjectStore") as mock_obj_store, \
             patch("xarray.open_zarr", return_value=mock_ds) as mock_open_zarr:

            extractor = ERA5Extractor(use_cds_arco=True)
            box = SpatialBoundingBox(lat_min=0.0, lat_max=1.0, lon_min=36.0, lon_max=37.0)
            res = extractor.extract_cds_arco_slice(
                bbox=box,
                start_date="2023-01-01",
                end_date="2023-01-02",
                variables=["2m_temperature"],
            )

            mock_http_store.assert_called_once()
            call_kwargs = mock_http_store.call_args[1]
            assert "Authorization" in call_kwargs["client_options"]["default_headers"]
            assert call_kwargs["client_options"]["default_headers"]["Authorization"] == "Bearer test-cds-token-12345"
            mock_obj_store.assert_called_once()
            assert res is not None
            assert "lat" in res.coords
            assert "lon" in res.coords

    def test_get_cds_api_key_discovery(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Verify get_cds_api_key discovers key from args, env, or ~/.cdsapirc."""
        # 1. Explicit argument
        assert get_cds_api_key(key="explicit_key_abc") == "explicit_key_abc"

        # 2. Environment variable
        monkeypatch.setenv("CDSAPI_KEY", "env_key_xyz")
        assert get_cds_api_key() == "env_key_xyz"

        # 3. File discovery
        monkeypatch.delenv("CDSAPI_KEY", raising=False)
        rc_file = tmp_path / ".cdsapirc"
        rc_file.write_text("url: https://cds.climate.copernicus.eu/api\nkey: file_key_789\n")
        assert get_cds_api_key(rc_file=rc_file) == "file_key_789"

        # 4. Error when missing
        with pytest.raises(CredentialsError):
            get_cds_api_key(rc_file=tmp_path / "nonexistent")

    def test_era5_backend_selection(self, tmp_path: Path) -> None:
        """Verify that ERA5Extractor respects use_legacy, use_cds_arco, and use_google_arco."""
        ext_default = ERA5Extractor(output_dir=tmp_path)
        assert ext_default.use_legacy is True
        assert ext_default.use_cds_arco is False
        assert ext_default.use_google_arco is False

        with patch.object(ext_default, "extract_cds_arco_slice") as mock_cds_arco, \
             patch.object(ext_default, "extract_google_arco_slice") as mock_google_arco, \
             patch.object(ext_default, "extract_cds_slice") as mock_legacy, \
             patch("masharikiweather.extract.apis.era5.open_era5_netcdf") as mock_open_nc:

            mock_cds_arco.return_value = xr.Dataset()
            mock_google_arco.return_value = xr.Dataset()
            mock_legacy.return_value = [tmp_path / "test.nc"]
            mock_open_nc.return_value = xr.Dataset()

            # Test 1: use_cds_arco=True
            ext_default.extract("2023-01-01", "2023-01-02", save_zarr=False, use_cds_arco=True)
            mock_cds_arco.assert_called_once()

            # Test 2: use_google_arco=True
            ext_default.extract("2023-01-01", "2023-01-02", save_zarr=False, use_google_arco=True)
            mock_google_arco.assert_called_once()

            # Test 3: use_legacy=True (default)
            ext_default.extract("2023-01-01", "2023-01-02", save_zarr=False, use_legacy=True)
            mock_legacy.assert_called_once()



# ==============================================================================
# TAHMO Extraction & Non-Destructive Storage Tests
# ==============================================================================

class TestTAHMOExtractor:
    """Test non-destructive multi-sensor data preservation and quality filtering."""

    def test_non_destructive_sensor_preservation(self, tmp_path: Path):
        extractor = TAHMOExtractor(output_dir=tmp_path, api_key="dummy", api_secret="dummy")

        # Simulate two collocated relative humidity sensors on same station
        holder = {
            "rh": [
                ("2024-01-01T00:00:00Z", 65.0, "S001402", 1),
                ("2024-01-01T01:00:00Z", 68.0, "S001402", 1),
                ("2024-01-01T00:00:00Z", 64.5, "S001403", 1),
                ("2024-01-01T01:00:00Z", 67.5, "S001403", 1),
            ],
            "te": [
                ("2024-01-01T00:00:00Z", 22.0, "S000100", 1),
                ("2024-01-01T01:00:00Z", 23.0, "S000100", 1),
            ],
        }

        df = extractor._build_non_destructive_dataframe(holder)
        # Verify that both sensors exist as distinct columns without averaging
        assert "rh_S001402" in df.columns
        assert "rh_S001403" in df.columns
        assert df["rh_S001402"].iloc[0] == pytest.approx(65.0)
        assert df["rh_S001403"].iloc[0] == pytest.approx(64.5)

    def test_quality_flag_filtering(self, tmp_path: Path):
        extractor = TAHMOExtractor(output_dir=tmp_path, api_key="dummy", api_secret="dummy", dataset="controlled")
        payload = {
            "results": [
                {
                    "series": [
                        {
                            "columns": ["time", "quality", "variable", "sensor", "value"],
                            "values": [
                                ["2024-01-01T00:00:00Z", 1, "te", "S1", 20.0],
                                ["2024-01-01T01:00:00Z", 0, "te", "S1", 99.0],  # Flagged bad
                            ],
                        }
                    ]
                }
            ]
        }
        holder = {}
        extractor._parse_measurements_payload(payload, variables=["te"], series_holder=holder)
        assert holder["te"][0][1] == 20.0
        assert np.isnan(holder["te"][1][1])  # Converted to NaN

    def test_generate_metadata_catalogs(self, tmp_path: Path):
        extractor = TAHMOExtractor(output_dir=tmp_path, api_key="dummy", api_secret="dummy")
        mock_stations = pd.DataFrame([
            {
                "station_code": "TA00001",
                "name": "Nairobi Test",
                "lat": -1.28,
                "lon": 36.82,
                "elevation": 1680.0,
                "country": "KE",
            }
        ])
        inst_p, param_p = extractor.generate_metadata_catalogs(stations_df=mock_stations, output_dir=tmp_path / "meta")
        assert inst_p.exists()
        assert param_p.exists()
        df_inst = pd.read_parquet(inst_p)
        assert "station_id" in df_inst.columns
        assert "latitude" in df_inst.columns
        assert "elevation" in df_inst.columns
        assert df_inst["station_id"].iloc[0] == "TA00001"

        df_param = pd.read_parquet(param_p)
        assert "parameter" in df_param.columns
        assert "canonical_name" in df_param.columns
        assert "physical_type" in df_param.columns

    def test_tahmo_open_zarr_fallback(self, tmp_path: Path):
        extractor = TAHMOExtractor(output_dir=tmp_path, api_key="dummy", api_secret="dummy")
        zarr_path = tmp_path / "mock.zarr"
        ds = xr.Dataset({"te": (("time", "station"), np.ones((2, 2)))})
        ds.to_zarr(str(zarr_path))
        opened = extractor.open_zarr(zarr_path)
        assert "te" in opened.data_vars

    def test_tahmo_variable_harmonization(self):
        assert canonicalize_tahmo_variable("te") == "2m_temperature"
        assert canonicalize_tahmo_variable("pr") == "total_precipitation"
        assert tahmo_canonical_to_short("2m_temperature") == "te"


# ==============================================================================
# IMERG Tests
# ==============================================================================

class TestIMERGExtractor:
    """Test IMERG result container, transition routing, and harmonization."""

    def test_imerg_result_properties(self, tmp_path: Path):
        res = IMERGResult(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            bbox=KENYA_BBOX,
            dataset_name="GPM_3IMERGHH",
            raw_dir=tmp_path / "raw",
            zarr_dir=tmp_path / "zarr",
            files=["file1.HDF5", "file2.HDF5"],
            zarr_stores=[str(tmp_path / "zarr" / "2024" / "IMERG_2024_01.zarr")],
        )
        assert res.ok is True
        assert len(res.files) == 2

    def test_imerg_v07_to_v08_transition_routing(self, tmp_path: Path, caplog):
        extractor = IMERGExtractor(output_dir=tmp_path, username="u", password="p")
        executed_searches = []

        with patch("earthaccess.login") as mock_login, \
             patch("earthaccess.search_data") as mock_search, \
             patch("earthaccess.download", return_value=[]):
            mock_login.return_value.authenticated = True
            mock_search.side_effect = lambda **kwargs: executed_searches.append(kwargs) or []

            with caplog.at_level("INFO"):
                extractor.extract(
                    start_date="2025-10-15",
                    end_date="2025-11-15",
                    bbox=KENYA_BBOX,
                    convert_to_zarr=False,
                )

            short_names = [s["short_name"] for s in executed_searches]
            assert "GPM_3IMERGHH" in short_names
            assert IMERG_LATE_RUN_NAME in short_names
            assert any("exceeds IMERG Final Run cutoff" in r.message for r in caplog.records)

    def test_imerg_variable_harmonization(self):
        assert canonicalize_imerg_variable("precipitationCal") == "total_precipitation"
        assert imerg_canonical_to_raw("total_precipitation") == "precipitationCal"
        ds = xr.Dataset({"precipitationCal": (("time", "lat", "lon"), np.ones((1, 2, 2)))})
        harmonized = harmonize_imerg_dataset(ds, to_canonical=True)
        assert "total_precipitation" in harmonized.data_vars


# ==============================================================================
# CDS In-Situ Tests
# ==============================================================================

class TestCDSInSituExtractor:
    """Test CDS In-Situ byte-string decoding, flat-index un-nesting, and metadata parsing."""

    def test_parse_cds_insitu_dataframe_complexities(self):
        raw_data = {
            "primary_station_id": [b"TA0001", b"TA0001", b"TA0002", b"TA0002"],
            "station_name": [b"Station Alpha", b"Station Alpha", b"Station Beta", b"Station Beta"],
            "latitude": [-1.28, -1.28, 0.51, 0.51],
            "longitude": [36.82, 36.82, 35.28, 35.28],
            "height_of_station_above_sea_level": [1680.0, 1680.0, 2100.0, 2100.0],
            "date_time": [
                "2023-01-01 00:00:00",
                "2023-01-01 00:00:00",
                "2023-01-01 00:00:00",
                "2023-01-01 00:00:00",
            ],
            "observed_variable": [
                b"air_temperature",
                b"accumulated_precipitation",
                b"air_temperature",
                b"accumulated_precipitation",
            ],
            "observation_value": [21.5, 0.2, 18.0, 1.4],
            "quality_flag": [0, 0, 0, 0],
        }
        raw_df = pd.DataFrame(raw_data)
        parsed = parse_cds_insitu_dataframe(raw_df, harmonize=True)

        # Verify un-nested columns
        assert "station_id" in parsed.columns
        assert "station_name" in parsed.columns
        assert "elevation" in parsed.columns
        assert "2m_temperature" in parsed.columns
        assert "total_precipitation" in parsed.columns

        # Verify byte-strings are decoded
        assert parsed["station_id"].iloc[0] == "TA0001"
        assert parsed["station_name"].iloc[0] == "Station Alpha"

        # Verify pivoted row count: 2 stations at 1 timestamp = 2 rows
        assert len(parsed) == 2
        row_alpha = parsed[parsed["station_id"] == "TA0001"].iloc[0]
        assert row_alpha["2m_temperature"] == pytest.approx(21.5)
        assert row_alpha["total_precipitation"] == pytest.approx(0.2)
        assert row_alpha["elevation"] == pytest.approx(1680.0)

    def test_cds_insitu_canonical_mapping(self):
        assert canonicalize_cds_insitu_variable("air_temperature") == "2m_temperature"
        assert canonicalize_cds_insitu_variable("accumulated_precipitation") == "total_precipitation"
        assert cds_insitu_canonical_to_raw("2m_temperature") == "air_temperature"


# ==============================================================================
# Ancillary APIs Tests
# ==============================================================================

class TestAncillaryExtractors:
    """Test GHCN, Reanalysis, and Satellite extraction contracts."""

    def test_ghcn_extractor_initialization(self, tmp_path: Path):
        extractor = GHCNExtractor(output_dir=tmp_path, resolution="daily", region="kenya")
        assert extractor.region == "kenya"

    def test_reanalysis_extractor_initialization(self, tmp_path: Path):
        extractor = ReanalysisExtractor(output_dir=tmp_path, source="open_meteo")
        assert extractor.source == "open_meteo"
        assert len(WMO_WEATHER_CODES) > 0

    def test_satellite_extractor_initialization(self, tmp_path: Path):
        extractor = SatelliteExtractor(output_dir=tmp_path, source="nasa_power")
        assert extractor.source == "nasa_power"

    def test_decode_weather_code_non_emoji(self) -> None:
        """Verify WMO codes decode to text descriptors without unicode emojis."""
        from masharikiweather.extract.apis.reanalysis import decode_weather_code
        res_0 = decode_weather_code(0)
        assert res_0["description"] == "Clear sky"
        assert res_0["icon"] == "clear_sky"

        res_95 = decode_weather_code(95)
        assert res_95["description"] == "Thunderstorm"
        assert res_95["icon"] == "thunderstorm"

        res_nan = decode_weather_code(float("nan"))
        assert res_nan["description"] == "Unknown"
        assert res_nan["icon"] == "unknown"

    def test_imerg_result_open_zarr_alias(self, tmp_path: Path):
        """Verify open_zarr is a resilient alias for open on IMERGResult."""
        res = IMERGResult(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            bbox=KENYA_BBOX,
            dataset_name="GPM_3IMERGHH",
            raw_dir=tmp_path / "raw",
            zarr_dir=tmp_path / "zarr",
        )
        assert hasattr(res, "open_zarr")
        assert res.open_zarr == res.open

    def test_reanalysis_extractor_extract_mock(self, tmp_path: Path):
        """Test ReanalysisExtractor.extract with mock OpenMeteo returning DataFrame."""
        extractor = ReanalysisExtractor(output_dir=tmp_path, source="open_meteo")
        mock_df = pd.DataFrame(
            {"temperature_2m": [24.0, 25.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="h"),
        )
        with patch("masharikiweather.extract.apis.reanalysis.OpenMeteo") as mock_om_cls:
            mock_inst = MagicMock()
            mock_om_cls.return_value = mock_inst
            mock_inst.return_value = mock_df

            res = extractor.extract(
                start_date="2023-01-01",
                end_date="2023-01-02",
                locations=[{"name": "nairobi", "lat": -1.28, "lon": 36.82}],
                save_parquet=True,
            )
            assert isinstance(res, Path)
            assert res.exists()
            loaded = pd.read_parquet(res)
            assert "temperature_2m" in loaded.columns

    def test_satellite_extractor_extract_mock(self, tmp_path: Path):
        """Test SatelliteExtractor.extract with mock NASAPower returning DataFrame."""
        extractor = SatelliteExtractor(output_dir=tmp_path, source="nasa_power")
        mock_df = pd.DataFrame(
            {"ALLSKY_SFC_SW_DWN": [200.0, 350.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="h"),
        )
        with patch("masharikiweather.extract.apis.satellite.NASAPower") as mock_np_cls:
            mock_inst = MagicMock()
            mock_np_cls.return_value = mock_inst
            mock_inst.return_value = mock_df

            res = extractor.extract(
                start_date="2023-01-01",
                end_date="2023-01-02",
                locations=[{"name": "nairobi", "lat": -1.28, "lon": 36.82}],
                save_parquet=True,
            )
            assert isinstance(res, Path)
            assert res.exists()
            loaded = pd.read_parquet(res)
            assert "ALLSKY_SFC_SW_DWN" in loaded.columns


# ==============================================================================
# Date Handling and Leap Year Tests
# ==============================================================================

class TestDateHandlingAndLeapYears:
    """Test date parsing, date ranges, and leap year management."""

    def test_parse_date_and_range(self) -> None:
        ts = parse_date("2024-02-29")
        assert ts.year == 2024 and ts.month == 2 and ts.day == 29

        s, e = parse_date_range("2024-01-01", "2024-02-29")
        assert s <= e

        with pytest.raises(ValueError, match="cannot exceed"):
            parse_date_range("2024-03-01", "2024-02-29")

    def test_leap_year_february_split_by_month(self) -> None:
        # 2024 is leap year: Feb has 29 days
        t_leap = TemporalRange.from_dates("2024-02-01", "2024-03-10")
        windows_leap = t_leap.split_by_month()
        assert len(windows_leap) == 2
        assert windows_leap[0][1].day == 29

        # 2023 is non-leap year: Feb has 28 days
        t_nonleap = TemporalRange.from_dates("2023-02-01", "2023-03-10")
        windows_nonleap = t_nonleap.split_by_month()
        assert len(windows_nonleap) == 2
        assert windows_nonleap[0][1].day == 28

    def test_generate_n_days_dict_leap_year(self) -> None:
        chunks = generate_n_days_dict("2024-02-25", "2024-03-05", n_days=5)
        assert 2024 in chunks
        feb_chunk = [days for (start, end), days in chunks[2024].items() if "02" in start]
        assert any("29" in d for d in feb_chunk)


# ==============================================================================
# Dataset Module Extractor Delegation Tests
# ==============================================================================

class TestDatasetExtractors:
    """Test extract_stations and extract_gridded signatures and argument propagation."""

    def test_extract_stations_propagation(self, tmp_path: Path):
        from masharikiweather.dataset import extract_stations
        with patch("masharikiweather.extract.apis.tahmo.TAHMOExtractor") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.extract.return_value = {"station_1": tmp_path / "st1.parquet"}

            res = extract_stations(
                source="tahmo",
                start_date="2023-01-01",
                end_date="2023-01-05",
                api_key="my_key",
                api_secret="my_secret",
                output_dir=tmp_path,
                save_parquet=True,
            )
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["api_key"] == "my_key"
            assert call_kwargs["api_secret"] == "my_secret"
            assert isinstance(res, dict)

    def test_extract_gridded_propagation(self, tmp_path: Path):
        from masharikiweather.dataset import extract_gridded
        with patch("masharikiweather.extract.apis.era5.ERA5Extractor") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.extract.return_value = tmp_path / "era5.zarr"

            res = extract_gridded(
                source="era5",
                start_date="2023-01-01",
                end_date="2023-01-05",
                cds_key="cds_123",
                cds_url="https://cds.copernicus.eu/api",
                output_dir=tmp_path,
                save_zarr=True,
            )
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["cds_key"] == "cds_123"
            assert call_kwargs["cds_url"] == "https://cds.copernicus.eu/api"
            assert isinstance(res, Path)


# ==============================================================================
# Decomposed Reanalysis Modules Tests
# ==============================================================================

class TestDecomposedReanalysisModules:
    """Test suite for decomposed reanalysis modules: cams, merra2, open_meteo, and extractor."""

    def test_cams_metadata_and_search(self) -> None:
        """Verify CAMS parameter searching, listing, and AQI classification."""
        from masharikiweather.extract.apis.reanalysis.cams import (
            classify_aqi,
            list_cams_parameters,
            search_cams_parameters,
        )

        df_search = search_cams_parameters("pm2_5")
        assert not df_search.empty
        assert "pm2_5" in df_search["Variable"].values

        df_cat = list_cams_parameters(category="Particulate Matter")
        assert not df_cat.empty
        assert all("Particulate Matter" in c for c in df_cat["Category"])

        # Validate AQI classification
        aqi_good = classify_aqi(8.0)
        assert aqi_good["label"] == "Good"
        assert aqi_good["color"] == "#00e400"

        aqi_mod = classify_aqi(65.0)
        assert aqi_mod["label"] == "Moderate"

    def test_cams_mock_fetch_point(self, tmp_path: Path):
        """Verify CAMS fetch_point parses Open-Meteo Air Quality payload offline."""
        import json
        from masharikiweather.extract.apis.reanalysis.cams import CAMS

        mock_payload = {
            "hourly": {
                "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
                "pm2_5": [15.2, 18.4],
                "pm10": [25.0, 30.0],
            }
        }
        with patch("masharikiweather.extract.apis.reanalysis.cams._safe_request") as mock_req:
            mock_req.return_value = json.dumps(mock_payload).encode("utf-8")
            client = CAMS(cache_dir=tmp_path)
            records = client.fetch_point(
                lat=-1.28,
                lon=36.82,
                start_date="2023-01-01",
                end_date="2023-01-02",
                variables=["pm2_5", "pm10"],
            )
            assert len(records) == 4
            assert all(r["lat"] == -1.28 for r in records)
            assert all(r["lon"] == 36.82 for r in records)
            assert {r["variable"] for r in records} == {"pm2_5", "pm10"}

    def test_merra2_metadata_and_search(self) -> None:
        """Verify MERRA-2 parameter discovery and collection catalog."""
        from masharikiweather.extract.apis.reanalysis.merra2 import (
            list_merra2_collections,
            list_merra2_parameters,
            search_merra2_parameters,
        )

        res_search = search_merra2_parameters("aod")
        assert len(res_search) > 0
        assert any("aod" in v["name"].lower() for v in res_search)

        df_params = list_merra2_parameters()
        assert not df_params.empty

        df_colls = list_merra2_collections()
        assert not df_colls.empty
        assert "Collection ID" in df_colls.columns

    def test_merra2_mock_fetch_point(self, tmp_path: Path):
        """Verify MERRA-2 fetch_point queries via NASA POWER / fallback offline."""
        import json
        from masharikiweather.extract.apis.reanalysis.merra2 import MERRA2

        mock_power = {
            "properties": {
                "parameter": {
                    "AOD_550": {"2023010100": 0.15, "2023010101": 0.18},
                    "ALLSKY_SFC_SW_DWN": {"2023010100": 100.0, "2023010101": 250.0},
                }
            }
        }
        with patch("masharikiweather.extract.apis.reanalysis.merra2._safe_request") as mock_req:
            mock_req.return_value = json.dumps(mock_power).encode("utf-8")
            client = MERRA2(cache_dir=tmp_path)
            records = client._fetch_point(
                lat=-1.28,
                lon=36.82,
                start_date="2023-01-01",
                end_date="2023-01-02",
                requested_vars=["aod_total", "swgdn"],
            )
            assert len(records) > 0
            assert all("time" in r for r in records)

    def test_open_meteo_metadata_and_search(self) -> None:
        """Verify Open-Meteo parameter searching and WMO weather code decoding."""
        from masharikiweather.extract.apis.reanalysis.open_meteo import (
            decode_weather_code,
            list_open_meteo_parameters,
            search_open_meteo_parameters,
        )

        df_search = search_open_meteo_parameters("temperature")
        assert not df_search.empty
        assert any("temperature" in v.lower() for v in df_search["Variable"])

        df_all = list_open_meteo_parameters()
        assert len(df_all) >= 20

        # WMO decoding
        dec_0 = decode_weather_code(0)
        assert dec_0["description"] == "Clear sky"

    def test_open_meteo_mock_fetch_point(self, tmp_path: Path):
        """Verify Open-Meteo fetch_point parses standard weather payload offline."""
        import json
        from masharikiweather.extract.apis.reanalysis.open_meteo import OpenMeteo

        mock_payload = {
            "hourly": {
                "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
                "temperature_2m": [22.4, 23.1],
                "precipitation": [0.0, 1.2],
            }
        }
        with patch("masharikiweather.extract.apis.reanalysis.open_meteo._safe_request") as mock_req:
            mock_req.return_value = json.dumps(mock_payload).encode("utf-8")
            client = OpenMeteo(cache_dir=tmp_path)
            records = client.fetch_point(
                lat=-1.28,
                lon=36.82,
                start_date="2023-01-01",
                end_date="2023-01-02",
                variables=["temperature_2m", "precipitation"],
            )
            assert len(records) == 4
            assert {r["variable"] for r in records} == {"temperature_2m", "precipitation"}

    def test_reanalysis_extractor_variables_all_cams(self, tmp_path: Path):
        """Verify ReanalysisExtractor passes variables='all' cleanly to CAMS."""
        extractor = ReanalysisExtractor(output_dir=tmp_path, source="cams")
        mock_df = pd.DataFrame({"pm2_5": [12.0]}, index=pd.date_range("2023-01-01", periods=1, freq="h"))

        with patch("masharikiweather.extract.apis.reanalysis.CAMS") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.return_value = mock_df

            res = extractor.extract(
                start_date="2023-01-01",
                end_date="2023-01-02",
                locations=[{"name": "nairobi", "lat": -1.28, "lon": 36.82}],
                variables="all",
                save_parquet=False,
            )
            assert isinstance(res, pd.DataFrame)
            mock_inst.assert_called_once()
            assert mock_inst.call_args[1]["variables"] == "all"

    def test_reanalysis_extractor_custom_variables_merra2(self, tmp_path: Path):
        """Verify ReanalysisExtractor passes arbitrary user-defined variables to MERRA-2."""
        extractor = ReanalysisExtractor(output_dir=tmp_path, source="merra2")
        mock_df = pd.DataFrame({"aod_total": [0.1]}, index=pd.date_range("2023-01-01", periods=1, freq="h"))

        with patch("masharikiweather.extract.apis.reanalysis.MERRA2") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.return_value = mock_df

            res = extractor.extract(
                start_date="2023-01-01",
                end_date="2023-01-02",
                locations=[{"name": "nairobi", "lat": -1.28, "lon": 36.82}],
                variables=["aod_total", "dust_surface"],
                save_parquet=False,
            )
            assert isinstance(res, pd.DataFrame)
            mock_inst.assert_called_once()
            assert mock_inst.call_args[1]["variables"] == ["aod_total", "dust_surface"]

    def test_reanalysis_extractor_variables_all_open_meteo(self, tmp_path: Path):
        """Verify ReanalysisExtractor passes variables='all' to OpenMeteo."""
        extractor = ReanalysisExtractor(output_dir=tmp_path, source="open_meteo")
        mock_df = pd.DataFrame({"temperature_2m": [22.0]}, index=pd.date_range("2023-01-01", periods=1, freq="h"))

        with patch("masharikiweather.extract.apis.reanalysis.OpenMeteo") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.return_value = mock_df

            res = extractor.extract(
                start_date="2023-01-01",
                end_date="2023-01-02",
                locations=[{"name": "nairobi", "lat": -1.28, "lon": 36.82}],
                variables="all",
                save_parquet=False,
            )
            assert isinstance(res, pd.DataFrame)
            mock_inst.assert_called_once()
            assert mock_inst.call_args[1]["variables"] == "all"

    def test_reanalysis_extractor_bbox_grid_sampling(self, tmp_path: Path):
        """Verify ReanalysisExtractor automatically samples a grid when bbox is specified."""
        extractor = ReanalysisExtractor(output_dir=tmp_path, source="open_meteo")
        mock_df = pd.DataFrame({"temperature_2m": [20.0]}, index=pd.date_range("2023-01-01", periods=1, freq="h"))

        with patch("masharikiweather.extract.apis.reanalysis.OpenMeteo") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.return_value = mock_df

            box = SpatialBoundingBox(lat_min=-1.0, lat_max=1.0, lon_min=36.0, lon_max=38.0)
            res = extractor.extract(
                start_date="2023-01-01",
                end_date="2023-01-02",
                bbox=box,
                save_parquet=False,
            )
            assert isinstance(res, pd.DataFrame)
            mock_inst.assert_called_once()
            locs = mock_inst.call_args[1]["locations"]
            # 3 x 3 grid sampling produces exactly 9 location points
            assert len(locs) == 9


# ==============================================================================
# Decomposed Satellite Modules Tests
# ==============================================================================

class TestDecomposedSatelliteModules:
    """Test suite for decomposed satellite modules: nasa_power, pvgis, sarah3, landsaf, tropomi, and extractor."""

    def test_nasa_power_metadata_and_search(self) -> None:
        """Verify NASA POWER parameter searching and catalog listing."""
        from masharikiweather.extract.apis.satellite.nasa_power import (
            list_parameters,
            search_parameters,
        )

        df_search = search_parameters("solar")
        assert not df_search.empty
        assert any("irradiance" in d.lower() for d in df_search["description"])

        params = list_parameters()
        assert len(params) >= 10
        assert "allsky_sfc_sw_dwn" in params

    def test_nasa_power_mock_fetch_point(self, tmp_path: Path):
        """Verify NASAPower fetch_point parses hourly NASA POWER payload offline."""
        import json
        from masharikiweather.extract.apis.satellite.nasa_power import NASAPower

        mock_payload = {
            "properties": {
                "parameter": {
                    "ALLSKY_SFC_SW_DWN": {"2023010100": 0.0, "2023010112": 850.5},
                    "T2M": {"2023010100": 18.2, "2023010112": 26.5},
                }
            }
        }
        with patch("masharikiweather.extract.apis.satellite.nasa_power._safe_request") as mock_req:
            mock_req.return_value = json.dumps(mock_payload).encode("utf-8")
            client = NASAPower(cache_dir=tmp_path)
            records = client.fetch_point(
                lat=-1.28,
                lon=36.82,
                start_date="2023-01-01",
                end_date="2023-01-02",
                variables=["ALLSKY_SFC_SW_DWN", "T2M"],
            )
            assert len(records) == 4
            assert all(r["lat"] == -1.28 for r in records)

    def test_pvgis_metadata_and_search(self) -> None:
        """Verify PVGIS radiation parameters and database catalog."""
        from masharikiweather.extract.apis.satellite.pvgis import (
            list_databases,
            search_parameters,
        )

        df_search = search_parameters("radiation")
        assert not df_search.empty

        dbs = list_databases()
        assert len(dbs) > 0
        assert any("PVGIS" in d for d in dbs)

    def test_pvgis_mock_fetch_point(self, tmp_path: Path):
        """Verify PVGIS fetch_point parses hourly radiation output offline."""
        import json
        from masharikiweather.extract.apis.satellite.pvgis import PVGIS

        mock_payload = {
            "outputs": {
                "hourly": [
                    {"time": "20200101:0010", "G(h)": 0.0, "T2m": 16.5},
                    {"time": "20200101:1210", "G(h)": 920.0, "T2m": 27.2},
                ]
            }
        }
        with patch("masharikiweather.extract.apis.satellite.pvgis._safe_request") as mock_req:
            mock_req.return_value = json.dumps(mock_payload).encode("utf-8")
            client = PVGIS(cache_dir=tmp_path)
            records = client.fetch_point(
                lat=-1.28,
                lon=36.82,
                start_year=2020,
                end_year=2020,
            )
            assert len(records) > 0
            assert all("time" in r for r in records)

    def test_sarah3_metadata_and_search(self) -> None:
        """Verify CM SAF SARAH-3 parameters and metadata."""
        from masharikiweather.extract.apis.satellite.sarah3 import (
            get_sarah3_metadata,
            list_sarah3_parameters,
            search_sarah3_parameters,
        )

        df_search = search_sarah3_parameters("dni")
        assert not df_search.empty
        assert "direct_normal_irradiance" in df_search["Variable"].values

        df_all = list_sarah3_parameters()
        assert len(df_all) >= 4

        meta = get_sarah3_metadata()
        assert "SARAH-3" in meta["dataset_name"]

    def test_sarah3_mock_fetch_point(self, tmp_path: Path):
        """Verify SARAH-3 fetch_point parses solar radiation payload offline."""
        import json
        from masharikiweather.extract.apis.satellite.sarah3 import SARAH3

        mock_payload = {
            "hourly": {
                "time": ["2023-01-01T00:00", "2023-01-01T12:00"],
                "shortwave_radiation": [0.0, 780.0],
                "direct_normal_irradiance": [0.0, 890.0],
            }
        }
        with patch("masharikiweather.extract.apis.satellite.sarah3._safe_request") as mock_req:
            mock_req.return_value = json.dumps(mock_payload).encode("utf-8")
            client = SARAH3(cache_dir=tmp_path)
            records = client.fetch_point(
                lat=-1.28,
                lon=36.82,
                start_date="2023-01-01",
                end_date="2023-01-02",
                variables=["sis", "dni"],
            )
            assert len(records) == 4
            assert {r["variable"] for r in records} == {"sis", "dni"}

    def test_landsaf_metadata_and_search(self) -> None:
        """Verify LSA SAF LANDSAF parameter discovery and products."""
        from masharikiweather.extract.apis.satellite.landsaf import (
            get_landsaf_metadata,
            list_landsaf_parameters,
            list_landsaf_products,
            search_landsaf_parameters,
        )

        res_search = search_landsaf_parameters("dssf")
        assert len(res_search) > 0

        df_params = list_landsaf_parameters()
        assert not df_params.empty

        df_prods = list_landsaf_products()
        assert len(df_prods) >= 3

        meta = get_landsaf_metadata("DSSF")
        assert "shortwave flux" in meta.get("description", "").lower()

    def test_landsaf_mock_fetch_point(self, tmp_path: Path):
        """Verify LANDSAF fetch_point parses surface radiation data offline."""
        import json
        from masharikiweather.extract.apis.satellite.landsaf import LANDSAF

        mock_payload = {
            "hourly": {
                "time": ["2023-01-01T00:00", "2023-01-01T12:00"],
                "shortwave_radiation": [0.0, 650.0],
                "direct_radiation": [0.0, 480.0],
            }
        }
        with patch("masharikiweather.extract.apis.satellite.landsaf._safe_request") as mock_req:
            mock_req.return_value = json.dumps(mock_payload).encode("utf-8")
            client = LANDSAF(cache_dir=tmp_path)
            records = client._fetch_point(
                lat=-1.28,
                lon=36.82,
                start_date="2023-01-01",
                end_date="2023-01-02",
                requested_vars=["dssf_tot", "dssf_dir"],
            )
            assert len(records) > 0
            assert all("variable" in r for r in records)

    def test_tropomi_metadata_and_overpass(self) -> None:
        """Verify Sentinel-5P TROPOMI parameter catalog and solar overpass hour calculation."""
        from masharikiweather.extract.apis.satellite.tropomi import (
            TROPOMI,
            list_tropomi_parameters,
            search_tropomi_parameters,
        )

        res_search = search_tropomi_parameters("no2")
        assert len(res_search) > 0
        assert any("nitrogen_dioxide" == v["name"] for v in res_search)

        df_all = list_tropomi_parameters()
        assert len(df_all) >= 4

        # Validate solar noon overpass calculation: Kenya (lon ~ 37 deg) overpass ~ 11 UTC
        tropomi_inst = TROPOMI()
        kenya_overpass = tropomi_inst._calculate_overpass_hour(36.82)
        assert kenya_overpass == 11

        # Prime meridian (lon 0 deg) overpass ~ 14 UTC (13:30 local solar time rounded)
        london_overpass = tropomi_inst._calculate_overpass_hour(0.0)
        assert london_overpass == 14

    def test_satellite_extractor_variables_all_nasa_power(self, tmp_path: Path):
        """Verify SatelliteExtractor passes variables='all' cleanly to NASAPower."""
        extractor = SatelliteExtractor(output_dir=tmp_path, source="nasa_power")
        mock_df = pd.DataFrame({"ALLSKY_SFC_SW_DWN": [500.0]}, index=pd.date_range("2023-01-01", periods=1, freq="h"))

        with patch("masharikiweather.extract.apis.satellite.NASAPower") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.return_value = mock_df

            res = extractor.extract(
                start_date="2023-01-01",
                end_date="2023-01-02",
                locations=[{"name": "nairobi", "lat": -1.28, "lon": 36.82}],
                variables="all",
                save_parquet=False,
            )
            assert isinstance(res, pd.DataFrame)
            mock_inst.assert_called_once()
            assert mock_inst.call_args[1]["variables"] == "all"

    def test_satellite_extractor_custom_variables_nasa_power(self, tmp_path: Path):
        """Verify SatelliteExtractor passes arbitrary variables to NASAPower."""
        extractor = SatelliteExtractor(output_dir=tmp_path, source="nasa_power")
        mock_df = pd.DataFrame({"ALLSKY_SFC_SW_DWN": [500.0]}, index=pd.date_range("2023-01-01", periods=1, freq="h"))

        with patch("masharikiweather.extract.apis.satellite.NASAPower") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.return_value = mock_df

            custom_vars = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN"]
            res = extractor.extract(
                start_date="2023-01-01",
                end_date="2023-01-02",
                locations=[{"name": "nairobi", "lat": -1.28, "lon": 36.82}],
                variables=custom_vars,
                save_parquet=False,
            )
            assert isinstance(res, pd.DataFrame)
            mock_inst.assert_called_once()
            assert mock_inst.call_args[1]["variables"] == custom_vars

    def test_satellite_extractor_variables_all_sarah3(self, tmp_path: Path):
        """Verify SatelliteExtractor passes variables='all' to SARAH3."""
        extractor = SatelliteExtractor(output_dir=tmp_path, source="sarah3")
        mock_df = pd.DataFrame({"sis": [600.0]}, index=pd.date_range("2023-01-01", periods=1, freq="h"))

        with patch("masharikiweather.extract.apis.satellite.SARAH3") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.return_value = mock_df

            res = extractor.extract(
                start_date="2023-01-01",
                end_date="2023-01-02",
                locations=[{"name": "nairobi", "lat": -1.28, "lon": 36.82}],
                variables="all",
                save_parquet=False,
            )
            assert isinstance(res, pd.DataFrame)
            mock_inst.assert_called_once()
            assert mock_inst.call_args[1]["variables"] == "all"

    def test_satellite_extractor_custom_variables_landsaf(self, tmp_path: Path):
        """Verify SatelliteExtractor passes arbitrary variables to LANDSAF."""
        extractor = SatelliteExtractor(output_dir=tmp_path, source="landsaf")
        mock_df = pd.DataFrame({"dssf_tot": [400.0]}, index=pd.date_range("2023-01-01", periods=1, freq="h"))

        with patch("masharikiweather.extract.apis.satellite.LANDSAF") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.return_value = mock_df

            res = extractor.extract(
                start_date="2023-01-01",
                end_date="2023-01-02",
                locations=[{"name": "nairobi", "lat": -1.28, "lon": 36.82}],
                variables=["dssf_tot", "dssf_dir"],
                save_parquet=False,
            )
            assert isinstance(res, pd.DataFrame)
            mock_inst.assert_called_once()
            assert mock_inst.call_args[1]["variables"] == ["dssf_tot", "dssf_dir"]

    def test_satellite_extractor_variables_all_tropomi(self, tmp_path: Path):
        """Verify SatelliteExtractor passes variables='all' to TROPOMI."""
        extractor = SatelliteExtractor(output_dir=tmp_path, source="tropomi")
        mock_df = pd.DataFrame({"nitrogen_dioxide": [0.0001]}, index=pd.date_range("2023-01-01", periods=1, freq="h"))

        with patch("masharikiweather.extract.apis.satellite.TROPOMI") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.return_value = mock_df

            res = extractor.extract(
                start_date="2023-01-01",
                end_date="2023-01-02",
                locations=[{"name": "nairobi", "lat": -1.28, "lon": 36.82}],
                variables="all",
                save_parquet=False,
            )
            assert isinstance(res, pd.DataFrame)
            mock_inst.assert_called_once()
            assert mock_inst.call_args[1]["variables"] == "all"

    def test_satellite_extractor_pvgis_delegation(self, tmp_path: Path):
        """Verify SatelliteExtractor extracts PVGIS simulation data."""
        extractor = SatelliteExtractor(output_dir=tmp_path, source="pvgis")
        mock_df = pd.DataFrame({"G(h)": [500.0]}, index=pd.date_range("2020-01-01", periods=1, freq="h"))

        with patch("masharikiweather.extract.apis.satellite.PVGIS") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.return_value = mock_df

            res = extractor.extract(
                start_date="2020-01-01",
                end_date="2020-12-31",
                locations=[{"name": "nairobi", "lat": -1.28, "lon": 36.82}],
                save_parquet=False,
            )
            assert isinstance(res, pd.DataFrame)
            mock_inst.assert_called_once()
            assert mock_inst.call_args[1]["start_year"] == 2020
            assert mock_inst.call_args[1]["end_year"] == 2020

    def test_satellite_extractor_bbox_sampling(self, tmp_path: Path):
        """Verify SatelliteExtractor samples bounding box points when bbox is passed."""
        extractor = SatelliteExtractor(output_dir=tmp_path, source="nasa_power")
        mock_df = pd.DataFrame({"ALLSKY_SFC_SW_DWN": [200.0]}, index=pd.date_range("2023-01-01", periods=1, freq="h"))

        with patch("masharikiweather.extract.apis.satellite.NASAPower") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.return_value = mock_df

            box = SpatialBoundingBox(lat_min=-2.0, lat_max=0.0, lon_min=35.0, lon_max=37.0)
            res = extractor.extract(
                start_date="2023-01-01",
                end_date="2023-01-02",
                bbox=box,
                save_parquet=False,
            )
            assert isinstance(res, pd.DataFrame)
            mock_inst.assert_called_once()
            locs = mock_inst.call_args[1]["locations"]
            assert len(locs) == 9

