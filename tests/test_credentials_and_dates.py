"""
Comprehensive Test Suite for Credentials Management, Temporal Boundary Edge Cases,
and ECMWF ERA5 GRIB Ingestion in MasharikiWeather.
"""

from __future__ import annotations

import calendar
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
    SpatialBoundingBox,
    TemporalRange,
)
from masharikiweather.extract.apis.credentials import (
    CredentialsError,
    CredentialsManager,
    get_credentials,
    parse_date,
    parse_date_range,
)
from masharikiweather.extract.apis.era5 import (
    ERA5_ACCUMULATED_VARIABLES,
    ERA5_INSTANTANEOUS_VARIABLES,
    generate_n_days_dict,
    open_era5_grib,
)


class TestCredentialsFlexibility:
    """Tests ensuring no hardcoded paths are required and credentials accept direct arguments."""

    def test_tahmo_direct_arguments(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TAHMO_API_KEY", raising=False)
        monkeypatch.delenv("TAHMO_API_SECRET", raising=False)
        creds = CredentialsManager.get_tahmo_credentials(
            api_key="direct_tahmo_key",
            api_secret="direct_tahmo_secret",
        )
        assert creds["api_key"] == "direct_tahmo_key"
        assert creds["api_secret"] == "direct_tahmo_secret"

    def test_tahmo_env_variables(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TAHMO_API_KEY", "env_tahmo_key")
        monkeypatch.setenv("TAHMO_API_SECRET", "env_tahmo_secret")
        creds = CredentialsManager.get_tahmo_credentials()
        assert creds["api_key"] == "env_tahmo_key"
        assert creds["api_secret"] == "env_tahmo_secret"

    def test_tahmo_missing_env_file_does_not_crash_when_args_provided(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TAHMO_API_KEY", raising=False)
        monkeypatch.delenv("TAHMO_API_SECRET", raising=False)
        # Passing an explicit key even if non-existent env_file is given
        creds = CredentialsManager.get_tahmo_credentials(
            api_key="valid_key",
            api_secret="valid_secret",
            env_file="/non_existent_path/fake.env",
        )
        assert creds["api_key"] == "valid_key"
        assert creds["api_secret"] == "valid_secret"

    def test_cds_direct_arguments(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CDSAPI_KEY", raising=False)
        monkeypatch.delenv("CDSAPI_URL", raising=False)
        creds = CredentialsManager.get_cds_credentials(
            cds_key="my_cds_key_123",
            cds_url="https://cds-test.copernicus.eu/api",
        )
        assert creds["key"] == "my_cds_key_123"
        assert creds["url"] == "https://cds-test.copernicus.eu/api"

    def test_cds_automatic_discovery_when_no_args_provided(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CDSAPI_KEY", raising=False)
        monkeypatch.delenv("CDSAPI_URL", raising=False)
        # When no explicit keys or env vars are given, returns None to let cdsapi.Client natively discover ~/.cdsapirc
        creds = CredentialsManager.get_cds_credentials()
        assert creds["key"] is None

    def test_cds_env_variables(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CDSAPI_KEY", "cds_env_key_456")
        monkeypatch.setenv("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
        creds = CredentialsManager.get_cds_credentials()
        assert creds["key"] == "cds_env_key_456"
        assert creds["url"] == "https://cds.climate.copernicus.eu/api"

    def test_earthdata_direct_arguments(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
        monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
        creds = CredentialsManager.get_earthdata_credentials(
            username="nasa_user_01",
            password="nasa_password_01",
        )
        assert creds["username"] == "nasa_user_01"
        assert creds["password"] == "nasa_password_01"

    def test_earthdata_token_argument(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
        monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
        monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
        creds = CredentialsManager.get_earthdata_credentials(token="bearer_token_xyz")
        assert creds["token"] == "bearer_token_xyz"

    def test_get_credentials_unified_dispatcher(self) -> None:
        tahmo = get_credentials("tahmo", api_key="k1", api_secret="s1")
        assert tahmo["api_key"] == "k1"

        cds = get_credentials("era5", key="k2", url="u2")
        assert cds["key"] == "k2"

        ed = get_credentials("imerg", username="u3", password="p3")
        assert ed["username"] == "u3"


class TestTemporalBoundaryManagement:
    """Tests verifying month-crossing, year-crossing, and leap year management."""

    def test_month_crossing_chunks(self) -> None:
        # Query spanning end of Jan to early Feb
        chunks = generate_n_days_dict("2024-01-28", "2024-02-04", n_days=5)
        assert 2024 in chunks
        year_chunks = chunks[2024]
        # Month boundary must enforce separation: Jan chunk ends on 31_01, Feb chunk starts on 01_02
        assert ("28_01", "31_01") in year_chunks
        assert year_chunks[("28_01", "31_01")] == ["28", "29", "30", "31"]
        assert ("01_02", "04_02") in year_chunks
        assert year_chunks[("01_02", "04_02")] == ["01", "02", "03", "04"]

    def test_year_crossing_chunks(self) -> None:
        # Query spanning end of 2023 to early 2024
        chunks = generate_n_days_dict("2023-12-28", "2024-01-04", n_days=5)
        assert 2023 in chunks
        assert 2024 in chunks
        # 2023 chunk
        assert ("28_12", "31_12") in chunks[2023]
        assert chunks[2023][("28_12", "31_12")] == ["28", "29", "30", "31"]
        # 2024 chunk
        assert ("01_01", "04_01") in chunks[2024]
        assert chunks[2024][("01_01", "04_01")] == ["01", "02", "03", "04"]

    def test_leap_year_february_29(self) -> None:
        # 2024 is a leap year (Feb 29 exists)
        chunks = generate_n_days_dict("2024-02-26", "2024-03-02", n_days=5)
        assert 2024 in chunks
        year_chunks = chunks[2024]
        # February chunk must include day 29 and end on 29_02
        assert ("26_02", "29_02") in year_chunks
        assert "29" in year_chunks[("26_02", "29_02")]
        assert year_chunks[("26_02", "29_02")] == ["26", "27", "28", "29"]
        # March chunk starts on 01_03
        assert ("01_03", "02_03") in year_chunks
        assert year_chunks[("01_03", "02_03")] == ["01", "02"]

    def test_non_leap_year_february_28(self) -> None:
        # 2023 is a non-leap year (Feb 28 is last day)
        chunks = generate_n_days_dict("2023-02-26", "2023-03-02", n_days=5)
        assert 2023 in chunks
        year_chunks = chunks[2023]
        assert ("26_02", "28_02") in year_chunks
        assert "29" not in year_chunks[("26_02", "28_02")]
        assert year_chunks[("26_02", "28_02")] == ["26", "27", "28"]
        assert ("01_03", "02_03") in year_chunks

    def test_temporal_range_split_by_month_leap_year(self) -> None:
        t_range = TemporalRange.from_dates("2024-02-15", "2024-03-15")
        windows = t_range.split_by_month()
        assert len(windows) == 2
        feb_start, feb_end = windows[0]
        mar_start, mar_end = windows[1]
        assert feb_end.day == 29
        assert feb_end.month == 2
        assert mar_start.day == 1
        assert mar_start.month == 3

    def test_parse_date_range_support_dates_and_timestamps(self) -> None:
        s1, e1 = parse_date_range("2024-01-01", "2024-01-10")
        assert isinstance(s1, pd.Timestamp)
        assert isinstance(e1, pd.Timestamp)

        s2, e2 = parse_date_range(date(2024, 2, 28), date(2024, 3, 1))
        assert s2.day == 28 and s2.month == 2
        assert e2.day == 1 and e2.month == 3


class TestGRIBHandlingAndVariables:
    """Tests verifying instantaneous vs. accumulated ERA5 variable differentiation."""

    def test_variable_sets_categorization(self) -> None:
        # Instantaneous variables
        assert "2m_temperature" in ERA5_INSTANTANEOUS_VARIABLES
        assert "10m_u_component_of_wind" in ERA5_INSTANTANEOUS_VARIABLES
        assert "surface_pressure" in ERA5_INSTANTANEOUS_VARIABLES
        assert "total_cloud_cover" in ERA5_INSTANTANEOUS_VARIABLES

        # Accumulated variables
        assert "total_precipitation" in ERA5_ACCUMULATED_VARIABLES
        assert "convective_precipitation" in ERA5_ACCUMULATED_VARIABLES
        assert "surface_solar_radiation_downwards" in ERA5_ACCUMULATED_VARIABLES
        assert "surface_sensible_heat_flux" in ERA5_ACCUMULATED_VARIABLES

        # No intersection between sets
        assert len(ERA5_INSTANTANEOUS_VARIABLES.intersection(ERA5_ACCUMULATED_VARIABLES)) == 0

    def test_open_era5_grib_merging_instant_and_accum(self, tmp_path: Path):
        # Create mock synthetic datasets simulating instant and accum GRIB messages
        times = pd.date_range("2024-02-28 00:00", periods=4, freq="1h", name="time")
        lats = np.array([0.0, 1.0], dtype=np.float32)
        lons = np.array([36.0, 37.0], dtype=np.float32)

        ds_inst = xr.Dataset(
            data_vars={"t2m": (("time", "lat", "lon"), np.random.randn(4, 2, 2).astype(np.float32))},
            coords={"time": times, "lat": lats, "lon": lons},
            attrs={"GRIB_stepType": "instant"},
        )
        ds_accum = xr.Dataset(
            data_vars={"tp": (("time", "lat", "lon"), np.random.rand(4, 2, 2).astype(np.float32))},
            coords={"time": times, "lat": lats, "lon": lons},
            attrs={"GRIB_stepType": "accum"},
        )

        dummy_file = tmp_path / "test_dummy.grib"
        dummy_file.touch()

        def mock_open_dataset(p, *args, **kwargs):
            backend_kw = kwargs.get("backend_kwargs", {})
            filter_keys = backend_kw.get("filter_by_keys", {})
            stype = filter_keys.get("stepType")
            if stype == "instant":
                return ds_inst
            elif stype == "accum":
                return ds_accum
            return ds_inst

        with patch("xarray.open_dataset", side_effect=mock_open_dataset):
            merged = open_era5_grib(dummy_file)
            assert "t2m" in merged.data_vars
            assert "tp" in merged.data_vars
            assert "lat" in merged.coords
            assert "lon" in merged.coords
            assert "time" in merged.coords
            assert len(merged.time) == 4


class TestArbitraryVariableRequests:
    """Tests ensuring users can request arbitrary variables supported by providers."""

    def test_era5_cds_arbitrary_variables_routing(self, tmp_path: Path):
        from masharikiweather.extract.apis.era5 import ERA5Extractor

        extractor = ERA5Extractor(output_dir=tmp_path)
        executed_requests = []

        def mock_execute(client, dataset_name, payload, target):
            executed_requests.append((dataset_name, payload))

        with patch.object(extractor, "_execute_cds_request", side_effect=mock_execute), \
             patch("cdsapi.Client") as mock_cds:
            extractor.extract_cds_slice(
                bbox=SpatialBoundingBox(lat_min=-1.0, lon_min=36.0, lat_max=1.0, lon_max=38.0),
                start_date="2023-01-01",
                end_date="2023-01-01",
                variables=["custom_single_var_xyz", "geopotential", "another_unknown_provider_var"],
                data_levels="both",
            )

        # geopotential is in pressure_fields, the other two are routed dynamically to single levels
        assert len(executed_requests) == 2
        single_req = next(r[1] for r in executed_requests if r[0] == "reanalysis-era5-single-levels")
        pressure_req = next(r[1] for r in executed_requests if r[0] == "reanalysis-era5-pressure-levels")
        assert "custom_single_var_xyz" in single_req["variable"]
        assert "another_unknown_provider_var" in single_req["variable"]
        assert "geopotential" in pressure_req["variable"]

    def test_open_meteo_arbitrary_variable_passthrough(self, tmp_path: Path):
        from masharikiweather.extract.apis.reanalysis import OpenMeteo

        client = OpenMeteo(cache_dir=str(tmp_path))
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b'{"hourly": {"time": ["2023-01-01T00:00"], "any_custom_provider_var": [42.0]}}'
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            records = client.fetch_point(
                lat=0.0,
                lon=37.0,
                start_date="2023-01-01",
                end_date="2023-01-01",
                variables=["any_custom_provider_var"],
            )
            assert len(records) > 0
            assert records[0]["variable"] == "any_custom_provider_var"
            assert records[0]["value"] == 42.0

