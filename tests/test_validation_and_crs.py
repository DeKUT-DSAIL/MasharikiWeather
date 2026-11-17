"""
Unit and Integration Tests for CRS Compliance and Centralized Safe Validation.

Tests:
1. WGS 84 (EPSG:4326) CRS enforcement, CF grid_mapping conventions, and lazy dask preservation.
2. Centralized variable validation with fuzzy matching and cross-source extractor suggestions.
3. Station identifier validation, cross-network mismatch diagnostics (KEMU on TAHMO, TA00185 on GHCN).
4. High-throughput vectorized temporal coverage validation across station catalogs.
5. BaseExtractor subclass validation method integration.
"""

from __future__ import annotations

import difflib
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from masharikiweather.extract.apis.base import (
    CRSComplianceError,
    CRSValidationResult,
    ExtractionValidationError,
    SpatialBoundingBox,
    StationNotFoundError,
    StationValidationResult,
    TemporalCoverageError,
    TemporalRange,
    TemporalValidationResult,
    VariableNotAvailableError,
    VariableValidationResult,
    detect_station_network_heuristic,
    enforce_wgs84_crs,
    find_spatial_dimensions,
    suggest_extractor_for_variable,
    validate_crs_compliance,
    validate_station_identifiers,
    validate_temporal_coverage,
    validate_variables,
)
from masharikiweather.extract.apis.era5 import ERA5Extractor
from masharikiweather.extract.apis.ghcnd import GHCNExtractor
from masharikiweather.extract.apis.imerg import IMERGExtractor
from masharikiweather.extract.apis.tahmo import TAHMOExtractor


# =====================================================================
# 1. CRS Compliance & Grid Mapping Tests
# =====================================================================

class TestCRSCompliance:
    """Test suite for WGS 84 (EPSG:4326) CF-1.8+ grid_mapping conventions."""

    def test_find_spatial_dimensions(self):
        """Test identification of varying latitude and longitude coordinate names."""
        ds1 = xr.Dataset(coords={"latitude": [0.0, 1.0], "longitude": [36.0, 37.0]})
        lat, lon = find_spatial_dimensions(ds1)
        assert lat == "latitude"
        assert lon == "longitude"

        ds2 = xr.Dataset(coords={"lat": [-1.0, 0.0], "lon": [34.0, 35.0]})
        lat2, lon2 = find_spatial_dimensions(ds2)
        assert lat2 == "lat"
        assert lon2 == "lon"

        ds3 = xr.Dataset(coords={"y": [10.0, 20.0], "x": [30.0, 40.0]})
        lat3, lon3 = find_spatial_dimensions(ds3)
        assert lat3 == "y"
        assert lon3 == "x"

    def test_enforce_wgs84_crs_standardization(self):
        """Test standardizing coordinate names, CF attributes, and grid_mapping variable."""
        lats = np.array([-2.0, -1.0, 0.0, 1.0])
        lons = np.array([35.0, 36.0, 37.0, 38.0])
        data = np.random.randn(4, 4)

        ds_raw = xr.Dataset(
            data_vars={"total_precipitation": (("lat", "lon"), data)},
            coords={"lat": lats, "lon": lons},
        )

        ds_cf = enforce_wgs84_crs(ds_raw, standardize_names=True)

        # Coordinate dimension standardization
        assert "latitude" in ds_cf.dims
        assert "longitude" in ds_cf.dims
        assert "lat" not in ds_cf.dims
        assert "lon" not in ds_cf.dims

        # CF coordinate attributes
        assert ds_cf["latitude"].attrs["standard_name"] == "latitude"
        assert ds_cf["latitude"].attrs["units"] == "degrees_north"
        assert ds_cf["latitude"].attrs["axis"] == "Y"

        assert ds_cf["longitude"].attrs["standard_name"] == "longitude"
        assert ds_cf["longitude"].attrs["units"] == "degrees_east"
        assert ds_cf["longitude"].attrs["axis"] == "X"

        # CF grid_mapping container variable
        assert "spatial_ref" in ds_cf
        gm_attrs = ds_cf["spatial_ref"].attrs
        assert gm_attrs["grid_mapping_name"] == "latitude_longitude"
        assert gm_attrs["epsg_code"] == "EPSG:4326"
        assert gm_attrs["semi_major_axis"] == 6378137.0
        assert "spatial_ref" in gm_attrs or "crs_wkt" in gm_attrs

        # Data variable attribute mapping
        assert ds_cf["total_precipitation"].attrs["grid_mapping"] == "spatial_ref"

    def test_enforce_wgs84_crs_longitude_wrapping(self):
        """Test wrapping longitudes from [0, 360) to [-180, 180] and maintaining monotonicity."""
        # Unwrapped longitudes crossing prime meridian
        raw_lons = np.array([350.0, 355.0, 0.0, 5.0, 10.0])
        raw_lats = np.array([-1.0, 0.0, 1.0])
        temp_data = np.arange(15).reshape(3, 5)

        ds = xr.Dataset(
            data_vars={"2m_temperature": (("latitude", "longitude"), temp_data)},
            coords={"latitude": raw_lats, "longitude": raw_lons},
        )

        ds_wrapped = enforce_wgs84_crs(ds, wrap_lon_to_180=True)

        expected_lons = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        np.testing.assert_allclose(ds_wrapped["longitude"].values, expected_lons)
        assert bool(np.all(np.diff(ds_wrapped["longitude"].values) > 0))

    def test_enforce_wgs84_crs_preserves_dask_laziness(self):
        """Test that CRS enforcement preserves lazy Dask arrays without computing them."""
        import dask.array as da

        lats = np.linspace(-4.0, 4.0, 10)
        lons = np.linspace(34.0, 42.0, 10)
        dask_data = da.zeros((10, 10), chunks=(5, 5))

        ds = xr.Dataset(
            data_vars={"wind_speed": (("lat", "lon"), dask_data)},
            coords={"lat": lats, "lon": lons},
        )

        assert isinstance(ds["wind_speed"].data, da.Array)
        ds_cf = enforce_wgs84_crs(ds, standardize_names=True)
        assert isinstance(ds_cf["wind_speed"].data, da.Array)

    def test_validate_crs_compliance_success_and_failure(self):
        """Test CRS compliance validator on valid and non-compliant datasets."""
        # 1. Compliant dataset
        lats = np.linspace(-4.0, 4.0, 5)
        lons = np.linspace(34.0, 42.0, 5)
        ds_valid = xr.Dataset(
            data_vars={"total_precipitation": (("latitude", "longitude"), np.zeros((5, 5)))},
            coords={"latitude": lats, "longitude": lons},
        )
        ds_valid = enforce_wgs84_crs(ds_valid)
        res_valid = validate_crs_compliance(ds_valid, strict=True)
        assert res_valid.is_compliant
        assert res_valid.epsg_code == 4326
        assert res_valid.has_grid_mapping

        # 2. Non-compliant: Out-of-bounds latitude
        ds_bad_lat = xr.Dataset(
            coords={"latitude": [-95.0, 0.0], "longitude": [35.0, 36.0]}
        )
        res_bad_lat = validate_crs_compliance(ds_bad_lat, strict=False)
        assert not res_bad_lat.is_compliant
        assert any("Latitude bounds" in issue for issue in res_bad_lat.issues)

        with pytest.raises(CRSComplianceError):
            validate_crs_compliance(ds_bad_lat, strict=True)


# =====================================================================
# 2. Centralized Variable Validation Tests
# =====================================================================

class TestVariableValidation:
    """Test suite for variable availability checks, fuzzy matching, and cross-source suggestions."""

    def test_variable_validation_success(self):
        """Test validation when all requested variables are available in extractor."""
        res = validate_variables(
            requested_variables=["total_precipitation", "uncalibrated_precipitation"],
            extractor_name="imerg",
            strict=True,
        )
        assert res.is_valid
        assert res.valid_variables == ["total_precipitation", "uncalibrated_precipitation"]
        assert len(res.invalid_variables) == 0

    def test_variable_validation_fuzzy_matching(self):
        """Test fuzzy matching suggestion when a typo is passed."""
        res = validate_variables(
            requested_variables=["preciptation"],
            extractor_name="imerg",
            strict=False,
        )
        assert not res.is_valid
        assert "preciptation" in res.invalid_variables
        assert "preciptation" in res.suggestions
        assert any("precipitation" in s for s in res.suggestions["preciptation"])

    def test_cross_extractor_suggestions_temperature_from_imerg(self):
        """
        Verify that requesting '2m_temperature' from IMERG produces a diagnostic
        identifying that IMERG is precipitation-only and suggests ERA5, TAHMO, GHCN, CDS-InSitu.
        """
        with pytest.raises(VariableNotAvailableError) as exc_info:
            validate_variables(
                requested_variables=["2m_temperature"],
                extractor_name="imerg",
                strict=True,
            )

        err_msg = str(exc_info.value)
        assert "2m_temperature" in err_msg
        assert "imerg" in err_msg.lower()
        # Ensure cross-source recommendations are listed
        assert "era5" in err_msg.lower()
        assert "tahmo" in err_msg.lower()

    def test_variable_validation_non_strict_warning(self):
        """Test that strict=False emits a UserWarning rather than raising an exception."""
        with pytest.warns(UserWarning, match="Variable 'humidity' is not provided"):
            res = validate_variables(
                requested_variables=["total_precipitation", "humidity"],
                extractor_name="imerg",
                strict=False,
            )
            assert not res.is_valid
            assert res.valid_variables == ["total_precipitation"]
            assert res.invalid_variables == ["humidity"]


# =====================================================================
# 3. Station Identifier & Network Diagnostic Tests
# =====================================================================

class TestStationValidation:
    """Test suite for station ID format detection and cross-network mismatch diagnostics."""

    def test_detect_station_network_heuristic(self):
        """Test regex pattern detection for TAHMO, GHCN, CDS, and ICAO/KMD."""
        assert detect_station_network_heuristic("TA00185") == "tahmo"
        assert detect_station_network_heuristic("TA00001") == "tahmo"
        assert detect_station_network_heuristic("KE000063740") == "ghcn"
        assert detect_station_network_heuristic("USW00094728") == "ghcn"
        assert detect_station_network_heuristic("63740") == "cds_insitu"
        assert detect_station_network_heuristic("0-20000-0-63740") == "cds_insitu"
        assert detect_station_network_heuristic("KEMU") == "icao_kmd"
        assert detect_station_network_heuristic("HKMO") == "icao_kmd"

    def test_station_cross_network_mismatch_kemu_on_tahmo(self):
        """
        Verify that requesting KEMU on TAHMO catches that KEMU is an ICAO/KMD airport code
        and suggests GHCNExtractor or CDSInSituExtractor.
        """
        with pytest.raises(StationNotFoundError) as exc_info:
            validate_station_identifiers(
                stations=["KEMU"],
                extractor_network="tahmo",
                strict=True,
            )

        err_msg = str(exc_info.value)
        assert "KEMU" in err_msg
        assert "TAHMO" in err_msg
        assert "Kenya Meteorological Department" in err_msg or "ICAO" in err_msg
        assert "GHCNExtractor" in err_msg or "CDSInSituExtractor" in err_msg

    def test_station_cross_network_mismatch_tahmo_on_ghcn(self):
        """
        Verify that requesting TA00185 on GHCN catches that TA corresponds to TAHMO
        and suggests TAHMOExtractor.
        """
        with pytest.raises(StationNotFoundError) as exc_info:
            validate_station_identifiers(
                stations=["TA00185"],
                extractor_network="ghcnd",
                strict=True,
            )

        err_msg = str(exc_info.value)
        assert "TA00185" in err_msg
        assert "TAHMOExtractor" in err_msg

    def test_vectorized_station_catalog_membership(self):
        """Test validation against a provided catalog DataFrame."""
        catalog = pd.DataFrame({
            "station_id": ["TA00001", "TA00002", "TA00003"],
            "station_name": ["Station A", "Station B", "Station C"],
        })

        # Valid query
        res = validate_station_identifiers(
            stations=["TA00001", "TA00002"],
            valid_catalog=catalog,
            extractor_network="tahmo",
            strict=True,
        )
        assert res.is_valid
        assert res.valid_stations == ["TA00001", "TA00002"]

        # Missing station with typo suggestion
        with pytest.raises(StationNotFoundError) as exc_info:
            validate_station_identifiers(
                stations=["TA00001", "TA00004"],
                valid_catalog=catalog,
                extractor_network="tahmo",
                strict=True,
            )
        assert "TA00004" in str(exc_info.value)


# =====================================================================
# 4. Vectorized Temporal Coverage Tests
# =====================================================================

class TestTemporalCoverageValidation:
    """Test suite for high-throughput vectorized temporal coverage validation."""

    def test_vectorized_temporal_coverage_fractions(self):
        """Test vectorized computation of active, pre-commission, and post-decommission stations."""
        catalog = pd.DataFrame([
            {"station_id": "TA00001", "start_date": "2016-01-01", "end_date": "2020-12-31"},
            {"station_id": "TA00002", "start_date": "2018-01-01", "end_date": "2024-01-01"},
            {"station_id": "TA00003", "start_date": "2010-01-01", "end_date": "2014-12-31"},  # Decommissioned
            {"station_id": "TA00004", "start_date": "2022-01-01", "end_date": "2025-01-01"},  # Pre-commissioned
        ])

        query_range = TemporalRange.from_dates("2015-01-01", "2019-12-31")

        res = validate_temporal_coverage(
            temporal_range=query_range,
            station_catalog=catalog,
            strict=False,
        )

        assert "TA00001" in res.active_stations
        assert "TA00002" in res.active_stations
        assert "TA00003" in res.post_decommission_stations
        assert "TA00004" in res.pre_commission_stations

        # Coverage fractions: TA00001 was active 2016 to 2019 (4 of 5 years = 0.8)
        assert pytest.approx(res.coverage_fractions["TA00001"], rel=1e-2) == 0.8
        # TA00003 and TA00004 had zero coverage
        assert res.coverage_fractions["TA00003"] == 0.0
        assert res.coverage_fractions["TA00004"] == 0.0

    def test_extractor_operational_bounds_check(self):
        """Test bounds check against API earliest/latest operational limits."""
        bounds = (pd.Timestamp("2000-01-01"), pd.Timestamp("2025-01-01"))
        # Range earlier than API existed
        out_of_bounds_range = TemporalRange.from_dates("1990-01-01", "1995-01-01")

        with pytest.raises(TemporalCoverageError):
            validate_temporal_coverage(
                temporal_range=out_of_bounds_range,
                extractor_bounds=bounds,
                strict=True,
            )


# =====================================================================
# 5. BaseExtractor Integration Tests
# =====================================================================

class TestBaseExtractorIntegration:
    """Test extractor instance method integration across IMERG, TAHMO, GHCN, and ERA5."""

    def test_imerg_extractor_rejects_temperature(self, tmp_path: Path):
        """Test that IMERGExtractor.extract rejects temperature with cross-source suggestion."""
        extractor = IMERGExtractor(output_dir=tmp_path)
        with pytest.raises(VariableNotAvailableError) as exc_info:
            extractor.extract(
                start_date="2024-01-01",
                end_date="2024-01-02",
                variables=["2m_temperature"],
            )
        assert "2m_temperature" in str(exc_info.value)
        assert "imerg" in str(exc_info.value).lower()

    def test_tahmo_extractor_rejects_kemu(self, tmp_path: Path):
        """Test that TAHMOExtractor.extract rejects KEMU with ICAO diagnostic."""
        extractor = TAHMOExtractor(output_dir=tmp_path, api_key="dummy", api_secret="dummy")
        with pytest.raises(StationNotFoundError) as exc_info:
            extractor.extract(
                start_date="2024-01-01",
                end_date="2024-01-02",
                stations=["KEMU"],
            )
        assert "KEMU" in str(exc_info.value)
        assert "TAHMO" in str(exc_info.value)

    def test_ghcn_extractor_rejects_tahmo_station(self, tmp_path: Path):
        """Test that GHCNExtractor.extract rejects TA00185 with TAHMO diagnostic."""
        extractor = GHCNExtractor(output_dir=tmp_path)
        with pytest.raises(StationNotFoundError) as exc_info:
            extractor.extract(
                start_date="2024-01-01",
                end_date="2024-01-02",
                stations=["TA00185"],
            )
        assert "TA00185" in str(exc_info.value)
        assert "TAHMOExtractor" in str(exc_info.value)

    def test_base_extractor_enforce_and_validate_crs(self, tmp_path: Path):
        """Test enforce_crs and validate_crs methods on BaseExtractor instance."""
        extractor = ERA5Extractor(output_dir=tmp_path)
        ds = xr.Dataset(
            data_vars={"2m_temperature": (("lat", "lon"), np.zeros((3, 3)))},
            coords={"lat": [-1.0, 0.0, 1.0], "lon": [35.0, 36.0, 37.0]},
        )
        ds_cf = extractor.enforce_crs(ds)
        res = extractor.validate_crs(ds_cf, strict=True)
        assert res.is_compliant
        assert res.has_grid_mapping
