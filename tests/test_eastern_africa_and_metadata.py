"""
Unit tests for Eastern Africa spatial domain, 12-country definitions,
non-NaN station elevation metadata extraction, and clean omission of
start/end dates for sources without operational date records (GHCN, CDS In-Situ).
"""

from __future__ import annotations

import pathlib
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

from masharikiweather import (
    MasharikiWeatherDataset,
    SpatialBoundingBox,
    AFRICA_BBOX,
    EAST_AFRICA_BBOX,
    EASTERN_AFRICA_BBOX,
    EAST_AFRICA_COUNTRIES,
    EASTERN_AFRICA_COUNTRIES,
    is_eastern_africa_country,
    parse_spatial_bbox,
)


class TestEasternAfricaDomainAndCountries:
    """Test Eastern Africa bounding box coordinates and 12-country definitions."""

    def test_eastern_africa_bbox_coordinates(self) -> None:
        """Verify cartographic domain 17E - 52E, 18S - 19N."""
        assert EASTERN_AFRICA_BBOX.lat_min == -18.0
        assert EASTERN_AFRICA_BBOX.lat_max == 19.0
        assert EASTERN_AFRICA_BBOX.lon_min == 17.0
        assert EASTERN_AFRICA_BBOX.lon_max == 52.0
        assert EAST_AFRICA_BBOX == EASTERN_AFRICA_BBOX

    def test_eastern_africa_12_countries_inventory(self) -> None:
        """Verify the 12 canonical countries defined in AI4Weather."""
        expected_codes = {"KE", "UG", "TZ", "RW", "BI", "ET", "SS", "CD", "ER", "DJ", "MW", "SO"}
        assert set(EASTERN_AFRICA_COUNTRIES.keys()) == expected_codes
        assert EAST_AFRICA_COUNTRIES == EASTERN_AFRICA_COUNTRIES

    def test_is_eastern_africa_country_helper(self) -> None:
        """Verify country identification helper for codes, names, and variations."""
        for code, name in EASTERN_AFRICA_COUNTRIES.items():
            assert is_eastern_africa_country(code)
            assert is_eastern_africa_country(name)
            assert is_eastern_africa_country(name.lower())

        # Test specific aliases and variations
        assert is_eastern_africa_country("DR Congo")
        assert is_eastern_africa_country("DRC")
        assert is_eastern_africa_country("Democratic Republic of the Congo")

        # Non-Eastern Africa countries
        assert not is_eastern_africa_country("US")
        assert not is_eastern_africa_country("United States")
        assert not is_eastern_africa_country("ZA")
        assert not is_eastern_africa_country("South Africa")
        assert not is_eastern_africa_country("Nigeria")
        assert not is_eastern_africa_country(None)
        assert not is_eastern_africa_country("")

    def test_parse_spatial_bbox_presets(self) -> None:
        """Verify string preset parsing for bounding boxes."""
        assert parse_spatial_bbox("eastern_africa") == EASTERN_AFRICA_BBOX
        assert parse_spatial_bbox("east_africa") == EAST_AFRICA_BBOX
        assert parse_spatial_bbox("africa") == AFRICA_BBOX
        assert parse_spatial_bbox(None) is None

        # Sequence format [lat_min, lon_min, lat_max, lon_max]
        custom = parse_spatial_bbox([-5.0, 33.0, 5.0, 42.0])
        assert custom.lat_min == -5.0
        assert custom.lon_min == 33.0
        assert custom.lat_max == 5.0
        assert custom.lon_max == 42.0

        with pytest.raises(ValueError):
            parse_spatial_bbox("unknown_region")


class TestStationMetadataElevationAndDates:
    """Test station elevation is never NaN and dates are cleanly omitted when not stored."""

    def test_station_elevation_pulled_from_metadata_aliases(self, tmp_path: pathlib.Path) -> None:
        """Verify elevation resolves from metadata aliases (elevationmsl, elev, etc.) and is never NaN."""
        obs_dir = tmp_path / "obs" / "tahmo" / "metadata"
        obs_dir.mkdir(parents=True, exist_ok=True)

        st_df = pd.DataFrame(
            {
                "station_id": ["TA00001", "TA00002", "TA00003"],
                "station_name": ["Station 1", "Station 2", "Station 3"],
                "latitude": [0.0, 1.0, -1.0],
                "longitude": [36.0, 37.0, 35.0],
                "elevationmsl": [1550.0, np.nan, 2100.0],
                "elevation": [np.nan, 1820.0, np.nan],
                "network": ["tahmo", "tahmo", "tahmo"],
                "country": ["KE", "KE", "KE"],
                "start_date": [pd.Timestamp("2023-01-01")] * 3,
                "end_date": [pd.Timestamp("2023-01-10")] * 3,
                "sensor_count": [12, 12, 12],
            }
        ).set_index("station_id")
        st_df.to_parquet(obs_dir / "installation.parquet")

        ds = MasharikiWeatherDataset(
            root_dir=tmp_path,
            source_obs=["tahmo"],
            use_hf=False,
        )

        assert "elevation" in ds.stations_table.columns
        assert pd.notna(ds.stations_table["elevation"]).all(), "Elevation must never contain NaNs"
        assert ds.stations_table.loc["TA00001", "elevation"] == 1550.0
        assert ds.stations_table.loc["TA00002", "elevation"] == 1820.0
        assert ds.stations_table.loc["TA00003", "elevation"] == 2100.0

    def test_ghcnd_omits_start_and_end_dates_when_not_stored(self, tmp_path: pathlib.Path) -> None:
        """Verify GHCN stations_table omits start_date and end_date columns when not provided by metadata."""
        obs_dir = tmp_path / "obs" / "ghcn" / "metadata"
        obs_dir.mkdir(parents=True, exist_ok=True)

        # GHCN installation file without start_date / end_date
        st_df = pd.DataFrame(
            {
                "station_id": ["KE000063612", "KE000063740"],
                "station_name": ["Lodwar", "Nairobi"],
                "latitude": [3.12, -1.32],
                "longitude": [35.61, 36.92],
                "elevation": [506.0, 1624.0],
                "network": ["ghcn_daily", "ghcn_daily"],
                "country": ["KE", "KE"],
                "sensor_count": [3, 3],
            }
        ).set_index("station_id")
        st_df.to_parquet(obs_dir / "installation.parquet")

        ds = MasharikiWeatherDataset(
            root_dir=tmp_path,
            source_obs=["ghcn"],
            use_hf=False,
        )

        assert "elevation" in ds.stations_table.columns
        assert pd.notna(ds.stations_table["elevation"]).all(), "Elevation must never be NaN"
        assert ds.stations_table.loc["KE000063612", "elevation"] == 506.0
        assert ds.stations_table.loc["KE000063740", "elevation"] == 1624.0

        # start_date and end_date columns should be omitted entirely rather than being fake or NaN
        assert "start_date" not in ds.stations_table.columns
        assert "end_date" not in ds.stations_table.columns

    def test_cds_insitu_omits_start_and_end_dates_when_not_stored(self, tmp_path: pathlib.Path) -> None:
        """Verify CDS In-Situ stations_table omits start_date and end_date when not provided by metadata."""
        obs_dir = tmp_path / "obs" / "cds_insitu" / "metadata"
        obs_dir.mkdir(parents=True, exist_ok=True)

        st_df = pd.DataFrame(
            {
                "station_id": ["CDS001", "CDS002"],
                "station_name": ["Mombasa", "Kisumu"],
                "latitude": [-4.05, -0.10],
                "longitude": [39.67, 34.75],
                "elevation": [16.0, 1146.0],
                "network": ["cds_insitu", "cds_insitu"],
                "country": ["KE", "KE"],
                "sensor_count": [1, 1],
            }
        ).set_index("station_id")
        st_df.to_parquet(obs_dir / "installation.parquet")

        ds = MasharikiWeatherDataset(
            root_dir=tmp_path,
            source_obs=["cds_insitu"],
            use_hf=False,
        )

        assert "elevation" in ds.stations_table.columns
        assert pd.notna(ds.stations_table["elevation"]).all(), "Elevation must never be NaN"
        assert ds.stations_table.loc["CDS001", "elevation"] == 16.0
        assert ds.stations_table.loc["CDS002", "elevation"] == 1146.0

        assert "start_date" not in ds.stations_table.columns
        assert "end_date" not in ds.stations_table.columns
