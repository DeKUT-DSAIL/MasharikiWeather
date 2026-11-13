"""
Tests for documentation examples and end-to-end workflows.

Validates that all code snippets and workflows documented in
docs/dataset_user_guide.md, docs/extraction.md, and docs/credentials.md
execute reliably and deterministically against local synthetic data fixtures.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from masharikiweather.dataset import (
    MasharikiWeatherDataset,
    ObservationDataFrame,
    SpatialBoundingBox,
    canonicalize_variable_name,
    extract_gridded,
    extract_stations,
    pull_weather_data,
)
from masharikiweather.extract.apis import (
    CredentialsManager,
    ERA5Extractor,
    TAHMOExtractor,
    get_credentials,
)


@pytest.fixture
def doc_data_root(tmp_path: Path) -> Path:
    """Populate temporary directory with complete synthetic test catalogs and observations."""
    (tmp_path / "obs" / "tahmo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "obs" / "cds_insitu").mkdir(parents=True, exist_ok=True)
    (tmp_path / "obs" / "ghcn").mkdir(parents=True, exist_ok=True)
    (tmp_path / "metadata").mkdir(parents=True, exist_ok=True)
    (tmp_path / "gridded" / "era5").mkdir(parents=True, exist_ok=True)
    (tmp_path / "topography").mkdir(parents=True, exist_ok=True)

    stations_df = pd.DataFrame(
        {
            "station_name": ["Nairobi East", "Eldoret High", "Mombasa Coastal"],
            "latitude": [-1.28, 0.51, -4.04],
            "longitude": [36.82, 35.28, 39.66],
            "elevation": [1661.0, 2085.0, 15.0],
            "network": ["tahmo", "tahmo", "tahmo"],
        },
        index=pd.Index(["TA00001", "TA00016", "TA00168"], name="station_id"),
    )
    stations_df.to_parquet(tmp_path / "metadata" / "installation.parquet")

    params_df = pd.DataFrame(
        {
            "standard_name": [
                "air_temperature",
                "precipitation_flux",
                "surface_air_pressure",
                "surface_downwelling_shortwave_flux_in_air",
                "relative_humidity",
                "dew_point_temperature",
                "wind_speed",
            ],
            "units": ["degC", "mm", "hPa", "W m-2", "%", "degC", "m s-1"],
            "resampling_method": ["mean", "sum", "mean", "sum", "mean", "mean", "mean"],
        },
        index=pd.Index(
            [
                "2m_temperature",
                "total_precipitation",
                "surface_pressure",
                "surface_solar_radiation_downwards",
                "relative_humidity",
                "2m_dewpoint_temperature",
                "wind_speed",
            ],
            name="parameter",
        ),
    )
    params_df.to_parquet(tmp_path / "metadata" / "parameters.parquet")

    times = pd.date_range("2023-01-01 00:00:00", "2023-01-07 23:00:00", freq="h")
    rng = np.random.default_rng(42)

    tahmo_rows = []
    for st in ["TA00001", "TA00016", "TA00168"]:
        temp = 20.0 + 5.0 * np.sin(np.linspace(0, 7 * 2 * np.pi, len(times))) + rng.normal(0, 0.5, len(times))
        precip = rng.exponential(scale=0.5, size=len(times))
        precip[precip < 0.2] = 0.0
        solar = np.maximum(0.0, 800.0 * np.sin(np.linspace(0, 7 * np.pi, len(times))))
        rh = np.clip(60.0 + 20.0 * np.cos(np.linspace(0, 7 * 2 * np.pi, len(times))), 10.0, 100.0)
        pres = 1013.25 - 0.12 * (1661.0 if st == "TA00001" else 2085.0) + rng.normal(0, 1, len(times))

        df_st = pd.DataFrame(
            {
                "time": times,
                "station_id": st,
                "temperature": temp,
                "precipitation": precip,
                "solar_radiation": solar,
                "relative_humidity": rh,
                "surface_pressure": pres,
            }
        )
        tahmo_rows.append(df_st)
    pd.concat(tahmo_rows, ignore_index=True).to_parquet(
        tmp_path / "obs" / "tahmo" / "2023.parquet", index=False
    )

    cds_rows = []
    for st in ["CDS_NBO"]:
        temp = 21.0 + rng.normal(0, 2, len(times))
        pres = 1013.25 + rng.normal(0, 3, len(times))
        cds_rows.append(
            pd.DataFrame({"time": times, "station_id": st, "temperature": temp, "surface_pressure": pres})
        )
    pd.concat(cds_rows, ignore_index=True).to_parquet(
        tmp_path / "obs" / "cds_insitu" / "2023.parquet", index=False
    )

    ghcn_rows = []
    for st in ["KE000063612"]:
        temp = 19.5 + rng.normal(0, 1.5, len(times))
        precip = rng.exponential(scale=0.4, size=len(times))
        precip[precip < 0.2] = 0.0
        ghcn_rows.append(
            pd.DataFrame({"time": times, "station_id": st, "temperature": temp, "precipitation": precip})
        )
    pd.concat(ghcn_rows, ignore_index=True).to_parquet(
        tmp_path / "obs" / "ghcn" / "2023.parquet", index=False
    )

    lats = np.linspace(-5.0, 5.0, 11)
    lons = np.linspace(33.0, 42.0, 10)
    gridded_times = times[:48]
    era5_ds = xr.Dataset(
        data_vars={
            "2m_temperature": (
                ("time", "latitude", "longitude"),
                rng.normal(295.15, 3.0, (len(gridded_times), len(lats), len(lons))),
            ),
            "total_precipitation": (
                ("time", "latitude", "longitude"),
                rng.exponential(0.001, (len(gridded_times), len(lats), len(lons))),
            ),
            "surface_pressure": (
                ("time", "latitude", "longitude"),
                rng.normal(101325.0, 500.0, (len(gridded_times), len(lats), len(lons))),
            ),
        },
        coords={
            "time": gridded_times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    era5_ds.to_zarr(tmp_path / "gridded" / "era5" / "2023.zarr", mode="w")

    topo_ds = xr.Dataset(
        data_vars={
            "elevation": (("latitude", "longitude"), rng.uniform(500, 3000, (len(lats), len(lons)))),
            "slope": (("latitude", "longitude"), rng.uniform(0, 30, (len(lats), len(lons)))),
            "aspect": (("latitude", "longitude"), rng.uniform(0, 360, (len(lats), len(lons)))),
            "TPI_2000M": (("latitude", "longitude"), rng.uniform(-1, 1, (len(lats), len(lons)))),
        },
        coords={
            "latitude": lats,
            "longitude": lons,
        },
    )
    topo_ds.to_netcdf(tmp_path / "topography" / "africa_static_priors.nc")

    return tmp_path


def test_doc_getting_started_and_properties(doc_data_root: Path) -> None:
    """Verify Section 1 and Section 2 dataset instantiation and properties."""
    dataset = MasharikiWeatherDataset(
        root=doc_data_root,
        years=[2023],
        freq="h",
        aggregation_methods={"precipitation": "sum", "temperature": "mean"},
    )
    assert dataset.root == doc_data_root
    assert dataset.freq == "h"
    assert "TA00001" in dataset.stations
    assert "TA00016" in dataset.stations
    assert len(dataset.stations) == 3

    assert isinstance(dataset.installation, pd.DataFrame)
    assert not dataset.installation.empty
    assert "latitude" in dataset.installation.columns
    assert "longitude" in dataset.installation.columns
    assert "elevation" in dataset.installation.columns

    assert isinstance(dataset.parameters_table, pd.DataFrame)
    assert not dataset.parameters_table.empty


def test_doc_station_observations_and_alignment(doc_data_root: Path) -> None:
    """Verify Section 3 and Section 4 observation retrieval and source alignment."""
    dataset = MasharikiWeatherDataset(
        root=doc_data_root,
        years=[2023],
        source_obs=["tahmo", "cds_insitu", "ghcn"],
    )

    # 3.1 TAHMO
    df_tahmo, mask_tahmo = dataset.get_tahmo_observations(
        stations=["TA00001", "TA00016"],
        parameters=["temperature", "precipitation"],
        first_date="2023-01-01 00:00",
        last_date="2023-01-07 23:00",
        return_mask=True,
    )
    assert not df_tahmo.empty
    assert df_tahmo.shape == mask_tahmo.shape
    assert "TA00001" in df_tahmo.columns.get_level_values(0)

    # 3.2 Copernicus In-Situ
    df_cds = dataset.get_cds_insitu_observations(
        stations=["CDS_NBO"],
        parameters=["temperature", "surface_pressure"],
        first_date="2023-01-01",
        last_date="2023-01-07",
    )
    assert not df_cds.empty
    assert "CDS_NBO" in df_cds.columns.get_level_values(0)

    # 3.3 NOAA GHCN
    df_ghcn = dataset.get_ghcn_observations(
        stations=["KE000063612"],
        parameters=["temperature", "precipitation"],
        first_date="2023-01-01",
        last_date="2023-01-07",
    )
    assert not df_ghcn.empty
    assert "KE000063612" in df_ghcn.columns.get_level_values(0)

    # 4.1 Align observations
    strands = {
        "tahmo": dataset.get_tahmo_observations(parameters=["temperature"]),
        "ghcn": dataset.get_ghcn_observations(parameters=["temperature"]),
        "cds_insitu": dataset.get_cds_insitu_observations(parameters=["temperature"]),
    }
    aligned_df = MasharikiWeatherDataset.align_observations(strands, how="outer")
    assert not aligned_df.empty
    assert aligned_df.columns.names == ["source", "station_id", "parameter"]


def test_doc_resampling_conservation_and_topography(doc_data_root: Path) -> None:
    """Verify Section 6 resampling conservation rules and Section 8 topography loading."""
    dataset = MasharikiWeatherDataset(root=doc_data_root, years=[2023], source_obs=["tahmo"])

    # 6. Temporal resampling: flux sum vs state mean
    dataset_3h = dataset.resample(freq="3h")
    assert dataset_3h.freq == "3h"

    df_orig = dataset.get_tahmo_observations(stations=["TA00001"])
    df_3h = dataset_3h.get_tahmo_observations(stations=["TA00001"])

    precip_col_orig = [c for c in df_orig.columns if "precip" in c[1]][0]
    precip_col_3h = [c for c in df_3h.columns if "precip" in c[1]][0]
    assert np.isclose(df_orig[precip_col_orig].sum(), df_3h[precip_col_3h].sum(), atol=1e-4)

    temp_col_orig = [c for c in df_orig.columns if "temp" in c[1]][0]
    temp_col_3h = [c for c in df_3h.columns if "temp" in c[1]][0]
    assert np.isclose(df_orig[temp_col_orig].mean(), df_3h[temp_col_3h].mean(), atol=1e-4)

    # 8. Topography loading and aspect decomposition
    topo_ds = dataset.load_topography(filename="africa_static_priors")
    assert topo_ds is not None
    assert "elevation" in topo_ds.data_vars
    assert "aspect_sin" in topo_ds.data_vars
    assert "aspect_cos" in topo_ds.data_vars


def test_doc_gridded_extraction_and_station_interpolation(doc_data_root: Path) -> None:
    """Verify Section 9 gridded loading and station interpolation."""
    dataset = MasharikiWeatherDataset(root=doc_data_root, years=[2023])
    gridded_dict = dataset.load_gridded_datasets(groups=["era5"], years=[2023])
    assert "era5" in gridded_dict
    assert "2m_temperature" in gridded_dict["era5"].data_vars

    gridded_station_df, mask_df = dataset.get_gridded_for_stations(
        groups=["era5"],
        stations=["TA00001", "TA00016"],
        variables=["total_precipitation", "2m_temperature"],
        method_overrides={"total_precipitation": "linear", "2m_temperature": "nearest"},
        first_date="2023-01-01",
        last_date="2023-01-02",
    )
    assert "era5" in gridded_station_df
    assert not gridded_station_df["era5"].empty
    assert "TA00001" in gridded_station_df["era5"].columns.get_level_values(0)


def test_doc_credentials_and_pull_weather_data(doc_data_root: Path) -> None:
    """Verify credentials resolution and pull_weather_data orchestration."""
    extractor = ERA5Extractor(cds_key="test-key-doc")
    assert extractor.cds_key == "test-key-doc"

    tahmo = TAHMOExtractor(api_key="t-key", api_secret="t-secret")
    assert tahmo.api_key == "t-key"
    assert tahmo.api_secret == "t-secret"

    with patch.dict(os.environ, {"CDSAPI_KEY": "env-key"}):
        creds_arg = CredentialsManager.get_cds_credentials(key="override-key")
        assert creds_arg["key"] == "override-key"

        creds_env = CredentialsManager.get_cds_credentials()
        assert creds_env["key"] == "env-key"

    africa_box = SpatialBoundingBox(lat_min=-35.0, lat_max=38.0, lon_min=-20.0, lon_max=55.0)
    with patch("masharikiweather.dataset.extract_stations") as mock_st, \
         patch("masharikiweather.dataset.extract_gridded") as mock_gr:
        mock_st.return_value = ["/path/st.parquet"]
        mock_gr.return_value = "/path/gr.zarr"

        result = pull_weather_data(
            start_date="2023-01-01",
            end_date="2023-01-03",
            bbox=africa_box,
            station_source="tahmo",
            gridded_source="open_meteo",
            output_dir=doc_data_root,
        )
        assert result["stations"] == ["/path/st.parquet"]
        assert result["gridded"] == "/path/gr.zarr"
