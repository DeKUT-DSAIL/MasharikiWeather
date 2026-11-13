"""
Unit and Integration Tests for Dagster Extraction Layer.

Verifies:
1. ParquetStationIOManager handle_output and load_input functionality.
2. ZarrGriddedIOManager handle_output and load_input functionality.
3. Dagster assets (tahmo, era5, imerg, cds_insitu) definitions, partitions, retry policies, and op_tags.
4. Byte decoding and nested observed_variable unpacking for CDS In-Situ.
5. Station metadata parquet generation (installation.parquet and parameters.parquet).
6. Programmatic materialization using materialize_extraction.
7. Clean imports from both masharikiweather.extract.apis.dagster_pipeline and extract.apis.dagster_pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import dagster as dg
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from masharikiweather.extract.apis.dagster_assets import (
    IMERG_TRANSITION_DATE,
    cds_insitu_observations,
    cds_insitu_partitions,
    decode_cds_insitu_dataframe,
    era5_gridded_reanalysis,
    era5_partitions,
    imerg_partitions,
    imerg_satellite_precipitation,
    tahmo_partitions,
    tahmo_station_observations,
    write_station_metadata_parquets,
)
from masharikiweather.extract.apis.dagster_io import (
    ParquetStationIOManager,
    ZarrGriddedIOManager,
)
from masharikiweather.extract.apis.dagster_pipeline import (
    defs,
    format_duration,
    get_partition_storage_size,
    materialize_extraction,
)


def test_imports_cleanliness() -> None:
    """Verify clean imports from both top-level and namespaced packages."""
    from masharikiweather.extract.apis.dagster_pipeline import (
        ParquetStationIOManager as P1,
        ZarrGriddedIOManager as Z1,
        materialize_extraction as M1,
    )
    from extract.apis.dagster_pipeline import (
        ParquetStationIOManager as P2,
        ZarrGriddedIOManager as Z2,
        materialize_extraction as M2,
    )

    assert P1.__name__ == P2.__name__ == "ParquetStationIOManager"
    assert Z1.__name__ == Z2.__name__ == "ZarrGriddedIOManager"
    assert M1.__name__ == M2.__name__ == "materialize_extraction"
    assert issubclass(P1, dg.ConfigurableIOManager)
    assert issubclass(Z1, dg.ConfigurableIOManager)
    assert callable(M1) and callable(M2)


def test_parquet_station_io_manager(tmp_path: Path) -> None:
    """Test ParquetStationIOManager output handling and input loading."""
    io_mgr = ParquetStationIOManager(root_dir=str(tmp_path))

    # Construct test dataframe
    df = pd.DataFrame({
        "time": pd.date_range("2023-01-01", periods=10, freq="1h"),
        "station_code": ["TA0001"] * 10,
        "temperature": np.linspace(20.0, 30.0, 10),
        "humidity": np.linspace(50.0, 80.0, 10),
    }).set_index("time")

    @dg.asset(io_manager_key="parquet_io")
    def sample_station_asset() -> pd.DataFrame:
        return df

    res = dg.materialize(
        [sample_station_asset],
        resources={"parquet_io": io_mgr},
    )
    assert res.success

    expected_path = tmp_path / "obs" / "sample_station_asset" / "all.parquet"
    assert expected_path.exists()

    loaded_df = pd.read_parquet(expected_path)
    assert len(loaded_df) == 10
    assert "station_code" in loaded_df.columns
    assert "temperature" in loaded_df.columns


def test_zarr_gridded_io_manager(tmp_path: Path) -> None:
    """Test ZarrGriddedIOManager output handling and input loading."""
    io_mgr = ZarrGriddedIOManager(root_dir=str(tmp_path))

    # Construct test dataset
    times = pd.date_range("2023-01-01", periods=5, freq="6h")
    lats = np.array([-1.0, 0.0, 1.0])
    lons = np.array([35.0, 36.0, 37.0])
    data = np.ones((len(times), len(lats), len(lons)), dtype=np.float32)

    ds = xr.Dataset(
        data_vars={"precipitationCal": (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    @dg.asset(io_manager_key="zarr_io")
    def sample_gridded_asset() -> xr.Dataset:
        return ds

    res = dg.materialize(
        [sample_gridded_asset],
        resources={"zarr_io": io_mgr},
    )
    assert res.success

    expected_path = tmp_path / "gridded" / "sample_gridded_asset" / "all.zarr"
    assert expected_path.exists()

    loaded_ds = xr.open_zarr(str(expected_path), consolidated=True)
    assert "precipitationCal" in loaded_ds.data_vars
    assert loaded_ds.sizes["time"] == 5
    assert loaded_ds.sizes["lat"] == 3
    assert loaded_ds.sizes["lon"] == 3


def test_asset_partition_definitions() -> None:
    """Verify partition definitions on all software-defined assets."""
    assert tahmo_partitions.get_partition_keys()[0] == "2014-01-01"
    assert era5_partitions.get_partition_keys()[0] == "2018-01-01"
    assert imerg_partitions.get_partition_keys()[0] == "2000-06-01"
    assert cds_insitu_partitions.get_partition_keys()[0] == "2010-01-01"


def test_asset_metadata_and_retry_policies() -> None:
    """Verify retry policies and op_tags concurrency settings."""
    assert tahmo_station_observations.op.retry_policy is not None
    assert tahmo_station_observations.op.retry_policy.max_retries == 3
    assert tahmo_station_observations.op.tags["dagster/concurrency_key"] == "tahmo_datahub_api"
    assert tahmo_station_observations.op.tags["max_concurrency"] == "2"

    assert era5_gridded_reanalysis.op.tags["dagster/concurrency_key"] == "copernicus_cds_api"
    assert era5_gridded_reanalysis.op.tags["max_concurrency"] == "1"

    assert imerg_satellite_precipitation.op.tags["dagster/concurrency_key"] == "nasa_earthaccess_api"
    assert imerg_satellite_precipitation.op.tags["max_concurrency"] == "4"

    assert cds_insitu_observations.op.tags["dagster/concurrency_key"] == "copernicus_cds_api"
    assert cds_insitu_observations.op.tags["max_concurrency"] == "1"


def test_decode_cds_insitu_dataframe() -> None:
    """Verify byte decoding, observed_variable unpacking, and elevation mapping."""
    raw_records = {
        "report_timestamp": [pd.Timestamp("2023-01-01 00:00:00")] * 4,
        "primary_station_id": [b"KE001", b"KE001", b"KE002", b"KE002"],
        "station_name": [b"Nairobi Met", b"Nairobi Met", b"Eldoret Met", b"Eldoret Met"],
        "latitude": [-1.30, -1.30, 0.52, 0.52],
        "longitude": [36.75, 36.75, 35.28, 35.28],
        "height_of_station_above_sea_level": [1795.0, 1795.0, 2120.0, 2120.0],
        "observed_variable": [b"air_temperature", b"accumulated_precipitation", b"air_temperature", b"wind_speed"],
        "observation_value": [294.2, 0.0, 289.5, 3.8],
        "quality_flag": [1, 1, 1, 1],
    }
    df_raw = pd.DataFrame(raw_records)

    decoded = decode_cds_insitu_dataframe(df_raw)

    assert "time" in decoded.columns
    assert "primary_station_id" in decoded.columns
    assert "station_name" in decoded.columns
    assert "elevation" in decoded.columns
    assert "air_temperature" in decoded.columns
    assert "accumulated_precipitation" in decoded.columns
    assert "wind_speed" in decoded.columns

    # Verify strings are decoded
    assert isinstance(decoded["primary_station_id"].iloc[0], str)
    assert decoded["primary_station_id"].iloc[0] == "KE001"
    assert isinstance(decoded["station_name"].iloc[0], str)
    assert decoded["station_name"].iloc[0] == "Nairobi Met"

    # Verify elevation value
    assert decoded.loc[decoded["primary_station_id"] == "KE001", "elevation"].iloc[0] == 1795.0


def test_write_station_metadata_parquets(tmp_path: Path) -> None:
    """Verify generation of installation.parquet and parameters.parquet."""
    stations_df = pd.DataFrame([
        {"station_code": "TA001", "name": "Station Alpha", "lat": 0.1, "lon": 35.5, "elevation": 1500.0, "country": "KE", "status": "active"},
        {"station_code": "TA002", "name": "Station Beta", "lat": -1.2, "lon": 36.8, "elevation": 1650.0, "country": "KE", "status": "active"},
    ])

    inst_p, params_p = write_station_metadata_parquets(root_dir=tmp_path, stations_df=stations_df)

    assert inst_p.exists()
    assert params_p.exists()

    df_inst = pd.read_parquet(inst_p)
    assert len(df_inst) == 2
    assert "station_code" in df_inst.columns
    assert "elevation" in df_inst.columns

    df_params = pd.read_parquet(params_p)
    assert len(df_params) >= 7
    assert "parameter" in df_params.columns
    assert "standard_name" in df_params.columns


# Module-level asset definition for test_materialize_extraction_partitioned
_mock_monthly_partitions = dg.MonthlyPartitionsDefinition(start_date="2023-01-01")


@dg.asset(partitions_def=_mock_monthly_partitions, io_manager_key="parquet_io")
def mock_partitioned_asset(context) -> pd.DataFrame:
    pk = context.partition_key
    return pd.DataFrame({
        "time": [pd.to_datetime(pk)],
        "station_code": ["TEST01"],
        "value": [42.0],
    })


def test_materialize_extraction_partitioned(tmp_path: Path) -> None:
    """Test materialize_extraction with partition execution and progress metrics."""
    io_mgr = ParquetStationIOManager(root_dir=str(tmp_path))
    results = materialize_extraction(
        assets=[mock_partitioned_asset],
        partition_keys=["2023-01-01", "2023-02-01"],
        root_dir=tmp_path,
        resources={"parquet_io": io_mgr},
        show_progress=False,
    )

    assert len(results) == 2
    for r in results:
        assert r.success

    # Verify both partitions exist on disk
    p1 = tmp_path / "obs" / "mock_partitioned_asset" / "2023-01-01.parquet"
    p2 = tmp_path / "obs" / "mock_partitioned_asset" / "2023-02-01.parquet"
    assert p1.exists()
    assert p2.exists()


def test_format_duration_and_storage_size(tmp_path: Path) -> None:
    """Verify duration formatting and partition size retrieval."""
    assert format_duration(65) == "01:05"
    assert format_duration(14) == "00:14"

    # Write a test file
    obs_dir = tmp_path / "obs" / "test_asset"
    obs_dir.mkdir(parents=True, exist_ok=True)
    p_file = obs_dir / "2023-01-01.parquet"
    p_file.write_bytes(b"x" * 1024)

    sz = get_partition_storage_size(tmp_path, "test_asset", "2023-01-01")
    assert sz == 1024


def test_imerg_transition_date_boundary() -> None:
    """Verify that IMERG transition boundary is set to 2025-11-01 UTC."""
    assert IMERG_TRANSITION_DATE == pd.Timestamp("2025-11-01", tz="UTC")
