"""
Tests for per-source observation accessors of MasharikiWeatherDataset.

All data are synthetic and written to temporary directories.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from masharikiweather.dataset import MasharikiWeatherDataset

YEAR = 2023
SOURCES: Dict[str, List[str]] = {
    "tahmo": ["TA00001", "TA00002"],
    "ghcn": ["KE000063612"],
    "cds_insitu": ["CDS_NBO"],
}


def _long_frame(stations: List[str], seed: int, tz: bool) -> pd.DataFrame:
    """Generate synthetic long-format observations with standard test columns."""
    times = pd.date_range(f"{YEAR}-01-01", periods=48, freq="h", tz="UTC" if tz else None)
    rng = np.random.default_rng(seed)
    rows: List[pd.DataFrame] = []
    for st in stations:
        rows.append(
            pd.DataFrame(
                {
                    "time": times,
                    "station_id": st,
                    "temperature": rng.normal(22.0, 2.0, len(times)),
                    "precipitation": rng.gamma(0.5, 1.0, len(times)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Populate temporary directory with synthetic TAHMO, GHCN, and CDS observations."""
    for i, (src, stations) in enumerate(SOURCES.items()):
        d = tmp_path / "obs" / src
        d.mkdir(parents=True)
        _long_frame(stations, i, tz=(src == "cds_insitu")).to_parquet(
            d / f"{YEAR}.parquet", index=False
        )
    return tmp_path


@pytest.fixture
def ds(root: Path) -> MasharikiWeatherDataset:
    """Instantiate dataset containing all three observation strands."""
    return MasharikiWeatherDataset(
        root_dir=root,
        years=[YEAR],
        source_obs=list(SOURCES),
        freq="h",
        test_set_start=pd.Timestamp(f"{YEAR}-01-02 00:00:00"),
    )


def _check_layout(df: pd.DataFrame) -> None:
    """Assert standard observational layout contracts."""
    assert df.index.name == "time"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is None
    assert df.columns.nlevels == 2
    assert list(df.columns.names) == ["station_id", "parameter"]


def test_each_source_returns_only_its_stations(ds: MasharikiWeatherDataset) -> None:
    """Verify each accessor restricts returned columns exclusively to its stations."""
    got: Dict[str, pd.DataFrame] = {
        "tahmo": ds.get_tahmo_observations(),
        "ghcn": ds.get_ghcn_observations(),
        "cds_insitu": ds.get_cds_insitu_observations(),
    }
    for src, df in got.items():
        _check_layout(df)
        assert sorted(df.columns.get_level_values(0).unique()) == sorted(SOURCES[src])
        assert not df.columns.duplicated().any()


def test_get_tahmo_observations_contract(ds: MasharikiWeatherDataset) -> None:
    """Verify full feature surface of get_tahmo_observations."""
    # 1. Default retrieval
    tahmo_df = ds.get_tahmo_observations()
    _check_layout(tahmo_df)
    assert set(tahmo_df.columns.get_level_values(0)) == set(SOURCES["tahmo"])

    # 2. Station and parameter filtering
    filtered_df = ds.get_tahmo_observations(
        stations=["TA00001"],
        parameters="temperature",
    )
    _check_layout(filtered_df)
    assert list(filtered_df.columns) == [("TA00001", "2m_temperature")]

    # 3. Date range slicing and validity mask
    obs, mask = ds.get_tahmo_observations(
        stations=["TA00001"],
        parameters="temperature",
        first_date=f"{YEAR}-01-01 06:00",
        last_date=f"{YEAR}-01-01 12:00",
        return_mask=True,
    )
    _check_layout(obs)
    _check_layout(mask)
    assert obs.shape == mask.shape
    assert mask.dtypes.eq(bool).all()
    assert obs.index.min() >= pd.Timestamp(f"{YEAR}-01-01 06:00")
    assert obs.index.max() <= pd.Timestamp(f"{YEAR}-01-01 12:00")

    # 4. 3D NumPy array export
    arr, arr_mask = ds.get_tahmo_observations(as_numpy=True, return_mask=True)
    assert isinstance(arr, np.ndarray)
    assert isinstance(arr_mask, np.ndarray)
    assert arr.shape == (48, 2, 2)
    assert arr_mask.shape == (48, 2, 2)
    assert arr_mask.dtype == bool

    # 5. Temporal partitioning (train vs test split)
    train_df = ds.get_tahmo_observations(split="train")
    test_df = ds.get_tahmo_observations(split="test")
    _check_layout(train_df)
    _check_layout(test_df)
    assert train_df.index.max() < pd.Timestamp(f"{YEAR}-01-02 00:00:00")
    assert test_df.index.min() >= pd.Timestamp(f"{YEAR}-01-02 00:00:00")
    assert len(train_df) + len(test_df) == len(tahmo_df)

    # 6. Copy isolation
    copied = ds.get_tahmo_observations(copy=True)
    copied.iloc[0, 0] = -999.0
    assert ds.get_tahmo_observations().iloc[0, 0] != -999.0


def test_get_ghcn_observations_contract(ds: MasharikiWeatherDataset) -> None:
    """Verify full feature surface of get_ghcn_observations."""
    ghcn_df = ds.get_ghcn_observations()
    _check_layout(ghcn_df)
    assert set(ghcn_df.columns.get_level_values(0)) == set(SOURCES["ghcn"])

    # Filtering by parameter and return_mask
    obs, mask = ds.get_ghcn_observations(
        parameters=["precipitation"],
        first_date=f"{YEAR}-01-01 00:00",
        last_date=f"{YEAR}-01-01 23:00",
        return_mask=True,
    )
    _check_layout(obs)
    assert list(obs.columns) == [("KE000063612", "total_precipitation")]
    assert len(obs) == 23
    assert mask.shape == obs.shape

    # NumPy export shape: (T=48, N=1, C=2)
    arr = ds.get_ghcn_observations(as_numpy=True)
    assert arr.shape == (48, 1, 2)

    # Train / test partitioning
    train_df = ds.get_ghcn_observations(split="train")
    test_df = ds.get_ghcn_observations(split="test")
    assert len(train_df) + len(test_df) == 48


def test_get_cds_insitu_observations_contract(ds: MasharikiWeatherDataset) -> None:
    """Verify full feature surface of get_cds_insitu_observations."""
    cds_df = ds.get_cds_insitu_observations()
    _check_layout(cds_df)
    assert set(cds_df.columns.get_level_values(0)) == set(SOURCES["cds_insitu"])

    # Slicing and mask
    obs, mask = ds.get_cds_insitu_observations(
        parameters="temperature",
        first_date=f"{YEAR}-01-01 10:00",
        last_date=f"{YEAR}-01-01 14:00",
        return_mask=True,
    )
    _check_layout(obs)
    assert list(obs.columns) == [("CDS_NBO", "2m_temperature")]
    assert len(obs) == 4

    # NumPy export
    arr = ds.get_cds_insitu_observations(as_numpy=True)
    assert arr.shape == (48, 1, 2)

    # Train / test partitioning
    train_df = ds.get_cds_insitu_observations(split="train")
    test_df = ds.get_cds_insitu_observations(split="test")
    assert len(train_df) + len(test_df) == 48


def test_observation_layout_contract(ds: MasharikiWeatherDataset, root: Path) -> None:
    """Verify strict adherence to DatetimeIndex and MultiIndex contracts across all accessors."""
    accessors = [
        ds.get_tahmo_observations,
        ds.get_ghcn_observations,
        ds.get_cds_insitu_observations,
    ]
    for fn in accessors:
        df = fn()
        _check_layout(df)
        assert df.columns.names == ["station_id", "parameter"]
        assert df.index.name == "time"
        assert df.index.tz is None

    # Empty frame contract when dataset does not include a given source
    only_ghcn = MasharikiWeatherDataset(root_dir=root, years=[YEAR], source_obs="ghcn")
    empty_tahmo = only_ghcn.get_tahmo_observations()
    empty_cds = only_ghcn.get_cds_insitu_observations()
    for empty_df in [empty_tahmo, empty_cds]:
        assert empty_df.empty
        _check_layout(empty_df)


def test_cross_analysis_merging_and_alignment(ds: MasharikiWeatherDataset) -> None:
    """Verify concatenation, join, and align_observations across all strands."""
    strands = {
        "tahmo": ds.get_tahmo_observations(),
        "ghcn": ds.get_ghcn_observations(),
        "cds_insitu": ds.get_cds_insitu_observations(),
    }

    # 1. pd.concat across strands produces 3-level columns
    joined = pd.concat(strands, axis=1)
    assert joined.index.name == "time"
    assert isinstance(joined.index, pd.DatetimeIndex)
    assert joined.index.tz is None
    assert joined.columns.nlevels == 3
    assert len(joined) == 48

    # 2. MasharikiWeatherDataset.align_observations outer alignment
    aligned_outer = MasharikiWeatherDataset.align_observations(strands, how="outer")
    assert aligned_outer.columns.names == ["source", "station_id", "parameter"]
    assert aligned_outer.index.name == "time"
    assert aligned_outer.index.tz is None
    assert aligned_outer.shape[1] == sum(s.shape[1] for s in strands.values())

    # 3. MasharikiWeatherDataset.align_observations inner alignment
    aligned_inner = MasharikiWeatherDataset.align_observations(strands, how="inner")
    assert len(aligned_inner) == 48
    assert aligned_inner.shape[1] == aligned_outer.shape[1]

    # 4. DataFrame join idiom
    joined_cols = strands["tahmo"].join(strands["ghcn"], how="outer")
    assert joined_cols.index.name == "time"
    assert joined_cols.index.tz is None

    # 5. Empty strands handling
    empty_aligned = MasharikiWeatherDataset.align_observations({})
    assert empty_aligned.empty
    assert empty_aligned.index.name == "time"
    assert isinstance(empty_aligned.index, pd.DatetimeIndex)


def test_multi_strand_plotting_idiom(ds: MasharikiWeatherDataset) -> None:
    """Simulate plotting series from TAHMO, GHCN, and CDS In-Situ on the same axes."""
    tahmo_df = ds.get_tahmo_observations()
    ghcn_df = ds.get_ghcn_observations()
    cds_df = ds.get_cds_insitu_observations()

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot individual series from each strand
    ax.plot(
        tahmo_df.index,
        tahmo_df[("TA00001", "2m_temperature")],
        label="TAHMO TA00001 Temp",
        color="blue",
    )
    ax.plot(
        ghcn_df.index,
        ghcn_df[("KE000063612", "2m_temperature")],
        label="GHCN KE000063612 Temp",
        color="orange",
    )
    ax.plot(
        cds_df.index,
        cds_df[("CDS_NBO", "2m_temperature")],
        label="CDS CDS_NBO Temp",
        color="green",
    )

    ax.set_title("Multi-Strand Temperature Comparison")
    ax.set_xlabel("Time (UTC naive)")
    ax.set_ylabel("Temperature (degC)")
    ax.legend(loc="upper right")

    # Verify canvas draw succeeds without timezone or dimensional conflicts
    fig.canvas.draw()
    plt.close(fig)

    # Plot from aligned DataFrame
    strands = {
        "tahmo": tahmo_df,
        "ghcn": ghcn_df,
        "cds_insitu": cds_df,
    }
    aligned = MasharikiWeatherDataset.align_observations(strands)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for src, st, param in aligned.columns:
        if param == "2m_temperature":
            ax2.plot(aligned.index, aligned[(src, st, param)], label=f"{src}:{st}")
    ax2.legend()
    fig2.canvas.draw()
    plt.close(fig2)


def test_tahmo_zarr_loading_prefer_zarr(tmp_path: Path) -> None:
    """Verify loading from local obs/tahmo.zarr store using prefer_zarr=True."""
    times = pd.date_range(f"{YEAR}-01-01", periods=24, freq="h")
    stations = ["TA00001", "TA00002"]
    rng = np.random.default_rng(42)
    zds = xr.Dataset(
        {
            "temperature": (("time", "station_id"), rng.normal(21.0, 1.5, (24, 2))),
            "precipitation": (("time", "station_id"), rng.exponential(0.5, (24, 2))),
        },
        coords={"time": times, "station_id": stations},
    )
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir(parents=True, exist_ok=True)
    zds.to_zarr(obs_dir / "tahmo.zarr")

    ds_zarr = MasharikiWeatherDataset(
        root_dir=tmp_path,
        years=[YEAR],
        source_obs=["tahmo"],
        prefer_zarr=True,
    )
    assert ds_zarr.prefer_zarr is True
    out = ds_zarr.get_tahmo_observations()
    _check_layout(out)
    assert sorted(out.columns.get_level_values(0).unique()) == sorted(stations)
    assert len(out) == 24
    assert ("TA00001", "2m_temperature") in out.columns


def test_tahmo_zarr_loading_use_zarr_kwarg(tmp_path: Path) -> None:
    """Verify loading from local obs/tahmo.zarr store using use_zarr=True kwarg."""
    times = pd.date_range(f"{YEAR}-01-01", periods=24, freq="h")
    stations = ["TA00016"]
    rng = np.random.default_rng(99)
    zds = xr.Dataset(
        {"temperature": (("time", "station_id"), rng.normal(24.0, 1.0, (24, 1)))},
        coords={"time": times, "station_id": stations},
    )
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir(parents=True, exist_ok=True)
    zds.to_zarr(obs_dir / "tahmo.zarr")

    ds_zarr = MasharikiWeatherDataset(
        root_dir=tmp_path,
        years=[YEAR],
        source_obs="tahmo",
        use_zarr=True,
    )
    assert ds_zarr.prefer_zarr is True
    out = ds_zarr.get_tahmo_observations()
    _check_layout(out)
    assert sorted(out.columns.get_level_values(0).unique()) == ["TA00016"]
    assert len(out) == 24


def test_tahmo_zarr_priority_over_parquet(tmp_path: Path) -> None:
    """Verify prefer_zarr chooses Zarr values over Parquet when both exist."""
    times = pd.date_range(f"{YEAR}-01-01", periods=12, freq="h")
    stations = ["TA00001"]

    # 1. Write Parquet with temperature = 10.0
    p_dir = tmp_path / "obs" / "tahmo"
    p_dir.mkdir(parents=True, exist_ok=True)
    df_p = pd.DataFrame(
        {
            "time": times,
            "station_id": "TA00001",
            "temperature": np.full(12, 10.0),
        }
    )
    df_p.to_parquet(p_dir / f"{YEAR}.parquet", index=False)

    # 2. Write Zarr with temperature = 99.0
    zds = xr.Dataset(
        {"temperature": (("time", "station_id"), np.full((12, 1), 99.0))},
        coords={"time": times, "station_id": stations},
    )
    zds.to_zarr(tmp_path / "obs" / "tahmo.zarr")

    # With prefer_zarr=True, should read 99.0 from Zarr
    ds_prefer_zarr = MasharikiWeatherDataset(
        root_dir=tmp_path, years=[YEAR], source_obs=["tahmo"], prefer_zarr=True
    )
    val_zarr = ds_prefer_zarr.get_tahmo_observations()[("TA00001", "2m_temperature")].iloc[0]
    assert np.isclose(val_zarr, 99.0)

    # With prefer_zarr=False, should read 10.0 from Parquet
    ds_prefer_parquet = MasharikiWeatherDataset(
        root_dir=tmp_path, years=[YEAR], source_obs=["tahmo"], prefer_zarr=False
    )
    val_parquet = ds_prefer_parquet.get_tahmo_observations()[("TA00001", "2m_temperature")].iloc[0]
    assert np.isclose(val_parquet, 10.0)


def test_alias_resolution_ghcnd(root: Path) -> None:
    """Verify source_obs=['ghcnd'] and string aliases resolve to get_ghcn_observations()."""
    # List alias: ['ghcnd']
    ds_list = MasharikiWeatherDataset(root_dir=root, years=[YEAR], source_obs=["ghcnd"])
    ghcn_df = ds_list.get_ghcn_observations()
    _check_layout(ghcn_df)
    assert not ghcn_df.empty
    assert list(ghcn_df.columns.get_level_values(0).unique()) == ["KE000063612"]
    assert ds_list.get_tahmo_observations().empty

    # String alias: 'ghcnd'
    ds_str = MasharikiWeatherDataset(root_dir=root, years=[YEAR], source_obs="ghcnd")
    assert not ds_str.get_ghcn_observations().empty

    # Resolution alias: ['ghcn_daily']
    ds_daily = MasharikiWeatherDataset(root_dir=root, years=[YEAR], source_obs=["ghcn_daily"])
    assert not ds_daily.get_ghcn_observations().empty


def test_alias_resolution_cds(root: Path) -> None:
    """Verify source_obs=['cds'] and string aliases resolve to get_cds_insitu_observations()."""
    # List alias: ['cds']
    ds_list = MasharikiWeatherDataset(root_dir=root, years=[YEAR], source_obs=["cds"])
    cds_df = ds_list.get_cds_insitu_observations()
    _check_layout(cds_df)
    assert not cds_df.empty
    assert list(cds_df.columns.get_level_values(0).unique()) == ["CDS_NBO"]
    assert ds_list.get_ghcn_observations().empty

    # String alias: 'cds'
    ds_str = MasharikiWeatherDataset(root_dir=root, years=[YEAR], source_obs="cds")
    assert not ds_str.get_cds_insitu_observations().empty

    # Alternate alias: ['insitu']
    ds_insitu = MasharikiWeatherDataset(root_dir=root, years=[YEAR], source_obs=["insitu"])
    assert not ds_insitu.get_cds_insitu_observations().empty


def test_alias_resolution_mixed(root: Path) -> None:
    """Verify multi-source initialization with aliases resolves all accessors simultaneously."""
    ds_mixed = MasharikiWeatherDataset(
        root_dir=root,
        years=[YEAR],
        source_obs=["tahmo", "ghcnd", "cds"],
    )
    tahmo_df = ds_mixed.get_tahmo_observations()
    ghcn_df = ds_mixed.get_ghcn_observations()
    cds_df = ds_mixed.get_cds_insitu_observations()

    _check_layout(tahmo_df)
    _check_layout(ghcn_df)
    _check_layout(cds_df)

    assert sorted(tahmo_df.columns.get_level_values(0).unique()) == sorted(SOURCES["tahmo"])
    assert sorted(ghcn_df.columns.get_level_values(0).unique()) == sorted(SOURCES["ghcn"])
    assert sorted(cds_df.columns.get_level_values(0).unique()) == sorted(SOURCES["cds_insitu"])


def test_source_without_data_returns_empty_frame(root: Path) -> None:
    """Verify requesting an unconfigured source returns an empty frame matching layout."""
    only_tahmo = MasharikiWeatherDataset(root_dir=root, years=[YEAR], source_obs="tahmo")
    empty = only_tahmo.get_ghcn_observations()
    assert empty.empty
    _check_layout(empty)
    assert not only_tahmo.get_tahmo_observations().empty


def test_get_observations_still_contains_all_sources(ds: MasharikiWeatherDataset) -> None:
    """Verify get_observations retains columns across all active observation strands."""
    all_obs = ds.get_observations()
    n = sum(len(v) for v in SOURCES.values()) * 2
    assert all_obs.shape[1] == n


def test_zarr_fallback_for_tahmo(root: Path) -> None:
    """Verify legacy explicit _obs_zarr fallback behavior."""
    times = pd.date_range(f"{YEAR}-01-01", periods=24, freq="h")
    stations = ["TA00001", "TA00016"]
    rng = np.random.default_rng(1)
    zds = xr.Dataset(
        {
            "temperature": (("time", "station_id"), rng.normal(20, 1, (24, 2))),
            "precipitation": (("time", "station_id"), rng.random((24, 2))),
        },
        coords={"time": times, "station_id": stations},
    )
    for f in (root / "obs" / "tahmo").glob("*.parquet"):
        f.unlink()
    ds = MasharikiWeatherDataset(
        root_dir=root, years=[YEAR], source_obs=["tahmo"], prefer_zarr=True
    )
    assert ds.prefer_zarr is True
    ds._obs_zarr["tahmo"] = zds
    ds._prepare_observations()
    assert sorted(ds.observations.columns.get_level_values(0).unique()) == sorted(stations)
    ds.stations_table = pd.DataFrame(index=pd.Index(stations, name="station_id"))
    out = ds.get_tahmo_observations()
    _check_layout(out)
    assert sorted(out.columns.get_level_values(0).unique()) == sorted(stations)
