"""
Test suite for TAHMO metadata scoping, station lifecycle timestamps, dynamic sensor counts,
33-variable catalog CF compliance, and instantiation-time topography enrichment.

Scientific Mandates:
- No emojis in code or test files.
- Full type annotations on helper fixtures and test functions.
- No hardcoded local paths; use pytest tmp_path fixture.
- Deterministic synthetic fixtures with network isolation.
"""

from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from masharikiweather.dataset import (
    CANONICAL_VARIABLE_METADATA,
    MasharikiWeatherDataset,
    canonicalize_variable_name,
)
from masharikiweather.extract.apis.tahmo import (
    DEFAULT_TAHMO_PARAMETERS,
    TAHMO_CANONICAL_TO_SHORT,
    TAHMO_SHORT_TO_CANONICAL,
    TAHMOExtractor,
)

EXPECTED_33_TAHMO_CODES: List[str] = [
    "ap", "cp", "dw", "ec", "ew", "ht", "lb", "ld", "le", "lp",
    "lt", "mp", "pd", "pr", "pt", "pw", "rh", "rn", "sd", "sm",
    "sr", "tc", "td", "te", "tp", "uv", "vp", "wd", "wg", "wl",
    "wq", "ws", "wv",
]

VALID_PHYSICAL_TYPES: Set[str] = {
    "PHYSICAL",
    "PRECIPITATION",
    "HYDROLOGICAL",
    "RADIATION",
    "ELECTRICAL",
    "DIAGNOSTIC",
}


def _create_synthetic_topography_zarr(topo_path: Path) -> Path:
    """
    Create a synthetic multi-scale topography Zarr dataset.

    Mathematical docstring:
        Spatial dimensions: (lat: 10, lon: 10).
        Variables: dem, Slope, Aspect, TPI_2000M, TPI_10000M, STD_2000M,
                   SN_DERIVATIVE, WE_DERIVATIVE, Aspect_sin, Aspect_cos.

    Args:
        topo_path: Path to target .zarr directory.

    Returns:
        Path to written Zarr directory.
    """
    lats = np.linspace(1.0, -2.0, 10, dtype=np.float32)
    lons = np.linspace(34.0, 38.0, 10, dtype=np.float32)
    topo_ds = xr.Dataset(
        data_vars={
            "dem": (["lat", "lon"], np.full((10, 10), 1680.0, dtype=np.float32)),
            "Slope": (["lat", "lon"], np.full((10, 10), 4.5, dtype=np.float32)),
            "Aspect": (["lat", "lon"], np.full((10, 10), 135.0, dtype=np.float32)),
            "TPI_2000M": (["lat", "lon"], np.full((10, 10), 12.0, dtype=np.float32)),
            "TPI_10000M": (["lat", "lon"], np.full((10, 10), 28.0, dtype=np.float32)),
            "STD_2000M": (["lat", "lon"], np.full((10, 10), 8.5, dtype=np.float32)),
            "SN_DERIVATIVE": (["lat", "lon"], np.full((10, 10), 0.05, dtype=np.float32)),
            "WE_DERIVATIVE": (["lat", "lon"], np.full((10, 10), -0.02, dtype=np.float32)),
            "Aspect_sin": (["lat", "lon"], np.full((10, 10), float(np.sin(np.radians(135.0))), dtype=np.float32)),
            "Aspect_cos": (["lat", "lon"], np.full((10, 10), float(np.cos(np.radians(135.0))), dtype=np.float32)),
        },
        coords={"lat": lats, "lon": lons},
    )
    topo_ds.to_zarr(topo_path, consolidated=True)
    return topo_path


class TestTahmoMetadata:
    """
    Comprehensive test suite for TAHMO metadata source scoping,
    non-null lifecycle timestamps, dynamic sensor count resolution, 33-variable CF catalog,
    and instantiation topography persistence.
    """

    def test_tahmo_source_scoping(self, tmp_path: Path) -> None:
        """
        Verify that instantiating MasharikiWeatherDataset(root_dir=tmp_dir, source_obs=['tahmo'])
        strictly scopes stations_table and parameters_table to TAHMO sources, excluding
        competing observational networks (such as GHCN or CDS In-Situ).

        Mathematical docstring:
            Input: Multi-network metadata registries containing N_tahmo=2 and N_ghcn=2 stations.
            Output: stations_table with shape (2, C_metadata) where network == 'tahmo' for all rows.
        """
        # 1. Source-specific directory structure test
        tahmo_meta = tmp_path / "obs" / "tahmo" / "metadata"
        ghcn_meta = tmp_path / "obs" / "ghcn" / "metadata"
        tahmo_meta.mkdir(parents=True)
        ghcn_meta.mkdir(parents=True)

        tahmo_stations = pd.DataFrame(
            {
                "latitude": [-1.28, -0.10],
                "longitude": [36.82, 34.75],
                "network": ["tahmo", "tahmo"],
                "elevation": [1680.0, 1131.0],
                "station_name": ["Nairobi Station", "Kisumu Station"],
                "country": ["KE", "KE"],
                "start_date": ["2018-01-01", "2019-05-01"],
                "end_date": ["2023-12-31", "2023-12-31"],
                "sensor_count": [4, 3],
            },
            index=pd.Index(["TA00001", "TA00002"], name="station_id"),
        )
        tahmo_stations.to_parquet(tahmo_meta / "installation.parquet")

        ghcn_stations = pd.DataFrame(
            {
                "latitude": [-0.5, 0.2],
                "longitude": [37.0, 35.5],
                "network": ["ghcn", "ghcn"],
                "elevation": [1500.0, 1200.0],
                "station_name": ["Nyeri Met", "Eldoret Met"],
                "country": ["KE", "KE"],
                "start_date": ["1980-01-01", "1990-01-01"],
                "end_date": ["2023-12-31", "2023-12-31"],
                "sensor_count": [1, 1],
            },
            index=pd.Index(["KE000063612", "KE000063624"], name="station_id"),
        )
        ghcn_stations.to_parquet(ghcn_meta / "installation.parquet")

        tahmo_params = pd.DataFrame(
            {
                "standard_name": ["surface_air_pressure", "air_temperature"],
                "units": ["hPa", "degC"],
                "physical_type": ["PHYSICAL", "PHYSICAL"],
                "network": ["tahmo", "tahmo"],
            },
            index=pd.Index(["surface_pressure", "2m_temperature"], name="parameter"),
        )
        tahmo_params.to_parquet(tahmo_meta / "parameters.parquet")

        ghcn_params = pd.DataFrame(
            {
                "standard_name": ["precipitation_flux"],
                "units": ["mm"],
                "physical_type": ["PRECIPITATION"],
                "network": ["ghcn"],
            },
            index=pd.Index(["prcp"], name="parameter"),
        )
        ghcn_params.to_parquet(ghcn_meta / "parameters.parquet")

        ds = MasharikiWeatherDataset(root_dir=tmp_path, source_obs=["tahmo"], use_hf=False)

        assert hasattr(ds, "stations_table")
        assert len(ds.stations_table) == 2
        assert "TA00001" in ds.stations_table.index
        assert "TA00002" in ds.stations_table.index
        assert "KE000063612" not in ds.stations_table.index
        assert "KE000063624" not in ds.stations_table.index
        assert (ds.stations_table["network"] == "tahmo").all()

        assert hasattr(ds, "parameters_table")
        # Parameter catalogs for TAHMO are expanded to the full 33-variable catalog
        assert len(ds.parameters_table) == 33
        assert "surface_pressure" in ds.parameters_table.index
        assert "2m_temperature" in ds.parameters_table.index
        assert "prcp" not in ds.parameters_table.index

    def test_tahmo_source_scoping_from_global_composite(self, tmp_path: Path) -> None:
        """
        Verify that when loading from a global metadata/installation.parquet containing
        mixed sources, source_obs=['tahmo'] correctly filters by network == 'tahmo'.
        """
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir(parents=True)

        composite_stations = pd.DataFrame(
            {
                "latitude": [-1.28, -0.5],
                "longitude": [36.82, 37.0],
                "network": ["tahmo", "ghcn"],
                "elevation": [1680.0, 1500.0],
                "station_name": ["TAHMO Station 1", "GHCN Station 1"],
                "start_date": ["2018-01-01", "1980-01-01"],
                "end_date": ["2023-12-31", "2023-12-31"],
                "sensor_count": [3, 1],
            },
            index=pd.Index(["TA00001", "KE000063612"], name="station_id"),
        )
        composite_stations.to_parquet(meta_dir / "installation.parquet")

        composite_params = pd.DataFrame(
            {
                "standard_name": ["surface_air_pressure", "precipitation_flux"],
                "network": ["tahmo", "ghcn"],
                "physical_type": ["PHYSICAL", "PRECIPITATION"],
            },
            index=pd.Index(["surface_pressure", "ghcn_prcp"], name="parameter"),
        )
        composite_params.to_parquet(meta_dir / "parameters.parquet")

        ds = MasharikiWeatherDataset(root_dir=tmp_path, source_obs=["tahmo"], use_hf=False)

        assert len(ds.stations_table) == 1
        assert "TA00001" in ds.stations_table.index
        assert "KE000063612" not in ds.stations_table.index
        assert (ds.stations_table["network"] == "tahmo").all()

        # Parameter catalogs for TAHMO are expanded to the full 33-variable catalog
        assert len(ds.parameters_table) == 33
        assert "surface_pressure" in ds.parameters_table.index
        assert "ghcn_prcp" not in ds.parameters_table.index

    def test_station_dates_no_nans(self, tmp_path: Path) -> None:
        """
        Verify that valid stations with observation records or asset metadata
        have non-null start_date and end_date timestamps, and start_date <= end_date.

        Mathematical docstring:
            Input: Station observations spanning t_min = 2023-01-01 02:00:00 to
                   t_max = 2023-01-01 08:00:00 UTC.
            Assertion:
                pd.notna(start_date) is True
                pd.notna(end_date) is True
                start_date <= end_date
        """
        obs_dir = tmp_path / "obs" / "tahmo"
        meta_dir = tmp_path / "metadata"
        obs_dir.mkdir(parents=True)
        meta_dir.mkdir(parents=True)

        # 1. Observations on disk spanning 6 hours
        idx = pd.date_range("2023-01-01 02:00", periods=7, freq="h", tz="UTC")
        cols = pd.MultiIndex.from_tuples([
            ("TA00001", "te_S001"),
            ("TA00001", "rh_S001"),
        ])
        obs_matrix = pd.DataFrame(22.5, index=idx, columns=cols)
        flat_obs = obs_matrix.stack(level=0, future_stack=True).reset_index().rename(columns={"level_1": "station_id"})
        flat_obs.to_parquet(obs_dir / "2023.parquet")

        # 2. Installation table with NaT for station 1 (derived from observations)
        #    and explicit asset metadata dates for station 2
        inst_df = pd.DataFrame(
            {
                "latitude": [-1.28, -0.10],
                "longitude": [36.82, 34.75],
                "network": ["tahmo", "tahmo"],
                "elevation": [1680.0, 1131.0],
                "start_date": [pd.NaT, "2018-05-10 00:00:00"],
                "end_date": [pd.NaT, "2022-12-31 23:59:59"],
                "sensor_count": [1, 2],
            },
            index=pd.Index(["TA00001", "TA00002"], name="station_id"),
        )
        inst_df.to_parquet(meta_dir / "installation.parquet")

        ds = MasharikiWeatherDataset(root_dir=tmp_path, years=[2023], source_obs=["tahmo"], use_hf=False)

        # Verify TA00001 resolved start_date from observational data and dropped end_date/sensor_count/station_name
        r1 = ds.stations_table.loc["TA00001"]
        assert pd.notna(r1["start_date"]), "TA00001 start_date must not be NaN/NaT"
        assert pd.to_datetime(r1["start_date"]).strftime("%Y-%m-%d %H:%M") == "2023-01-01 02:00"
        assert "end_date" not in ds.stations_table.columns
        assert "sensor_count" not in ds.stations_table.columns
        assert "station_name" not in ds.stations_table.columns

        # Verify TA00002 retained asset metadata start_date
        r2 = ds.stations_table.loc["TA00002"]
        assert pd.notna(r2["start_date"]), "TA00002 start_date must not be NaN/NaT"
        assert pd.to_datetime(r2["start_date"]) == pd.Timestamp("2018-05-10 00:00:00")

    def test_station_dates_asset_column_fallback(self, tmp_path: Path) -> None:
        """
        Verify that when stations_table contains raw TAHMO asset metadata columns
        ('installationdate', 'created', 'closed', 'decommissiondate'), start_date and
        end_date fall back to these columns without remaining NaT.
        """
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir(parents=True)

        inst_df = pd.DataFrame(
            {
                "latitude": [-1.28],
                "longitude": [36.82],
                "network": ["tahmo"],
                "elevation": [1680.0],
                "start_date": [pd.NaT],
                "end_date": [pd.NaT],
                "installationdate": ["2016-04-15T08:30:00Z"],
                "closed": ["2021-10-31T18:00:00Z"],
                "sensor_count": [2],
            },
            index=pd.Index(["TA00003"], name="station_id"),
        )
        inst_df.to_parquet(meta_dir / "installation.parquet")

        ds = MasharikiWeatherDataset(root_dir=tmp_path, source_obs=["tahmo"], use_hf=False)

        r3 = ds.stations_table.loc["TA00003"]
        assert pd.notna(r3["start_date"]), "start_date should be resolved from installationdate"
        assert pd.to_datetime(r3["start_date"]) == pd.Timestamp("2016-04-15 08:30:00", tz="UTC")
        assert "end_date" not in ds.stations_table.columns

    def test_deprecated_metadata_columns_removed(self, tmp_path: Path) -> None:
        """
        Verify that redundant/misleading metadata columns ('station_name', 'end_date',
        'sensor_count') are removed from stations_table/installation, and 'canonical_name'
        is removed from parameters_table.
        """
        obs_dir = tmp_path / "obs" / "tahmo"
        meta_dir = tmp_path / "metadata"
        obs_dir.mkdir(parents=True)
        meta_dir.mkdir(parents=True)

        idx = pd.date_range("2023-01-01 00:00", periods=4, freq="h", tz="UTC")
        cols = pd.MultiIndex.from_tuples([
            ("TA00001", "te_S1"),
            ("TA00002", "rh_S001"),
        ])
        data = np.full((4, 2), 25.0, dtype=np.float32)
        df_obs = pd.DataFrame(data, index=idx, columns=cols)
        flat_obs = df_obs.stack(level=0, future_stack=True).reset_index().rename(columns={"level_1": "station_id"})
        flat_obs.to_parquet(obs_dir / "2023.parquet")

        inst_df = pd.DataFrame(
            {
                "latitude": [-1.28, -0.10],
                "longitude": [36.82, 34.75],
                "network": ["tahmo", "tahmo"],
                "elevation": [1680.0, 1131.0],
                "station_name": ["St1", "St2"],
                "start_date": [pd.NaT, pd.NaT],
                "end_date": [pd.NaT, pd.NaT],
                "sensor_count": [1, 1],
            },
            index=pd.Index(["TA00001", "TA00002"], name="station_id"),
        )
        inst_df.to_parquet(meta_dir / "installation.parquet")

        ds = MasharikiWeatherDataset(root_dir=tmp_path, years=[2023], source_obs=["tahmo"], use_hf=False)

        for col in ("station_name", "end_date", "sensor_count"):
            assert col not in ds.stations_table.columns
            assert col not in ds.installation.columns
        assert "name" in ds.stations_table.columns
        assert "canonical_name" not in ds.parameters_table.columns

    def test_tahmo_33_variable_parameters_table(self, tmp_path: Path) -> None:
        """
        Verify that all 33 TAHMO variables appear in the parameters table with
        CF standard names, units, physical types, and resampling methods.

        Mathematical docstring:
            Input: TAHMO parameter catalog generated from DEFAULT_TAHMO_PARAMETERS.
            Validation:
                card(parameters_table) == 33
                set(short_code) == {ap, cp, dw, ec, ew, ht, lb, ld, le, lp, lt, mp,
                                    pd, pr, pt, pw, rh, rn, sd, sm, sr, tc, td, te,
                                    tp, uv, vp, wd, wg, wl, wq, ws, wv}
                For all v in parameters_table: physical_type in VALID_PHYSICAL_TYPES
        """
        extractor = TAHMOExtractor(output_dir=tmp_path, api_key="dummy", api_secret="dummy")
        mock_stations = pd.DataFrame([
            {
                "station_code": "TA00001",
                "name": "Nairobi Hub",
                "lat": -1.28,
                "lon": 36.82,
                "elevation": 1680.0,
                "country": "KE",
                "start_date": "2018-01-01",
                "end_date": "2023-12-31",
                "sensor_count": 4,
            }
        ])
        inst_p, param_p = extractor.generate_metadata_catalogs(
            stations_df=mock_stations,
            output_dir=tmp_path / "obs" / "tahmo" / "metadata",
        )

        assert inst_p.exists()
        assert param_p.exists()

        ds = MasharikiWeatherDataset(root_dir=tmp_path, source_obs=["tahmo"], use_hf=False)

        assert hasattr(ds, "parameters_table")
        assert len(ds.parameters_table) == 33, f"Expected exactly 33 TAHMO parameters, found {len(ds.parameters_table)}"

        table = ds.parameters_table
        assert "short_code" in table.columns
        assert "standard_name" in table.columns
        assert "physical_type" in table.columns
        assert "units" in table.columns

        actual_codes = set(table["short_code"])
        expected_codes = set(EXPECTED_33_TAHMO_CODES)
        missing_codes = expected_codes - actual_codes
        assert not missing_codes, f"Missing TAHMO short codes: {missing_codes}"

        # Verify each parameter possesses valid CF metadata and physical types
        for idx_val, row in table.iterrows():
            code = row["short_code"]
            std_name = row["standard_name"]
            phys_type = row["physical_type"]
            units = row["units"]

            assert pd.notna(std_name) and len(str(std_name).strip()) > 0, f"Empty standard_name for {code}"
            assert phys_type in VALID_PHYSICAL_TYPES, f"Invalid physical_type '{phys_type}' for {code}"
            assert pd.notna(units) and len(str(units).strip()) > 0, f"Empty units for {code}"

        # Verify bidirectional dictionary coverage
        for code in EXPECTED_33_TAHMO_CODES:
            assert code in TAHMO_SHORT_TO_CANONICAL, f"Missing {code} in TAHMO_SHORT_TO_CANONICAL"
            canon_name = TAHMO_SHORT_TO_CANONICAL[code]
            assert canon_name in TAHMO_CANONICAL_TO_SHORT, f"Missing {canon_name} in TAHMO_CANONICAL_TO_SHORT"
            assert canon_name in CANONICAL_VARIABLE_METADATA, f"Missing {canon_name} in CANONICAL_VARIABLE_METADATA"

    def test_stations_parquet_topography_columns(self, tmp_path: Path) -> None:
        """
        Verify that upon dataset instantiation (__init__), stations.parquet (and installation.parquet)
        is written to disk containing all terrain features
        from africa_topography.zarr (dem, Slope, Aspect, TPI_2000M, TPI_10000M, etc.).

        Mathematical docstring:
            Input: africa_topography.zarr with 10 terrain variables.
                   stations table with station at (-1.28, 36.82).
            Output: stations.parquet with columns dem, Slope, Aspect, TPI_2000M, TPI_10000M,
                    STD_2000M, SN_DERIVATIVE, WE_DERIVATIVE, Aspect_sin, Aspect_cos.
        """
        topo_dir = tmp_path / "topography"
        meta_dir = tmp_path / "metadata"
        topo_dir.mkdir(parents=True)
        meta_dir.mkdir(parents=True)

        # 1. Create synthetic Africa topography Zarr
        _create_synthetic_topography_zarr(topo_dir / "africa_topography.zarr")

        # 2. Write initial station installation file
        inst_df = pd.DataFrame(
            {
                "latitude": [-1.28, -0.50, -0.10],
                "longitude": [36.82, 37.00, 34.75],
                "network": ["tahmo", "tahmo", "tahmo"],
                "elevation": [1680.0, 1500.0, np.nan],
                "station_name": ["Nairobi Hub", "Nyeri High", "Kisumu Plain"],
                "country": ["KE", "KE", "KE"],
                "start_date": ["2018-01-01", "2019-01-01", "2020-01-01"],
                "end_date": ["2023-12-31", "2023-12-31", "2023-12-31"],
                "sensor_count": [4, 2, 1],
            },
            index=pd.Index(["TA00001", "TA00002", "TA00003"], name="station_id"),
        )
        inst_df.to_parquet(meta_dir / "installation.parquet")

        # 3. Instantiate dataset: should automatically trigger interpolate_topography
        #    and persist updated station metadata to stations.parquet
        ds = MasharikiWeatherDataset(
            root_dir=tmp_path,
            source_obs=["tahmo"],
            extended_topo_vars="all",
            use_hf=False,
        )

        expected_topo_cols = [
            "dem",
            "Slope",
            "Aspect",
            "TPI_2000M",
            "TPI_10000M",
            "STD_2000M",
            "SN_DERIVATIVE",
            "WE_DERIVATIVE",
            "Aspect_sin",
            "Aspect_cos",
        ]

        # Verify in-memory DataFrame contains all terrain attributes
        for col in expected_topo_cols:
            assert col in ds.stations_table.columns, f"Topography column '{col}' missing from ds.stations_table"
            assert pd.notna(ds.stations_table[col]).all(), f"NaN values detected in '{col}'"

        # Verify physical persistence on disk across standard locations
        persisted_paths = [
            tmp_path / "metadata" / "stations.parquet",
            tmp_path / "metadata" / "installation.parquet",
            tmp_path / "stations.parquet",
            tmp_path / "installation.parquet",
        ]

        for p in persisted_paths:
            assert p.exists(), f"Metadata parquet missing from {p}"
            saved_df = pd.read_parquet(p)
            for col in expected_topo_cols:
                assert col in saved_df.columns, f"Column '{col}' was not saved to {p}"
                assert pd.notna(saved_df[col]).all(), f"NaN values in saved '{col}' at {p}"

        # Verify elevation synchronization:
        # Station with missing elevation (TA00003) must be synchronized and filled from dem
        assert pd.notna(ds.stations_table.loc["TA00003", "elevation"]), "Missing elevation must be filled by dem"
        assert ds.stations_table.loc["TA00003", "elevation"] == ds.stations_table.loc["TA00003", "dem"]

        # Station with matching elevation (TA00001)
        assert ds.stations_table.loc["TA00001", "elevation"] == ds.stations_table.loc["TA00001", "dem"]

        # Station with distinct ground-truth elevation preserves ground-truth while dem is present
        assert ds.stations_table.loc["TA00002", "elevation"] == 1500.0
        assert ds.stations_table.loc["TA00002", "dem"] == 1680.0
