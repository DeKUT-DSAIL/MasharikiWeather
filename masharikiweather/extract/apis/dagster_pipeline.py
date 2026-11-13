"""
Dagster Pipeline Definitions and Programmatic Materialization Runner.

Exposes unified Dagster definitions and high-throughput extraction helpers:
1. defs: Production Dagster Definitions bundle binding assets and IO managers.
2. materialize_extraction: Programmatic execution helper orchestrating partitioned
   ingestion with dynamic progress bars (e.g. '2017.parquet: 56.9MB [00:14, 3.93MB/s]').
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import dagster as dg
import pandas as pd

from .dagster_assets import (
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
from .dagster_io import ParquetStationIOManager, ZarrGriddedIOManager

logger = logging.getLogger("MasharikiWeather.DagsterPipeline")

DEFAULT_ROOT_DIR = "./mashariki_data"

# Standard definitions bundling all software-defined assets and IO managers
defs = dg.Definitions(
    assets=[
        tahmo_station_observations,
        era5_gridded_reanalysis,
        imerg_satellite_precipitation,
        cds_insitu_observations,
    ],
    resources={
        "parquet_io": ParquetStationIOManager(root_dir=DEFAULT_ROOT_DIR),
        "zarr_io": ZarrGriddedIOManager(root_dir=DEFAULT_ROOT_DIR),
    },
)


def format_duration(seconds: float) -> str:
    """Format elapsed seconds into MM:SS string."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def get_partition_storage_size(
    root_dir: Path,
    asset_name: str,
    partition_key: str,
) -> int:
    """
    Locate persisted partition artifact and return size in bytes.
    """
    parquet_p = root_dir / "obs" / asset_name / f"{partition_key}.parquet"
    if parquet_p.exists():
        return parquet_p.stat().st_size

    zarr_p = root_dir / "gridded" / asset_name / f"{partition_key}.zarr"
    if zarr_p.exists():
        total = 0
        for p in zarr_p.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total

    return 0


def materialize_extraction(
    assets: Sequence[Any],
    partition_keys: Optional[Sequence[str]] = None,
    root_dir: Union[str, Path] = DEFAULT_ROOT_DIR,
    resources: Optional[Dict[str, Any]] = None,
    run_config: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,
) -> List[dg.ExecuteInProcessResult]:
    """
    Programmatically materialize software-defined extraction assets with progress reporting.

    Parameters
    ----------
    assets : Sequence[Any]
        Dagster software-defined assets to execute.
    partition_keys : Optional[Sequence[str]], default None
        Specific partition keys to materialize (e.g. ['2023-01-01', '2023-02-01']).
        If None and assets are partitioned, executes single pass or unpartitioned execution.
    root_dir : Union[str, Path], default './mashariki_data'
        Local directory path where 'obs/' and 'gridded/' stores are rooted.
    resources : Optional[Dict[str, Any]], default None
        Custom Dagster resource definitions. If None, defaults to ParquetStationIOManager
        and ZarrGriddedIOManager pointing to root_dir.
    run_config : Optional[Dict[str, Any]], default None
        Optional Dagster run configuration dict.
    show_progress : bool, default True
        Whether to display terminal progress bars and throughput metrics.

    Returns
    -------
    List[dg.ExecuteInProcessResult]
        Execution results per partition or run.
    """
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)

    effective_resources = resources or {
        "parquet_io": ParquetStationIOManager(root_dir=str(root_path)),
        "zarr_io": ZarrGriddedIOManager(root_dir=str(root_path)),
    }

    # Identify primary asset name for logging/progress display
    asset_obj = assets[0]
    asset_name = getattr(asset_obj, "key", None)
    asset_name_str = asset_name.path[-1] if asset_name else "asset"

    results: List[dg.ExecuteInProcessResult] = []

    if partition_keys is not None and len(partition_keys) > 0:
        try:
            from tqdm import tqdm
            iterator = tqdm(partition_keys, desc=f"Materializing {asset_name_str}", disable=not show_progress)
        except ImportError:
            iterator = partition_keys

        for pk in iterator:
            t0 = time.time()
            res = dg.materialize(
                assets,
                partition_key=pk,
                resources=effective_resources,
                run_config=run_config,
            )
            elapsed = max(time.time() - t0, 0.001)
            results.append(res)

            size_bytes = get_partition_storage_size(root_path, asset_name_str, pk)
            size_mb = size_bytes / (1024 * 1024)
            speed_mb_s = size_mb / elapsed

            ext = "parquet" if (root_path / "obs" / asset_name_str / f"{pk}.parquet").exists() else "zarr"
            progress_msg = f"{pk}.{ext}: {size_mb:.1f}MB [{format_duration(elapsed)}, {speed_mb_s:.2f}MB/s]"

            if hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(progress_msg)
            else:
                logger.info(progress_msg)

    else:
        t0 = time.time()
        res = dg.materialize(
            assets,
            resources=effective_resources,
            run_config=run_config,
        )
        elapsed = max(time.time() - t0, 0.001)
        results.append(res)
        logger.info(
            "Materialized asset(s) '%s' in %s",
            asset_name_str,
            format_duration(elapsed),
        )

    return results


__all__: List[str] = [
    # Pipeline definitions and materialization
    "defs",
    "materialize_extraction",
    # IO Managers
    "ParquetStationIOManager",
    "ZarrGriddedIOManager",
    # Software-Defined Assets
    "tahmo_station_observations",
    "era5_gridded_reanalysis",
    "imerg_satellite_precipitation",
    "cds_insitu_observations",
    # Partitions
    "tahmo_partitions",
    "era5_partitions",
    "imerg_partitions",
    "cds_insitu_partitions",
    # Helpers
    "decode_cds_insitu_dataframe",
    "write_station_metadata_parquets",
]
