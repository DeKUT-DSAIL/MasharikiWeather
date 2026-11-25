"""
Dagster Pipeline Definitions and Programmatic Materialization Runner.

Exposes unified Dagster definitions and batch extraction helpers:
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
    EXTRACTION_CONFIG_SCHEMA,
    cds_insitu_observations,
    cds_insitu_partitions,
    decode_cds_insitu_dataframe,
    era5_gridded_reanalysis,
    era5_partitions,
    ghcn_partitions,
    ghcn_station_observations,
    imerg_partitions,
    imerg_satellite_precipitation,
    tahmo_partitions,
    tahmo_station_observations,
    write_station_metadata_parquets,
)
from .dagster_io import ParquetStationIOManager, ZarrGriddedIOManager

logger = logging.getLogger("MasharikiWeather.DagsterPipeline")

DEFAULT_ROOT_DIR = "./mashariki-data"

SOURCE_ASSET_MAP: Dict[str, Any] = {
    "tahmo": tahmo_station_observations,
    "ghcn": ghcn_station_observations,
    "ghcnd": ghcn_station_observations,
    "ghcn_daily": ghcn_station_observations,
    "ghcn_hourly": ghcn_station_observations,
    "era5": era5_gridded_reanalysis,
    "imerg": imerg_satellite_precipitation,
    "cds_insitu": cds_insitu_observations,
    "cds": cds_insitu_observations,
}

# Standard definitions bundling all software-defined assets and IO managers
defs = dg.Definitions(
    assets=[
        tahmo_station_observations,
        ghcn_station_observations,
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
    assets: Optional[Sequence[Any]] = None,
    partition_keys: Optional[Sequence[str]] = None,
    root_dir: Union[str, Path] = DEFAULT_ROOT_DIR,
    resources: Optional[Dict[str, Any]] = None,
    run_config: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,
    *,
    sources: Optional[Sequence[str]] = None,
    years: Optional[Sequence[int]] = None,
    max_workers: Optional[int] = None,
    resume: bool = True,
    keep_cache: bool = False,
) -> List[dg.ExecuteInProcessResult]:
    """
    Programmatically materialize software-defined extraction assets with progress reporting.

    Supports both explicit Dagster ``assets`` / ``partition_keys`` and source-based
    invocation via ``sources`` / ``years`` (used by ``MasharikiWeatherDataset``).
    """
    resolved_assets: List[Any] = list(assets) if assets is not None else []
    if not resolved_assets and sources is not None:
        for src in sources:
            asset_obj = SOURCE_ASSET_MAP.get(str(src).lower().strip())
            if asset_obj is not None and asset_obj not in resolved_assets:
                resolved_assets.append(asset_obj)
    if not resolved_assets:
        raise ValueError("materialize_extraction requires either 'assets' or valid 'sources'.")

    resolved_partitions: Optional[List[str]] = list(partition_keys) if partition_keys is not None else None
    if resolved_partitions is None and years:
        resolved_partitions = [f"{int(y):04d}-{m:02d}-01" for y in sorted(set(years)) for m in range(1, 13)]

    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)

    effective_resources = resources or {
        "parquet_io": ParquetStationIOManager(root_dir=str(root_path)),
        "zarr_io": ZarrGriddedIOManager(root_dir=str(root_path)),
    }

    effective_run_config: Optional[Dict[str, Any]] = dict(run_config) if run_config is not None else None
    if max_workers is not None or not resume or keep_cache:
        effective_run_config = effective_run_config or {}
        ops_cfg = dict(effective_run_config.get("ops", {}))
        for asset_item in resolved_assets:
            key = getattr(asset_item, "key", None)
            op_name = key.path[-1] if key else None
            if op_name:
                existing_op = dict(ops_cfg.get(op_name, {}))
                existing_cfg = dict(existing_op.get("config", {}))
                if max_workers is not None and "max_workers" not in existing_cfg:
                    existing_cfg["max_workers"] = int(max_workers)
                existing_cfg.setdefault("resume", resume)
                existing_cfg.setdefault("keep_cache", keep_cache)
                existing_cfg.setdefault("show_progress", False)
                existing_op["config"] = existing_cfg
                ops_cfg[op_name] = existing_op
        effective_run_config["ops"] = ops_cfg

    # Identify primary asset name for logging/progress display
    asset_obj = resolved_assets[0]
    asset_name = getattr(asset_obj, "key", None)
    asset_name_str = asset_name.path[-1] if asset_name else "asset"

    results: List[dg.ExecuteInProcessResult] = []

    if resolved_partitions is not None and len(resolved_partitions) > 0:
        try:
            from tqdm import tqdm
            iterator = tqdm(resolved_partitions, desc=f"Materializing {asset_name_str}", disable=not show_progress)
        except ImportError:
            iterator = resolved_partitions

        for pk in iterator:
            t0 = time.time()
            res = dg.materialize(
                resolved_assets,
                partition_key=pk,
                resources=effective_resources,
                run_config=effective_run_config,
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
            resolved_assets,
            resources=effective_resources,
            run_config=effective_run_config,
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
    "EXTRACTION_CONFIG_SCHEMA",
    # IO Managers
    "ParquetStationIOManager",
    "ZarrGriddedIOManager",
    # Software-Defined Assets
    "tahmo_station_observations",
    "ghcn_station_observations",
    "era5_gridded_reanalysis",
    "imerg_satellite_precipitation",
    "cds_insitu_observations",
    # Partitions
    "tahmo_partitions",
    "ghcn_partitions",
    "era5_partitions",
    "imerg_partitions",
    "cds_insitu_partitions",
    # Helpers
    "decode_cds_insitu_dataframe",
    "write_station_metadata_parquets",
]
