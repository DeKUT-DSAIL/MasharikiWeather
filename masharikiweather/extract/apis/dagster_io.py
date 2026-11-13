"""
Dagster IO Managers for Tabular Station and Gridded Meteorological Datasets.

Defines production-grade Dagster IO Managers handling framework-agnostic persistence:
1. ParquetStationIOManager: Tabular station observations stored in partitioned Apache Parquet
   format with Snappy compression under '{root_dir}/obs/{asset_name}/{partition}.parquet'.
2. ZarrGriddedIOManager: Multidimensional gridded atmospheric/satellite datasets stored in
   consolidated Zarr format under '{root_dir}/gridded/{asset_name}/{partition}.zarr'.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dagster as dg
import pandas as pd
import xarray as xr

logger = logging.getLogger("MasharikiWeather.DagsterIO")


class ParquetStationIOManager(dg.ConfigurableIOManager):
    """
    Dagster IO Manager for persisting and loading station observations via Apache Parquet.

    Mathematical docstring:
    Input dataframe schema: (N_rows, C_columns)
    where columns typically include ['time', 'station_code', 'lat', 'lon', 'elevation', ...]
    Output disk structure: '{root_dir}/obs/{asset_name}/{partition}.parquet'
    """

    root_dir: str = "./mashariki_data"

    def _get_path(self, context: Union[dg.OutputContext, dg.InputContext]) -> Path:
        """
        Derive local file path for asset partition.

        Parameters
        ----------
        context : Union[dg.OutputContext, dg.InputContext]
            Dagster execution context.

        Returns
        -------
        Path
            Target Parquet file path on disk.
        """
        asset_name = context.asset_key.path[-1]
        partition = context.partition_key if context.has_partition_key else "all"
        return Path(self.root_dir) / "obs" / asset_name / f"{partition}.parquet"

    def handle_output(
        self,
        context: dg.OutputContext,
        obj: Union[pd.DataFrame, Path, str],
    ) -> None:
        """
        Persist tabular station data to Apache Parquet with Snappy compression.

        Parameters
        ----------
        context : dg.OutputContext
            Dagster output context.
        obj : Union[pd.DataFrame, Path, str]
            Station observations DataFrame or existing file path.
        """
        dest_path = self._get_path(context)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(obj, pd.DataFrame):
            # Write to parquet using PyArrow engine and Snappy compression
            obj.to_parquet(dest_path, engine="pyarrow", compression="snappy", index=True)
            row_count = len(obj)
            col_names = [str(c) for c in obj.columns]
        elif isinstance(obj, (str, Path)):
            src_path = Path(obj)
            if src_path != dest_path and src_path.exists():
                shutil.copy2(src_path, dest_path)
            try:
                meta_df = pd.read_parquet(dest_path)
                row_count = len(meta_df)
                col_names = [str(c) for c in meta_df.columns]
            except Exception:
                row_count = -1
                col_names = []
        else:
            raise TypeError(
                f"ParquetStationIOManager expected pd.DataFrame or Path/str, received {type(obj)}"
            )

        file_size_bytes = dest_path.stat().st_size if dest_path.exists() else 0
        file_size_mb = round(file_size_bytes / (1024 * 1024), 3)

        context.add_output_metadata({
            "path": str(dest_path),
            "row_count": row_count,
            "columns": col_names[:25],
            "num_columns": len(col_names),
            "size_bytes": file_size_bytes,
            "size_mb": file_size_mb,
        })
        logger.info(
            "Persisted station asset '%s' partition '%s' to %s (%d rows, %.2f MB)",
            context.asset_key.to_user_string(),
            context.partition_key if context.has_partition_key else "all",
            dest_path,
            row_count,
            file_size_mb,
        )

    def load_input(self, context: dg.InputContext) -> pd.DataFrame:
        """
        Load tabular station data from Apache Parquet.

        Parameters
        ----------
        context : dg.InputContext
            Dagster input context.

        Returns
        -------
        pd.DataFrame
            Loaded station observations DataFrame.
        """
        dest_path = self._get_path(context)
        if not dest_path.exists():
            raise FileNotFoundError(
                f"Parquet station file does not exist at expected location: {dest_path}"
            )
        return pd.read_parquet(dest_path)


class ZarrGriddedIOManager(dg.ConfigurableIOManager):
    """
    Dagster IO Manager for persisting and loading gridded spatial datasets via Zarr.

    Mathematical docstring:
    Input dataset dimensions: (time: T, [level: L], lat: Y, lon: X)
    Output disk structure: '{root_dir}/gridded/{asset_name}/{partition}.zarr'
    """

    root_dir: str = "./mashariki_data"

    def _get_path(self, context: Union[dg.OutputContext, dg.InputContext]) -> Path:
        """
        Derive local directory path for gridded asset partition.

        Parameters
        ----------
        context : Union[dg.OutputContext, dg.InputContext]
            Dagster execution context.

        Returns
        -------
        Path
            Target Zarr store directory path on disk.
        """
        asset_name = context.asset_key.path[-1]
        partition = context.partition_key if context.has_partition_key else "all"
        return Path(self.root_dir) / "gridded" / asset_name / f"{partition}.zarr"

    def handle_output(
        self,
        context: dg.OutputContext,
        obj: Union[xr.Dataset, Path, str],
    ) -> None:
        """
        Persist multidimensional gridded dataset to Zarr store with consolidated metadata.

        Parameters
        ----------
        context : dg.OutputContext
            Dagster output context.
        obj : Union[xr.Dataset, Path, str]
            Gridded dataset or existing Zarr directory path.
        """
        dest_path = self._get_path(context)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        dims: Dict[str, int] = {}
        vars_list: List[str] = []

        if isinstance(obj, xr.Dataset):
            # Clear individual variable encodings to prevent chunking or filter conflicts
            for v in obj.variables:
                obj[v].encoding.clear()

            obj.to_zarr(str(dest_path), mode="w", consolidated=True)
            dims = {str(d): int(s) for d, s in obj.sizes.items()}
            vars_list = [str(k) for k in obj.data_vars.keys()]
        elif isinstance(obj, (str, Path)):
            src_path = Path(obj)
            if src_path != dest_path and src_path.exists():
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path)
            if dest_path.exists():
                try:
                    with xr.open_zarr(str(dest_path), consolidated=True) as ds:
                        dims = {str(d): int(s) for d, s in ds.sizes.items()}
                        vars_list = [str(k) for k in ds.data_vars.keys()]
                except Exception as read_err:
                    logger.warning("Could not read Zarr metadata from %s: %s", dest_path, read_err)
        else:
            raise TypeError(
                f"ZarrGriddedIOManager expected xr.Dataset or Path/str, received {type(obj)}"
            )

        context.add_output_metadata({
            "path": str(dest_path),
            "dimensions": dims,
            "variables": vars_list[:30],
            "num_variables": len(vars_list),
        })
        logger.info(
            "Persisted gridded asset '%s' partition '%s' to %s (%s)",
            context.asset_key.to_user_string(),
            context.partition_key if context.has_partition_key else "all",
            dest_path,
            dims,
        )

    def load_input(self, context: dg.InputContext) -> xr.Dataset:
        """
        Load multidimensional gridded dataset from consolidated Zarr store.

        Parameters
        ----------
        context : dg.InputContext
            Dagster input context.

        Returns
        -------
        xr.Dataset
            Lazily loaded xarray Dataset.
        """
        dest_path = self._get_path(context)
        if not dest_path.exists():
            raise FileNotFoundError(
                f"Zarr store does not exist at expected location: {dest_path}"
            )
        return xr.open_zarr(str(dest_path), consolidated=True)
