"""
NASA IMERG GPM Satellite Precipitation Extraction Engine.

Downloads and processes NASA Global Precipitation Measurement (GPM) Integrated
Multi-satellitE Retrievals for GPM (IMERG) half-hourly and daily products.
Features automated, non-interactive Earthdata authentication, spatial granule
subsetting, bounding box cropping, and local Zarr group compilation.
"""

from __future__ import annotations

import calendar
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import (
    AFRICA_BBOX,
    BaseExtractor,
    SpatialBoundingBox,
    TemporalRange,
)
from .credentials import CredentialsError, CredentialsManager, parse_date, parse_date_range

logger = logging.getLogger("MasharikiWeather.IMERG")

IMERG_EPOCH = datetime(1980, 1, 6)
SKIP_VARS = {"nv", "lonv", "latv", "time_bnds", "lon_bnds", "lat_bnds", "lat", "lon", "time"}
DEFAULT_ZARR_CHUNKS = {"time": 48, "lat": 100, "lon": 100}

# NASA GPM IMERG Operational Transition (V07 to V08):
# The IMERG V07 Final Run (GPM_3IMERGHH) operational data processing terminated
# around October 2025. For dates beginning November 1, 2025 onwards, precipitation
# data is drawn directly from the IMERG Late Run stream (GPM_3IMERGHHL).
IMERG_TRANSITION_DATE = datetime(2025, 11, 1)
IMERG_FINAL_RUN_NAME = "GPM_3IMERGHH"
IMERG_LATE_RUN_NAME = "GPM_3IMERGHHL"

# Canonical Descriptive Variable Harmonization Registry for IMERG
IMERG_RAW_TO_CANONICAL: Dict[str, str] = {
    "precipitationCal": "total_precipitation",
    "precipitationUncal": "uncalibrated_precipitation",
    "randomError": "precipitation_error",
    "HQprecipitation": "high_quality_precipitation",
    "probabilityLiquidPrecipitation": "probability_liquid_precipitation",
}

IMERG_CANONICAL_TO_RAW: Dict[str, str] = {
    v: k for k, v in IMERG_RAW_TO_CANONICAL.items()
}


def canonicalize_imerg_variable(var: str) -> str:
    """Map raw IMERG variable name to canonical descriptive name."""
    return IMERG_RAW_TO_CANONICAL.get(var, var)


def imerg_canonical_to_raw(var: str) -> str:
    """Map canonical descriptive name to raw IMERG HDF5 variable name."""
    return IMERG_CANONICAL_TO_RAW.get(var, var)


def harmonize_imerg_dataset(ds: xr.Dataset, to_canonical: bool = True) -> xr.Dataset:
    """Harmonize IMERG dataset variables between raw and canonical descriptive names."""
    mapping = IMERG_RAW_TO_CANONICAL if to_canonical else IMERG_CANONICAL_TO_RAW
    rename_map = {k: v for k, v in mapping.items() if k in ds.data_vars and k != v}
    if rename_map:
        return ds.rename(rename_map)
    return ds


@dataclass
class IMERGResult:
    """
    Container capturing the status and local artifacts of an IMERG extraction.
    """

    start_date: datetime
    end_date: datetime
    bbox: SpatialBoundingBox
    dataset_name: str
    raw_dir: Path
    zarr_dir: Optional[Path] = None
    files: List[str] = field(default_factory=list)
    failed_months: List[str] = field(default_factory=list)
    zarr_stores: List[str] = field(default_factory=list)
    zarr_errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True if extraction succeeded without failed months or zarr compilation errors."""
        return not (self.failed_months or self.zarr_errors)

    def open(self, variables: Optional[List[str]] = None) -> xr.Dataset:
        """
        Lazily open all converted monthly Zarr stores trimmed to the requested period.
        """
        if self.zarr_dir and self.zarr_stores:
            return open_imerg_zarr(self.zarr_dir, self.start_date, self.end_date, variables)
        h5_files = sorted(str(f) for f in self.files if str(f).upper().endswith((".HDF5", ".H5")))
        if not h5_files:
            raise RuntimeError(
                "No IMERG granules were downloaded. Check Earthdata credentials, the requested period and "
                f"failed months: {self.failed_months}."
            )
        dsets = [_read_hdf5_granule(p, bbox=self.bbox, variables=variables) for p in h5_files]
        combined = xr.concat(dsets, dim="time", coords="minimal", compat="override").sortby("time")
        _, uniq_idx = np.unique(combined.time.values, return_index=True)
        return combined.isel(time=np.sort(uniq_idx))

    # Alias for method name consistency across extractors
    open_zarr = open


def _login_earthdata_non_interactive(
    username: Optional[str] = None,
    password: Optional[str] = None,
    token: Optional[str] = None,
    env_file: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Authenticate with NASA Earthdata using credentials discovered via BYOK CredentialsManager.
    Guarantees non-interactive operation without prompting for terminal input.
    """
    try:
        import earthaccess
    except ImportError as exc:
        raise ImportError(
            "earthaccess is required for IMERG extraction. Install via 'pip install earthaccess'."
        ) from exc

    creds = CredentialsManager.get_earthdata_credentials(
        username=username, password=password, token=token, env_file=env_file
    )
    if creds.get("username"):
        os.environ["EARTHDATA_USERNAME"] = str(creds["username"])
    if creds.get("password"):
        os.environ["EARTHDATA_PASSWORD"] = str(creds["password"])
    if creds.get("token"):
        os.environ["EARTHDATA_TOKEN"] = str(creds["token"])

    auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        raise CredentialsError(
            "NASA Earthdata",
            "Authentication failed via environment and netrc.",
            [
                "Verify EARTHDATA_USERNAME and EARTHDATA_PASSWORD (or EARTHDATA_TOKEN) in your environment or .env file.",
                "Ensure your account is active at https://urs.earthdata.nasa.gov/",
            ],
        )
    return auth


def _read_hdf5_granule(
    path: Union[str, Path],
    bbox: Optional[SpatialBoundingBox] = None,
    variables: Optional[Sequence[str]] = None,
    harmonize: bool = False,
) -> xr.Dataset:
    """
    Read a single IMERG HDF5 granule with immediate spatial truncation.

    Truncates global 1800x3600 grid to spatial bounding box immediately upon HDF5 decode
    to prevent disk and memory bloat.

    Mathematical docstring:
    Output tensor shape: (T=1, Lat, Lon)
    where:
      Lat in [bbox.lat_min, bbox.lat_max]
      Lon in [bbox.lon_min, bbox.lon_max]
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to read IMERG granules. Install via 'pip install h5py'.") from exc

    p_str = str(path)
    with h5py.File(p_str, "r") as f:
        grid = f["Grid"]
        lat = grid["lat"][:]
        lon = grid["lon"][:]
        seconds_val = int(grid["time"][:][0])
        dt = IMERG_EPOCH + timedelta(seconds=seconds_val)

        # Map requested canonical names back to raw HDF5 dataset names
        if variables is not None:
            raw_target_vars = [imerg_canonical_to_raw(v) for v in variables]
            target_vars = [k for k in raw_target_vars if k in grid]
        else:
            target_vars = [k for k in grid.keys() if k not in SKIP_VARS]

        # Immediate spatial bounding box subsetting upon decode
        if bbox is not None:
            lat_mask = (lat >= bbox.lat_min) & (lat <= bbox.lat_max)
            lon_mask = (lon >= bbox.lon_min) & (lon <= bbox.lon_max)
            lat_indices = np.where(lat_mask)[0]
            lon_indices = np.where(lon_mask)[0]
            if lat_indices.size > 0 and lon_indices.size > 0:
                lat_start, lat_stop = int(lat_indices[0]), int(lat_indices[-1]) + 1
                lon_start, lon_stop = int(lon_indices[0]), int(lon_indices[-1]) + 1
                lat_sub = lat[lat_start:lat_stop]
                lon_sub = lon[lon_start:lon_stop]
            else:
                lat_start, lat_stop = 0, len(lat)
                lon_start, lon_stop = 0, len(lon)
                lat_sub, lon_sub = lat, lon
        else:
            lat_start, lat_stop = 0, len(lat)
            lon_start, lon_stop = 0, len(lon)
            lat_sub, lon_sub = lat, lon

        data_vars: Dict[str, Tuple[Tuple[str, ...], np.ndarray]] = {}
        for k in target_vars:
            h5_ds = grid[k]
            if not isinstance(h5_ds, h5py.Dataset):
                continue

            arr_shape = h5_ds.shape
            # Slice along bounding box directly from HDF5 if dimensions match lon/lat
            if len(arr_shape) == 3 and arr_shape[1] == len(lon) and arr_shape[2] == len(lat):
                arr = h5_ds[:, lon_start:lon_stop, lat_start:lat_stop]
                # Transpose to CF standard (time, lat, lon)
                arr = np.transpose(arr, (0, 2, 1))
                dims = ("time", "lat", "lon")
            elif len(arr_shape) == 3 and arr_shape[1] == len(lat) and arr_shape[2] == len(lon):
                arr = h5_ds[:, lat_start:lat_stop, lon_start:lon_stop]
                dims = ("time", "lat", "lon")
            elif len(arr_shape) == 2 and arr_shape[0] == len(lon) and arr_shape[1] == len(lat):
                arr = h5_ds[lon_start:lon_stop, lat_start:lat_stop]
                arr = np.transpose(arr, (1, 0))
                dims = ("lat", "lon")
            elif len(arr_shape) == 2 and arr_shape[0] == len(lat) and arr_shape[1] == len(lon):
                arr = h5_ds[lat_start:lat_stop, lon_start:lon_stop]
                dims = ("lat", "lon")
            else:
                # Fallback for 1D or invariant datasets
                arr = h5_ds[:]
                dims_list: List[str] = []
                for i, s in enumerate(arr.shape):
                    if s == len(lat_sub):
                        dims_list.append("lat")
                    elif s == len(lon_sub):
                        dims_list.append("lon")
                    elif s == 1:
                        dims_list.append("time")
                    else:
                        dims_list.append(f"dim_{i}")
                dims = tuple(dims_list)

            out_var_name = canonicalize_imerg_variable(k) if harmonize else k
            data_vars[out_var_name] = (dims, arr)

    ds = xr.Dataset(
        data_vars,
        coords={"lat": ("lat", lat_sub), "lon": ("lon", lon_sub), "time": [np.datetime64(dt)]},
    )

    return ds.copy(deep=True)


def open_imerg_zarr(
    zarr_dir: Union[str, Path],
    start_date: Union[str, datetime, date, pd.Timestamp],
    end_date: Union[str, datetime, date, pd.Timestamp],
    variables: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Lazily open and concatenate monthly IMERG Zarr stores.
    Accurately handles month-start/end, year-start/end, and leap years (Feb 28 vs Feb 29).
    """
    s_dt, e_dt = parse_date_range(start_date, end_date)

    root = Path(zarr_dir)
    stores = sorted(root.glob("*/IMERG_*_*.zarr"))
    valid_stores: List[Path] = []

    for st in stores:
        try:
            parts = st.stem.replace("IMERG_", "").split("_")
            y, m = int(parts[0]), int(parts[1])
            num_days = calendar.monthrange(y, m)[1]
            m_start = datetime(y, m, 1, 0, 0, 0)
            m_end = datetime(y, m, num_days, 23, 59, 59)
            if not (m_start <= e_dt and m_end >= s_dt):
                continue
            valid_stores.append(st)
        except Exception:
            continue

    if not valid_stores:
        raise FileNotFoundError(f"No IMERG Zarr stores matching period {s_dt}..{e_dt} in {root}")

    dsets = [xr.open_zarr(p) for p in valid_stores]
    if variables:
        dsets = [d[[v for v in variables if v in d.data_vars]] for d in dsets]

    combined = xr.concat(dsets, dim="time", coords="minimal", compat="override")
    return combined.sel(time=slice(s_dt, e_dt))


class IMERGExtractor(BaseExtractor):
    """
    NASA IMERG Extraction and Zarr Compilation Pipeline.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "./data/imerg",
        dataset_name: str = "GPM_3IMERGHH",
        version: str = "07",
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        env_file: Optional[Union[str, Path]] = None,
        max_workers: int = 4,
    ) -> None:
        """Initialize NASA IMERG extraction engine and output directory structure."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.version = version
        self.username = username
        self.password = password
        self.token = token
        self.env_file = env_file
        self.max_workers = max_workers

    def extract(
        self,
        start_date: Union[str, datetime, date, pd.Timestamp],
        end_date: Union[str, datetime, date, pd.Timestamp],
        bbox: Optional[SpatialBoundingBox] = None,
        variables: Optional[Sequence[str]] = None,
        convert_to_zarr: bool = True,
        zarr_chunks: Optional[Dict[str, int]] = None,
        harmonize: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        env_file: Optional[Union[str, Path]] = None,
    ) -> IMERGResult:
        """
        Execute end-to-end IMERG search, non-interactive download, and Zarr compilation.

        Transitions from Final Run (GPM_3IMERGHH) to Late Run (GPM_3IMERGHHL)
        for queries intersecting or following the November 1, 2025 boundary.

        Mathematical docstring:
        Precipitation Rate Variable: 'precipitationCal' / 'total_precipitation' (mm/hr).
        Output Zarr array shape: (T_time, Lat, Lon).
        """
        import earthaccess

        box = bbox if bbox is not None else AFRICA_BBOX
        t_range = self.validate_temporal_range(start_date, end_date)
        s_dt = t_range.start.to_pydatetime()
        e_dt = t_range.end.to_pydatetime()

        _login_earthdata_non_interactive(
            username=username or self.username,
            password=password or self.password,
            token=token or self.token,
            env_file=env_file or self.env_file,
        )

        raw_root = self.output_dir / "raw"
        zarr_root = self.output_dir / "zarr"
        raw_root.mkdir(parents=True, exist_ok=True)

        result = IMERGResult(
            start_date=s_dt,
            end_date=e_dt,
            bbox=box,
            dataset_name=self.dataset_name,
            raw_dir=raw_root,
            zarr_dir=zarr_root if convert_to_zarr else None,
        )

        monthly_windows = t_range.split_by_month()
        logger.info(
            "Searching IMERG granules for %s (%d monthly window(s))...",
            self.dataset_name,
            len(monthly_windows),
        )

        bbox_tuple = box.to_bbox_tuple()

        for win_start, win_end in monthly_windows:
            w_start_str = win_start.strftime("%Y-%m-%d %H:%M:%S")
            w_end_str = win_end.strftime("%Y-%m-%d %H:%M:%S")

            # Operational transition handling:
            # IMERG V07 Final Run (GPM_3IMERGHH) ended around Oct 2025.
            # Merge Final Run with Late Run (GPM_3IMERGHHL) for timestamps >= 2025-11-01.
            if win_start >= IMERG_TRANSITION_DATE:
                active_dataset = IMERG_LATE_RUN_NAME
                logger.info(
                    "Requested date %s exceeds IMERG Final Run cutoff; appending Late Run (GPM_3IMERGHHL) records.",
                    win_start.strftime("%Y-%m-%d"),
                )
            else:
                active_dataset = self.dataset_name

            month_dir = raw_root / active_dataset / str(win_start.year) / f"{win_start.month:02d}"
            month_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Searching granules (%s): %s to %s", active_dataset, w_start_str, w_end_str)
            try:
                granules = earthaccess.search_data(
                    short_name=active_dataset,
                    version=self.version,
                    temporal=(w_start_str, w_end_str),
                    bounding_box=bbox_tuple,
                )
                if not granules:
                    logger.warning("No granules discovered for %s-%02d (%s)", win_start.year, win_start.month, active_dataset)
                    result.failed_months.append(f"{win_start.year}-{win_start.month:02d}")
                    continue

                logger.info("Found %d granules. Downloading to %s...", len(granules), month_dir)
                downloaded = earthaccess.download(granules, str(month_dir))
                result.files.extend([str(f) for f in downloaded])

            except Exception as e:
                logger.error("Download error for %d-%02d (%s): %s", win_start.year, win_start.month, active_dataset, e)
                result.failed_months.append(f"{win_start.year}-{win_start.month:02d}")

        if convert_to_zarr:
            zarr_root.mkdir(parents=True, exist_ok=True)
            chunks = zarr_chunks or DEFAULT_ZARR_CHUNKS

            for win_start, _ in monthly_windows:
                y, m = win_start.year, win_start.month
                active_dataset = IMERG_LATE_RUN_NAME if win_start >= IMERG_TRANSITION_DATE else self.dataset_name
                run_type = "Late" if win_start >= IMERG_TRANSITION_DATE else "Final"

                m_dir = raw_root / active_dataset / str(y) / f"{m:02d}"
                if not m_dir.exists():
                    # Fallback to legacy single dataset directory
                    m_dir = self.output_dir / self.dataset_name / str(y) / f"{m:02d}"

                h5_files = sorted(str(p) for p in m_dir.glob("*.HDF5")) if m_dir.exists() else []
                if not h5_files:
                    continue

                store_path = zarr_root / str(y) / f"IMERG_{y}_{m:02d}.zarr"
                store_path.parent.mkdir(parents=True, exist_ok=True)

                logger.info("Compiling %d HDF5 granules into Zarr store: %s", len(h5_files), store_path)
                try:
                    dsets: List[xr.Dataset] = []
                    for h5_p in h5_files:
                        try:
                            ds_one = _read_hdf5_granule(h5_p, bbox=box, variables=variables, harmonize=harmonize)
                            dsets.append(ds_one)
                        except Exception as rd_err:
                            logger.warning("Skipping unreadable file %s: %s", h5_p, rd_err)

                    if not dsets:
                        continue

                    combined = xr.concat(dsets, dim="time", coords="minimal", compat="override")
                    # Deduplicate and sort by time
                    _, uniq_idx = np.unique(combined.time.values, return_index=True)
                    if len(uniq_idx) < combined.sizes["time"]:
                        combined = combined.isel(time=np.sort(uniq_idx))
                    combined = combined.sortby("time")

                    # Attach transition provenance metadata
                    combined.attrs["source_run"] = run_type
                    combined.attrs["transition_boundary"] = "2025-11-01"

                    chunked = combined.chunk(chunks)
                    for var in chunked.variables:
                        chunked[var].encoding.clear()

                    chunked.to_zarr(str(store_path), mode="w", consolidated=True)
                    result.zarr_stores.append(str(store_path))
                    logger.info("Created Zarr store: %s", store_path)

                except Exception as z_err:
                    err_msg = f"{y}_{m:02d}: {z_err}"
                    logger.error("Zarr compilation failure: %s", err_msg)
                    result.zarr_errors.append(err_msg)

        return result
