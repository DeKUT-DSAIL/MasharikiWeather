"""
imerg_wrapper.py
================
Self-contained IMERG toolkit: download from NASA Earthdata, convert to monthly
Zarr, and read it back. It replaces `download_imerg.py` and `access.py`, so
this is the only file you need.

    from imerg_wrapper import get_imerg

    result = get_imerg(
        start_date="2025-01-01",
        end_date="2025-01-03",
        bbox=(36.0, -1.5, 38.0, 0.5),      # west, south, east, north
        output_dir="/content/IMERG",
        dataset_name="GPM_3IMERGHH",
        convert_to_zarr=True,
    )
    ds = result.open()                      # lazy xarray.Dataset

Layout on disk
--------------
    <output_dir>/<dataset_name>/<YYYY>/<MM>/*.HDF5     raw granules
    <output_dir>/zarr/<YYYY>/IMERG_<YYYY>_<MM>.zarr    converted (if requested)

Also exported: `convert_to_zarr_only` (convert files you already have) and
`open_imerg_zarr` (read an existing Zarr folder).

Dependencies: earthaccess, h5py, numpy, pandas, xarray, dask, tqdm and, for Zarr
conversion, zarr>=3 (needs Python >= 3.11)  ->  pip install "zarr>=3"

When you call `get_imerg` it first shows how to create a (free) NASA Earthdata
account and asks whether you have one. Answer "yes" to log in and download; answer
"no" and it walks you through registration first. In non-interactive jobs (SLURM,
cron) pass has_account=True to skip the question.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("imerg_wrapper")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

DateLike = Union[str, datetime, pd.Timestamp]
BBox = Tuple[float, float, float, float]

_IMERG_EPOCH = datetime(1980, 1, 6)  # IMERG 'time' is seconds since this date
_SKIP_VARS = {"nv", "lonv", "latv", "time_bnds", "lon_bnds", "lat_bnds", "lat", "lon", "time"}
MIN_ZARR_MAJOR = 3  # Zarr conversion requires zarr >= 3 (which needs Python >= 3.11)
DEFAULT_ZARR_CHUNKS = {"time": 96, "lat": 200, "lon": 200}


# =========================================================================== #
# Result object
# =========================================================================== #
@dataclass
class IMERGResult:
    """What `get_imerg` returns."""

    start: datetime
    end: datetime                        # inclusive, end of the last requested day
    bbox: BBox
    dataset_name: str
    raw_dir: Path                        # <output_dir>/<dataset_name>
    zarr_dir: Optional[Path] = None
    files: List[str] = field(default_factory=list)          # raw files downloaded
    failed_months: List[str] = field(default_factory=list)  # "YYYY-MM" that failed
    empty_months: List[str] = field(default_factory=list)   # "YYYY-MM" with no granules
    zarr_stores: List[str] = field(default_factory=list)
    zarr_errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.failed_months or self.zarr_errors)

    def open(self, variables: Optional[List[str]] = None):
        """Lazily open the Zarr data trimmed to the requested period (end date inclusive)."""
        if not self.zarr_dir or not self.zarr_stores:
            raise RuntimeError("No Zarr stores available. Call get_imerg(..., convert_to_zarr=True).")
        return open_imerg_zarr(self.zarr_dir, self.start, self.end, variables)

    def __repr__(self) -> str:
        return (
            f"IMERGResult({self.dataset_name} {self.start:%Y-%m-%d}..{self.end:%Y-%m-%d}, "
            f"files={len(self.files)}, zarr_stores={len(self.zarr_stores)}, "
            f"failed_months={self.failed_months}, zarr_errors={len(self.zarr_errors)})"
        )


# =========================================================================== #
# Small helpers
# =========================================================================== #
def _to_datetime(d: DateLike, name: str) -> datetime:
    try:
        return pd.Timestamp(d).to_pydatetime().replace(tzinfo=None)
    except Exception as e:
        raise ValueError(f"Invalid {name}: {d!r}. Use YYYY-MM-DD.") from e


def _validate_bbox(bbox: Sequence[float]) -> BBox:
    if len(bbox) != 4:
        raise ValueError("bbox must be (west, south, east, north).")
    w, s, e, n = map(float, bbox)
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError(
            "Invalid bbox. Expected (west, south, east, north) with west<east and "
            f"south<north, got {tuple(bbox)}."
        )
    return w, s, e, n


def _day_range(start: DateLike, end: DateLike) -> Tuple[datetime, datetime]:
    s = _to_datetime(start, "start_date").replace(hour=0, minute=0, second=0, microsecond=0)
    e = _to_datetime(end, "end_date").replace(hour=23, minute=59, second=59, microsecond=0)
    if s > e:
        raise ValueError("start_date must be on or before end_date.")
    return s, e


def _month_windows(start: datetime, end: datetime) -> List[Tuple[datetime, datetime]]:
    """Split [start, end] into per-month windows, clipped to the request (partial months kept)."""
    out, cur = [], start
    while cur <= end:
        m_end = (pd.Timestamp(cur) + pd.offsets.MonthEnd(0)).to_pydatetime()
        m_end = m_end.replace(hour=23, minute=59, second=59)
        out.append((cur, min(m_end, end)))
        cur = (pd.Timestamp(cur.year, cur.month, 1) + pd.offsets.MonthBegin(1)).to_pydatetime()
    return out


# =========================================================================== #
# 1. Earthdata account + download (earthaccess)
# =========================================================================== #
_ACCOUNT_GUIDE = """
==============================================================================
 NASA Earthdata account required
==============================================================================
 IMERG data is free, but NASA needs you to be logged in to download it.

 If you do NOT have an account yet:
   1. Open   https://urs.earthdata.nasa.gov/users/new
   2. Fill in the form (username, password, email) and submit.
   3. Click the verification link NASA sends to your email.
   4. Log in at https://urs.earthdata.nasa.gov  ->  "Applications"
      ->  "Authorized Apps"  ->  "Approve More Applications", and approve
      "NASA GESDISC DATA ARCHIVE". IMERG lives there, and downloads fail
      (HTTP 401/403) until you do this.

 If you already have an account, you'll be asked for your username and
 password next. (Tip: set EARTHDATA_USERNAME and EARTHDATA_PASSWORD, or create
 ~/.netrc, to skip the prompt next time.)
==============================================================================
"""


def _ask_yes_no(question: str) -> bool:
    while True:
        try:
            ans = input(f"{question} [yes/no]: ").strip().lower()
        except EOFError:
            raise RuntimeError(
                "No interactive input available to ask about your Earthdata account. "
                "Create one at https://urs.earthdata.nasa.gov/users/new, then call "
                "get_imerg(..., has_account=True) (e.g. in SLURM or cron jobs)."
            ) from None
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer yes or no.")


def _ensure_account(has_account: Optional[bool]) -> None:
    """Show the account guide and keep asking until the user confirms an account (or quits)."""
    if has_account is True:
        return
    print(_ACCOUNT_GUIDE)
    if has_account is False:
        raise RuntimeError("Create an Earthdata account first (see the guide above), "
                           "then call get_imerg(..., has_account=True).")
    if _ask_yes_no("Do you have a NASA Earthdata account?"):
        return
    while True:
        print("\nNo problem - create one now: https://urs.earthdata.nasa.gov/users/new")
        print("(Remember to verify your email and approve 'NASA GESDISC DATA ARCHIVE'.)")
        if _ask_yes_no("Have you created your account now?"):
            return
        if not _ask_yes_no("Do you want to keep going and create it?"):
            raise RuntimeError("Earthdata account is required to download IMERG. "
                               "Create one at https://urs.earthdata.nasa.gov/users/new and re-run.")


def _login(attempts: int = 3) -> None:
    import earthaccess

    for i in range(1, attempts + 1):
        try:
            auth = earthaccess.login()  # env vars -> ~/.netrc -> interactive prompt
            if getattr(auth, "authenticated", True):
                logger.info("Logged in to NASA Earthdata.")
                return
        except Exception as e:
            logger.warning("Login error: %s", e)
        print(f"Login failed (attempt {i}/{attempts}). Check your username/password "
              "(use your Earthdata username, not your email).")
    raise RuntimeError(
        "Could not log in to Earthdata. Reset your password or create an account at "
        "https://urs.earthdata.nasa.gov, then try again."
    )


def _check_zarr() -> None:
    """Zarr conversion needs zarr>=3; fail early, before any download."""
    import sys

    fix = (
        'Fix:  pip install "zarr>=3"      (keep the quotes; in PowerShell/cmd an unquoted '
        '">" is a redirect)'
    )
    if sys.version_info < (3, 11):
        v = ".".join(map(str, sys.version_info[:3]))
        raise ImportError(
            f"convert_to_zarr=True needs zarr>=3, but zarr 3 requires Python >= 3.11 and this "
            f"environment runs Python {v}, so pip cannot install it here.\n"
            "Fix:  create an environment with a newer Python, e.g.\n"
            "        conda create -n imerg python=3.12 -y && conda activate imerg\n"
            '        pip install earthaccess h5py xarray dask pandas tqdm "zarr>=3"'
        )
    try:
        import zarr
    except ImportError:
        raise ImportError(f"convert_to_zarr=True needs zarr>=3, which is not installed.\n{fix}") from None
    if int(zarr.__version__.split(".")[0]) < MIN_ZARR_MAJOR:
        raise ImportError(
            f"convert_to_zarr=True needs zarr>=3, found {zarr.__version__}.\n{fix}\n"
            "(then restart the Python session/kernel)"
        )


def _default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _download_month(dataset_name, version, bbox, win, out_dir: Path, threads, retries):
    """Search + download one month window. Returns (files, failed, empty)."""
    import earthaccess

    label = f"{win[0]:%Y-%m}"
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        short_name=dataset_name,
        bounding_box=bbox,
        temporal=win,          # datetimes, so the whole last day is included
        cloud_hosted=False,
        count=-1,
    )
    if version:
        kwargs["version"] = version

    granules = None
    for attempt in range(1, retries + 1):
        try:
            granules = list(earthaccess.search_data(**kwargs))
            break
        except Exception as e:
            logger.warning("Search failed for %s (attempt %d/%d): %s", label, attempt, retries, e)
            time.sleep(2 * attempt)
    if granules is None:
        return [], True, False
    if not granules:
        logger.warning("No granules found for %s", label)
        return [], False, True

    logger.info("%s: %d granules -> %s", label, len(granules), out_dir)

    files: List[str] = []
    for attempt in range(1, retries + 1):
        try:
            got = earthaccess.download(granules, local_path=str(out_dir), threads=threads)
            files = [str(f) for f in got if isinstance(f, (str, Path))]
            if len(files) >= len(granules):
                return files, False, False
            logger.warning("%s: got %d/%d files (attempt %d/%d)",
                           label, len(files), len(granules), attempt, retries)
        except Exception as e:
            logger.warning("Download failed for %s (attempt %d/%d): %s", label, attempt, retries, e)
        time.sleep(2 * attempt)
    return files, True, False


# =========================================================================== #
# 2. HDF5 -> Zarr
# =========================================================================== #
def _read_one(path: str, variables, bbox, attrs: Optional[dict]):
    """Read one IMERG HDF5 file into a (cropped, copied) xarray Dataset with a length-1 time axis."""
    import h5py
    import xarray as xr

    with h5py.File(path, "r") as f:
        grid = f["Grid"]
        lat, lon = grid["lat"][:], grid["lon"][:]
        t = _IMERG_EPOCH + timedelta(seconds=int(grid["time"][:][0]))

        keys = (
            [k for k in variables if k in grid]
            if variables is not None
            else [k for k in grid.keys() if k not in _SKIP_VARS]
        )
        data_vars = {}
        for k in keys:
            if not isinstance(grid[k], h5py.Dataset):
                continue
            arr = grid[k][:]
            dims = []
            for i, size in enumerate(arr.shape):
                if size == len(lat):
                    dims.append("lat")
                elif size == len(lon):
                    dims.append("lon")
                elif size == 1:
                    dims.append("time")
                else:
                    dims.append(f"dim_{i}")
            data_vars[k] = (tuple(dims), arr)

    ds = xr.Dataset(
        data_vars,
        coords={"lat": ("lat", lat), "lon": ("lon", lon), "time": [np.datetime64(t)]},
    )
    if bbox is not None:
        ds = ds.sel(lat=slice(bbox[1], bbox[3]), lon=slice(bbox[0], bbox[2]))
    if attrs:
        for var, var_attr in attrs.items():
            if var in ds.data_vars:
                ds[var].attrs.update(var_attr)
            for k, v in var_attr.items():
                ds.attrs[f"{var}_{k}"] = v
    # .sel() returns views that pin the full global array in memory; copy to release it.
    return ds.copy(deep=True)


def _read_chunk(files: List[str], variables, bbox, attrs):
    """Worker: read a list of files. Returns (datasets, errors)."""
    out, errs = [], []
    for p in files:
        try:
            out.append(_read_one(p, variables, bbox, attrs))
        except Exception as e:
            errs.append(f"{Path(p).name}: {e}")
    return out, errs


def _write_zarr(ds, out_path: str, chunks: Dict[str, int]) -> None:
    """Write a dataset to Zarr (zarr 3.x format, Blosc/LZ4 compression)."""
    from zarr.codecs import BloscCodec

    ck = (chunks["time"], chunks["lat"], chunks["lon"])
    comp = (BloscCodec(cname="lz4", clevel=3, shuffle="shuffle"),)
    encoding = {v: {"chunks": ck, "compressors": comp} for v in ds.data_vars}
    ds.chunk(chunks).to_zarr(out_path, mode="w", encoding=encoding, consolidated=True)


def _sort_dedupe_time(ds):
    _, idx = np.unique(ds.time.values, return_index=True)
    if len(idx) < ds.sizes["time"]:
        ds = ds.isel(time=np.sort(idx))
    return ds.sortby("time")


class _ZarrConverter:
    """Converts <raw_root>/<YYYY>/<MM>/*.HDF5 into <zarr_root>/<YYYY>/IMERG_<YYYY>_<MM>.zarr."""

    def __init__(self, raw_root: Path, zarr_root: Path, bbox: Optional[BBox],
                 variables: Optional[List[str]], attrs_path: Optional[str],
                 max_workers: int, batch_size: int = 48,
                 chunks: Optional[Dict[str, int]] = None):
        self.raw_root, self.zarr_root = Path(raw_root), Path(zarr_root)
        self.bbox, self.variables = bbox, variables
        self.max_workers = max(1, max_workers)
        self.batch_size = batch_size
        self.chunks = chunks or DEFAULT_ZARR_CHUNKS
        self.attrs = None
        if attrs_path and Path(attrs_path).exists():
            with open(attrs_path) as f:
                self.attrs = json.load(f)
        self.zarr_root.mkdir(parents=True, exist_ok=True)

    # -- paths / state ------------------------------------------------------
    def store_path(self, y: int, m: int) -> Path:
        return self.zarr_root / str(y) / f"IMERG_{y}_{m:02d}.zarr"

    def raw_files(self, y: int, m: int) -> List[str]:
        d = self.raw_root / str(y) / f"{m:02d}"
        return sorted(str(p) for p in d.glob("*.HDF5")) if d.exists() else []

    @staticmethod
    def zarr_steps(store: Path) -> Optional[int]:
        try:
            import xarray as xr

            with xr.open_zarr(store) as ds:
                return int(ds.sizes["time"])
        except Exception:
            return None

    # -- core ---------------------------------------------------------------
    def _read_all(self, files: List[str]):
        if len(files) <= self.batch_size or self.max_workers == 1:
            return _read_chunk(files, self.variables, self.bbox, self.attrs)

        size = max(self.batch_size, -(-len(files) // self.max_workers))
        parts = [files[i:i + size] for i in range(0, len(files), size)]
        dsets, errs = [], []
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            futs = [ex.submit(_read_chunk, p, self.variables, self.bbox, self.attrs) for p in parts]
            for fut in futs:
                try:
                    d, e = fut.result()
                    dsets.extend(d)
                    errs.extend(e)
                except Exception as e:  # e.g. worker killed (out of memory)
                    errs.append(f"worker failed: {type(e).__name__}: {e}")
        return dsets, errs

    def convert_month(self, y: int, m: int, overwrite: bool = False) -> Tuple[str, Optional[str]]:
        """Returns (status, error). status is 'ok' | 'skipped' | 'error'."""
        import xarray as xr

        files = self.raw_files(y, m)
        if not files:
            return "skipped", None
        store = self.store_path(y, m)

        if store.exists() and not overwrite and self.zarr_steps(store) == len(files):
            return "skipped", None  # already up to date
        if store.exists():
            shutil.rmtree(store)
        store.parent.mkdir(parents=True, exist_ok=True)

        try:
            dsets, errs = self._read_all(files)
            if errs:
                logger.warning("%d-%02d: %d/%d files unreadable, e.g. %s",
                               y, m, len(errs), len(files), errs[0])
            if not dsets:
                return "error", f"{y}_{m:02d}: no files could be read ({errs[0] if errs else ''})"

            ds = xr.concat(dsets, dim="time", coords="minimal", compat="override")
            ds = _sort_dedupe_time(ds)
            _write_zarr(ds, str(store), self.chunks)

            with xr.open_zarr(store) as chk:
                if not np.all(np.diff(chk.time.values) > np.timedelta64(0, "s")):
                    return "error", f"{y}_{m:02d}: written time axis is not monotonic"
            return "ok", None
        except Exception as e:
            if store.exists():
                shutil.rmtree(store, ignore_errors=True)
            return "error", f"{y}_{m:02d}: {type(e).__name__}: {e}"

    def convert(self, months: Sequence[Tuple[int, int]], overwrite: bool = False):
        stores, errors = [], []
        for y, m in tqdm(list(months), desc="Converting months to Zarr"):
            status, err = self.convert_month(y, m, overwrite)
            if err:
                errors.append(err)
                logger.error(err)
            if self.store_path(y, m).exists():
                stores.append(str(self.store_path(y, m)))
        return stores, errors


# =========================================================================== #
# 3. Reading Zarr
# =========================================================================== #
def open_imerg_zarr(zarr_dir, start_date: DateLike, end_date: DateLike,
                    variables: Optional[List[str]] = None):
    """Lazily open IMERG monthly Zarr stores for [start_date, end_date] (end inclusive)."""
    import xarray as xr

    start, end = _day_range(start_date, end_date)
    stores = sorted(Path(zarr_dir).glob("*/IMERG_*_*.zarr"))
    dsets = []
    for s in stores:
        try:
            y, m = s.name.replace("IMERG_", "").replace(".zarr", "").split("_")
            m_start = datetime(int(y), int(m), 1)
        except ValueError:
            continue
        m_end = (pd.Timestamp(m_start) + pd.offsets.MonthBegin(1)).to_pydatetime()
        if not (m_start <= end and m_end > start):
            continue
        ds = xr.open_zarr(s)
        if variables:
            keep = [v for v in variables if v in ds.data_vars]
            if not keep:
                continue
            ds = ds[keep]
        dsets.append(ds)

    if not dsets:
        raise RuntimeError(f"No Zarr data in {zarr_dir} for {start:%Y-%m-%d}..{end:%Y-%m-%d}.")
    out = xr.concat(dsets, dim="time") if len(dsets) > 1 else dsets[0]
    return out.sortby("time").sel(time=slice(start, end))


# =========================================================================== #
# Public entry points
# =========================================================================== #
def get_imerg(
    start_date: DateLike,
    end_date: DateLike,
    bbox: Sequence[float],
    output_dir: str,
    dataset_name: str = "GPM_3IMERGHH",
    convert_to_zarr: bool = True,
    max_workers: Optional[int] = None,
    *,
    has_account: Optional[bool] = None,
    version: Optional[str] = "07",
    variables: Optional[List[str]] = None,
    zarr_dir: Optional[str] = None,
    overwrite_zarr: bool = False,
    attrs_path: Optional[str] = None,
    retries: int = 3,
) -> IMERGResult:
    """Download IMERG for [start_date, end_date] (both inclusive) and optionally convert to Zarr.

    Parameters
    ----------
    start_date, end_date : "YYYY-MM-DD" (or datetime). End date is inclusive.
    bbox : (west, south, east, north) in degrees.
    output_dir : root folder; raw files go to <output_dir>/<dataset_name>/YYYY/MM/.
    dataset_name : CMR short name, e.g. "GPM_3IMERGHH".
    convert_to_zarr : also build <output_dir>/zarr/YYYY/IMERG_YYYY_MM.zarr, cropped to bbox
        (requires zarr>=3).
    has_account : None (default) = show the account guide and ask. True = skip the question
        (for batch jobs). Requires an Earthdata account either way.
    max_workers : optional. Parallel download threads / file readers. Default: CPU count - 1 (max 8).
    version : dataset version ("07"). None = no filter (may mix versions).
    variables : Zarr step only; variables to keep (None = all).
    zarr_dir : override the Zarr output folder.
    overwrite_zarr : rebuild the Zarr for the requested months even if up to date.
    attrs_path : optional JSON of per-variable attributes to attach.
    retries : attempts per month for search/download.

    IMERG granules are global files, so `bbox` does not shrink the download; it only
    crops the Zarr output.
    """
    start, end = _day_range(start_date, end_date)
    bbox_t = _validate_bbox(bbox)
    max_workers = _default_workers() if max_workers is None else int(max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1.")
    if convert_to_zarr:
        _check_zarr()          # fail now, not after a multi-GB download

    out_root = Path(output_dir)
    raw_root = out_root / dataset_name
    raw_root.mkdir(parents=True, exist_ok=True)
    zroot = Path(zarr_dir) if zarr_dir else out_root / "zarr"

    result = IMERGResult(start=start, end=end, bbox=bbox_t, dataset_name=dataset_name,
                         raw_dir=raw_root, zarr_dir=zroot if convert_to_zarr else None)

    _ensure_account(has_account)
    _login()
    windows = _month_windows(start, end)
    logger.info("Downloading %s %s..%s -> %s (%d month window%s)", dataset_name,
                f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", raw_root,
                len(windows), "" if len(windows) == 1 else "s")

    for win in windows:
        month_dir = raw_root / str(win[0].year) / f"{win[0].month:02d}"
        files, failed, empty = _download_month(
            dataset_name, version, bbox_t, win, month_dir, max_workers, retries)
        result.files.extend(files)
        if failed:
            result.failed_months.append(f"{win[0]:%Y-%m}")
        if empty:
            result.empty_months.append(f"{win[0]:%Y-%m}")

    logger.info("Download finished: %d files, %d failed month(s), %d empty month(s)",
                len(result.files), len(result.failed_months), len(result.empty_months))

    if convert_to_zarr:
        months = sorted({(w[0].year, w[0].month) for w in windows})
        conv = _ZarrConverter(raw_root, zroot, bbox_t, variables, attrs_path, max_workers)
        result.zarr_stores, result.zarr_errors = conv.convert(months, overwrite_zarr)

    return result


def convert_to_zarr_only(
    raw_dir: str,
    zarr_dir: str,
    bbox: Optional[Sequence[float]] = None,
    years: Optional[Sequence[Union[int, str]]] = None,
    variables: Optional[List[str]] = None,
    max_workers: Optional[int] = None,
    overwrite: bool = False,
    attrs_path: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Convert already-downloaded <raw_dir>/<YYYY>/<MM>/*.HDF5 files. Returns (stores, errors)."""
    _check_zarr()
    max_workers = _default_workers() if max_workers is None else int(max_workers)
    raw = Path(raw_dir)
    months = []
    for ydir in sorted(p for p in raw.iterdir() if p.is_dir() and p.name.isdigit()):
        if years and int(ydir.name) not in {int(y) for y in years}:
            continue
        for mdir in sorted(p for p in ydir.iterdir() if p.is_dir() and p.name.isdigit()):
            months.append((int(ydir.name), int(mdir.name)))
    b = _validate_bbox(bbox) if bbox is not None else None
    conv = _ZarrConverter(raw, Path(zarr_dir), b, variables, attrs_path, max_workers)
    return conv.convert(months, overwrite)


__all__ = ["get_imerg", "IMERGResult", "convert_to_zarr_only", "open_imerg_zarr"]
