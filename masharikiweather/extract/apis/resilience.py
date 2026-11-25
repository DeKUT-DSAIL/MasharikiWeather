"""
Extraction Resilience, Progress Reporting, Rate Limiting, Caching, and Dry-Run Estimation.

Provides thread-safe checkpointing/manifest caching, token-bucket rate limiting,
exponential backoff with Retry-After support, source-aware progress bars, and
zero-network pre-flight extraction estimation for MasharikiWeather.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import threading
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pandas as pd
import xarray as xr

T = TypeVar("T")

logger = logging.getLogger("MasharikiWeather.Extract.Resilience")

TAHMO_MAX_WORKERS_CAP: int = 16
GHCN_MAX_WORKERS_CAP: int = 24
MAX_WORKERS_CAP: int = TAHMO_MAX_WORKERS_CAP

SOURCE_MAX_WORKERS_CAPS: Dict[str, Optional[int]] = {
    "tahmo": TAHMO_MAX_WORKERS_CAP,
    "ghcn": GHCN_MAX_WORKERS_CAP,
    "ghcnd": GHCN_MAX_WORKERS_CAP,
    "ghcn_daily": GHCN_MAX_WORKERS_CAP,
    "ghcn_hourly": GHCN_MAX_WORKERS_CAP,
}


def clamp_max_workers(
    max_workers: Optional[int],
    *,
    default: int = 4,
    max_cap: Optional[int] = None,
    source: str = "",
) -> int:
    """
    Clamp thread-pool concurrency to [1, effective_cap] with explicit warnings.

    TAHMO is capped at 16, GHCN sources are capped at 24, and other sources
    (e.g., IMERG, ERA5, reanalysis, satellite) scale without a 16-worker cap
    unless ``max_cap`` is explicitly supplied.

    Args:
        max_workers: Requested worker thread count, or None to use default.
        default: Fallback worker count when max_workers is None.
        max_cap: Optional explicit upper bound on concurrent workers. If None,
            looks up ``SOURCE_MAX_WORKERS_CAPS`` for ``source``.
        source: Optional source name for cap lookup and warning messages.

    Returns:
        Bounded integer worker count (>= 1).
    """
    if max_workers is None:
        return default
    val = int(max_workers)
    src_key = source.lower().strip()
    src_note = f" for {source}" if source else ""
    if val < 1:
        msg = f"Requested max_workers={val} is below minimum (1){src_note}; clamping to 1."
        warnings.warn(msg, UserWarning, stacklevel=2)
        logger.warning(msg)
        return 1
    effective_cap = max_cap if max_cap is not None else SOURCE_MAX_WORKERS_CAPS.get(src_key)
    if effective_cap is not None and val > effective_cap:
        msg = f"Requested max_workers={val} exceeds maximum allowed ({effective_cap}){src_note}; capping to {effective_cap}."
        warnings.warn(msg, UserWarning, stacklevel=2)
        logger.warning(msg)
        return effective_cap
    return val

SOURCE_PROGRESS_PROFILES: Dict[str, Dict[str, str]] = {
    "tahmo": {"unit": "station", "label": "TAHMO stations"},
    "ghcnd": {"unit": "station", "label": "GHCN stations"},
    "cds_insitu": {"unit": "chunk", "label": "CDS in-situ chunks"},
    "era5": {"unit": "chunk", "label": "ERA5 chunks"},
    "era5_arco": {"unit": "chunk", "label": "ERA5 ARCO chunks"},
    "imerg": {"unit": "granule", "label": "IMERG granules"},
    "cams": {"unit": "chunk", "label": "CAMS chunks"},
    "merra2": {"unit": "chunk", "label": "MERRA-2 chunks"},
    "open_meteo": {"unit": "tile", "label": "Open-Meteo tiles"},
    "nasa_power": {"unit": "tile", "label": "NASA POWER tiles"},
    "pvgis": {"unit": "tile", "label": "PVGIS tiles"},
    "sarah3": {"unit": "granule", "label": "SARAH-3 granules"},
    "tropomi": {"unit": "granule", "label": "TROPOMI granules"},
    "landsaf": {"unit": "granule", "label": "LANDSAF granules"},
}

_SOURCE_ALIASES: Dict[str, str] = {
    "ghcn": "ghcnd",
    "ghcn_daily": "ghcnd",
    "ghcn_hourly": "ghcnd",
    "cds-insitu": "cds_insitu",
    "cds": "cds_insitu",
    "insitu": "cds_insitu",
    "copernicus_cams": "cams",
    "nasa_merra2": "merra2",
    "openmeteo": "open_meteo",
    "power": "nasa_power",
}


class _NoOpProgressBar:
    """Minimal no-op fallback progress bar when tqdm is unavailable."""

    def __init__(
        self,
        iterable: Optional[Iterable[T]] = None,
        *,
        total: Optional[int] = None,
        initial: int = 0,
        desc: Optional[str] = None,
        unit: str = "it",
        disable: bool = False,
        **kwargs: Any,
    ) -> None:
        self._iterable = iterable
        self.total = total
        self.n = initial
        self.desc = desc
        self.unit = unit
        self.disable = disable

    def __iter__(self) -> Iterator[T]:
        if self._iterable is None:
            return iter(())
        for item in self._iterable:
            yield item
            self.n += 1

    def __enter__(self) -> _NoOpProgressBar:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        self.close()
        return False

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_description(self, desc: Optional[str] = None, refresh: bool = True) -> None:
        self.desc = desc

    def set_postfix(self, ordered_dict: Optional[Dict[str, Any]] = None, refresh: bool = True, **kwargs: Any) -> None:
        return None

    def refresh(self) -> None:
        return None

    def write(self, s: str, file: Any = None, end: str = "\n", nolock: bool = False) -> None:
        return None

    def close(self) -> None:
        return None


def _normalize_source_name(source: str) -> str:
    src = source.lower().strip()
    return _SOURCE_ALIASES.get(src, src)


def make_progress_bar(
    iterable: Optional[Iterable[T]] = None,
    *,
    source: str = "base",
    unit: Optional[str] = None,
    total: Optional[int] = None,
    initial: int = 0,
    desc: Optional[str] = None,
    show_progress: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Create a source-aware tqdm progress bar with a no-op fallback.

    Args:
        iterable: Optional iterable to wrap.
        source: Source identifier used to look up unit and label defaults.
        unit: Explicit unit override (e.g., "station", "chunk", "granule").
        total: Expected number of iterations.
        initial: Initial counter value.
        desc: Explicit description override.
        show_progress: If False, disables progress bar output.
        **kwargs: Additional keyword arguments forwarded to tqdm.

    Returns:
        Configured tqdm progress bar or _NoOpProgressBar fallback.
    """
    norm_source = _normalize_source_name(source)
    profile = SOURCE_PROGRESS_PROFILES.get(norm_source, {"unit": "item", "label": f"{source} items"})

    resolved_unit = unit if unit is not None else profile["unit"]
    resolved_desc = desc if desc is not None else profile["label"]

    tqdm_kwargs = dict(kwargs)
    tqdm_kwargs.setdefault("dynamic_ncols", True)
    tqdm_kwargs["disable"] = not show_progress

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        return _NoOpProgressBar(
            iterable=iterable,
            total=total,
            initial=initial,
            desc=resolved_desc,
            unit=resolved_unit,
            **tqdm_kwargs,
        )

    return _tqdm(
        iterable=iterable,
        total=total,
        initial=initial,
        desc=resolved_desc,
        unit=resolved_unit,
        **tqdm_kwargs,
    )


class RateLimiter:
    """
    Thread-safe token-bucket rate limiter with context manager support.

    Args:
        requests_per_second: Sustained request rate (> 0).
        burst: Maximum token burst capacity (>= 1).
    """

    def __init__(self, requests_per_second: float, burst: int = 1) -> None:
        if requests_per_second <= 0:
            raise ValueError(f"requests_per_second must be > 0, got {requests_per_second}")
        if burst < 1:
            raise ValueError(f"burst must be >= 1, got {burst}")

        self.requests_per_second = float(requests_per_second)
        self.burst = int(burst)
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> float:
        """
        Block until the requested number of tokens are available.

        Args:
            tokens: Number of tokens to consume (1 <= tokens <= burst).

        Returns:
            Total sleep duration in seconds incurred while waiting.
        """
        if tokens < 1 or tokens > self.burst:
            raise ValueError(f"tokens must be in [1, {self.burst}], got {tokens}")

        total_waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self._last_refill)
                self._tokens = min(float(self.burst), self._tokens + elapsed * self.requests_per_second)
                self._last_refill = now

                if self._tokens >= tokens:
                    self._tokens -= float(tokens)
                    return total_waited

                deficit = float(tokens) - self._tokens
                wait_seconds = deficit / self.requests_per_second

            time.sleep(wait_seconds)
            total_waited += wait_seconds

    def __enter__(self) -> RateLimiter:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        return False


def _extract_retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Parse HTTP Retry-After header from an exception's response attribute if present."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    raw_val = None
    for key in ("Retry-After", "retry-after"):
        if hasattr(headers, "get"):
            raw_val = headers.get(key)
            if raw_val is not None:
                break
    if raw_val is None:
        return None

    text = str(raw_val).strip()
    try:
        seconds = float(text)
        return max(0.0, seconds)
    except ValueError:
        pass

    try:
        target_dt = parsedate_to_datetime(text)
        now_ts = time.time()
        return max(0.0, target_dt.timestamp() - now_ts)
    except (TypeError, ValueError):
        return None


def retry_with_backoff(
    func: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    rate_limiter: Optional[RateLimiter] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> T:
    """
    Execute a callable with exponential backoff, jitter, and Retry-After handling.

    Args:
        func: Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts after the initial call.
        base_delay: Initial backoff delay in seconds.
        max_delay: Maximum backoff delay cap in seconds.
        jitter: If True, applies randomized multiplicative jitter in [0.5, 1.0].
        retry_on: Tuple of exception classes that trigger a retry.
        rate_limiter: Optional RateLimiter acquired before each attempt.
        on_retry: Optional callback invoked as on_retry(attempt, exc, sleep_seconds).

    Returns:
        Return value of func().
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")

    attempt = 0
    while True:
        if rate_limiter is not None:
            rate_limiter.acquire()
        try:
            return func()
        except retry_on as exc:
            if attempt >= max_retries:
                raise

            delay = min(max_delay, base_delay * (2 ** attempt))
            if jitter:
                delay = delay * (0.5 + 0.5 * random.random())

            retry_after = _extract_retry_after_seconds(exc)
            if retry_after is not None:
                delay = min(max_delay, max(delay, retry_after))

            if on_retry is not None:
                on_retry(attempt + 1, exc, delay)

            time.sleep(delay)
            attempt += 1


@dataclass
class ExtractionManifestSummary:
    """Summary statistics for a cached extraction run."""

    source: str
    request_hash: str
    cache_dir: Path
    total: int
    completed: int
    cached_hits: int
    newly_fetched: int
    empty: int
    failed: int
    failed_items: Dict[str, str]
    elapsed_seconds: float


def _canonicalize_value(val: Any) -> Any:
    """Convert arbitrary request parameter values into deterministic JSON-serializable structures."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(val).isoformat()
    if hasattr(val, "lat_min") and hasattr(val, "lat_max") and hasattr(val, "lon_min") and hasattr(val, "lon_max"):
        return {
            "lat_max": float(val.lat_max),
            "lat_min": float(val.lat_min),
            "lon_max": float(val.lon_max),
            "lon_min": float(val.lon_min),
        }
    if isinstance(val, dict):
        return {str(k): _canonicalize_value(v) for k, v in sorted(val.items(), key=lambda item: str(item[0]))}
    if isinstance(val, (list, tuple)):
        return [_canonicalize_value(v) for v in val]
    if isinstance(val, (set, frozenset)):
        items = [_canonicalize_value(v) for v in val]
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True))
    return str(val)


DEFAULT_CACHE_TTL_DAYS: float = 7.0
_KEY_EXPLICIT_DATE_RE = re.compile(r"(?:\d{8}_\d{8}|^\d{4}-\d{2}$)")


def _compute_request_hash(source: str, request_params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Compute deterministic 16-character SHA-256 prefix from source and canonicalized parameters."""
    canonical_params = _canonicalize_value(request_params)
    payload = {
        "source": _normalize_source_name(source),
        "params": canonical_params,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return digest, canonical_params


def _sanitize_key(key: str) -> str:
    """Convert an arbitrary partition key into a filesystem-safe stem."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key).strip())
    if not cleaned:
        cleaned = "item"
    if len(cleaned) > 120:
        suffix = hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[:100]}_{suffix}"
    return cleaned


class ExtractionCache:
    """
    Thread-safe, manifest-backed extraction checkpoint cache.

    Partitions are stored under:
        Path(output_dir) / ".extract_cache" / <source> / <request_hash>
    alongside a `manifest.json` tracking `"pending"`, `"completed"`, `"empty"`,
    and `"failed"` partition states.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        source: str,
        request_params: Dict[str, Any],
        resume: bool = True,
        overwrite: bool = False,
        cache_ttl_days: Optional[float] = DEFAULT_CACHE_TTL_DAYS,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.source = _normalize_source_name(source)
        self.resume = bool(resume)
        self.overwrite = bool(overwrite)
        self.cache_ttl_days = cache_ttl_days
        self.request_hash, canonical_params = _compute_request_hash(self.source, request_params)
        self._canonical_params: Dict[str, Any] = canonical_params
        self.cache_dir = self.output_dir / ".extract_cache" / self.source / self.request_hash
        self.manifest_path = self.cache_dir / "manifest.json"
        self._lock = threading.RLock()
        self._start_time = time.monotonic()
        self._cached_hits: Set[str] = set()
        self._newly_fetched: Set[str] = set()

        if self.cache_ttl_days is not None and self.cache_ttl_days > 0:
            self.prune_expired(max_age_days=self.cache_ttl_days)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: Dict[str, Any] = self._init_manifest(canonical_params)

    def _init_manifest(self, canonical_params: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self.resume and not self.overwrite and self.manifest_path.is_file():
                try:
                    with open(self.manifest_path, "r", encoding="utf-8") as fh:
                        loaded = json.load(fh)
                    if isinstance(loaded, dict) and isinstance(loaded.get("partitions"), dict):
                        return loaded
                except (OSError, json.JSONDecodeError):
                    pass

            manifest = {
                "source": self.source,
                "request_hash": self.request_hash,
                "request_params": canonical_params,
                "updated_at": time.time(),
                "partitions": {},
            }
            self._write_manifest_locked(manifest)
            return manifest

    def _write_manifest_locked(self, manifest: Optional[Dict[str, Any]] = None) -> None:
        target = manifest if manifest is not None else self._manifest
        target["updated_at"] = time.time()
        tmp_path = self.cache_dir / f"manifest.json.tmp.{uuid.uuid4().hex}"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(target, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, self.manifest_path)

    def _set_partition_locked(
        self,
        key: str,
        status: Literal["pending", "completed", "empty", "failed"],
        rel_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self._manifest["partitions"][str(key)] = {
            "status": status,
            "path": rel_path,
            "error": error,
        }
        self._write_manifest_locked()

    def _is_peer_manifest_compatible(self, peer_params: Dict[str, Any], key: str) -> bool:
        """Return True if a sibling cache manifest is safe to reuse for partition `key`."""
        if not isinstance(peer_params, dict):
            return False

        cur_vars = self._canonical_params.get("variables")
        peer_vars = peer_params.get("variables")
        if peer_vars is not None:
            if cur_vars is None:
                return False
            if not set(str(v) for v in cur_vars).issubset(set(str(v) for v in peer_vars)):
                return False

        for strict_key in ("resolution", "format", "apply_qc", "parquet_layout", "dataset_name", "region"):
            if strict_key in self._canonical_params or strict_key in peer_params:
                if self._canonical_params.get(strict_key) != peer_params.get(strict_key):
                    return False

        if _KEY_EXPLICIT_DATE_RE.search(str(key)):
            return True

        return (
            peer_params.get("start_date") == self._canonical_params.get("start_date")
            and peer_params.get("end_date") == self._canonical_params.get("end_date")
        )

    def _promote_from_peer_caches_locked(self, key: str) -> bool:
        """Search sibling request hashes for a completed `key` and promote it locally."""
        source_root = self.output_dir / ".extract_cache" / self.source
        if not source_root.is_dir():
            return False

        for peer_dir in sorted(source_root.iterdir()):
            if not peer_dir.is_dir() or peer_dir.name == self.request_hash:
                continue
            peer_manifest_path = peer_dir / "manifest.json"
            if not peer_manifest_path.is_file():
                continue
            try:
                with open(peer_manifest_path, "r", encoding="utf-8") as fh:
                    peer_manifest = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue

            if not self._is_peer_manifest_compatible(peer_manifest.get("request_params", {}), key):
                continue

            entry = peer_manifest.get("partitions", {}).get(str(key))
            if not entry or entry.get("status") != "completed":
                continue

            rel_path = entry.get("path")
            if not rel_path:
                continue
            src_target = peer_dir / rel_path
            if not src_target.exists():
                continue

            dest_name = Path(rel_path).name
            dest_target = self.cache_dir / dest_name
            try:
                if src_target.is_file() and src_target.stat().st_size > 0:
                    try:
                        if dest_target.exists():
                            dest_target.unlink()
                        os.link(src_target, dest_target)
                    except OSError:
                        shutil.copy2(src_target, dest_target)
                    self._set_partition_locked(key, "completed", rel_path=dest_name, error=None)
                    return True
                if src_target.is_dir() and any(src_target.iterdir()):
                    if dest_target.exists():
                        shutil.rmtree(dest_target)
                    shutil.copytree(src_target, dest_target)
                    self._set_partition_locked(key, "completed", rel_path=dest_name, error=None)
                    return True
            except OSError:
                continue

        return False

    def prune_expired(self, max_age_days: float = DEFAULT_CACHE_TTL_DAYS) -> int:
        """Remove sibling request-hash cache directories older than `max_age_days`."""
        if max_age_days <= 0:
            return 0
        source_root = self.output_dir / ".extract_cache" / self.source
        if not source_root.is_dir():
            return 0

        cutoff = time.time() - (float(max_age_days) * 86400.0)
        removed = 0
        for peer_dir in list(source_root.iterdir()):
            if not peer_dir.is_dir() or peer_dir.name == self.request_hash:
                continue
            manifest_p = peer_dir / "manifest.json"
            updated_at: Optional[float] = None
            if manifest_p.is_file():
                try:
                    with open(manifest_p, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, dict) and isinstance(data.get("updated_at"), (int, float)):
                        updated_at = float(data["updated_at"])
                except (OSError, json.JSONDecodeError):
                    pass
            if updated_at is None:
                try:
                    updated_at = peer_dir.stat().st_mtime
                except OSError:
                    continue
            if updated_at < cutoff:
                shutil.rmtree(peer_dir, ignore_errors=True)
                removed += 1
        return removed

    def is_done(self, key: str) -> bool:
        """Return True if key is already completed (locally or in a compatible peer hash) or empty."""
        if not self.resume or self.overwrite:
            return False
        with self._lock:
            entry = self._manifest["partitions"].get(str(key))
            if not entry:
                if self._promote_from_peer_caches_locked(key):
                    entry = self._manifest["partitions"].get(str(key))
                else:
                    return False
            if not entry:
                return False
            status = entry.get("status")
            if status == "empty":
                return True
            if status != "completed":
                return False
            rel_path = entry.get("path")
            if not rel_path:
                return False
            target = self.cache_dir / rel_path
            if target.is_file():
                return target.stat().st_size > 0
            if target.is_dir():
                return any(target.iterdir())
            return False

    def get_path(self, key: str) -> Optional[Path]:
        """Return the cached file or directory Path for key if recorded and present."""
        with self._lock:
            entry = self._manifest["partitions"].get(str(key))
            if not entry and self.resume and not self.overwrite:
                if self._promote_from_peer_caches_locked(key):
                    entry = self._manifest["partitions"].get(str(key))
            if not entry:
                return None
            rel_path = entry.get("path")
            if not rel_path:
                return None
            target = self.cache_dir / rel_path
            return target if target.exists() else None

    def save_parquet(self, key: str, df: pd.DataFrame) -> Path:
        """Atomically write a DataFrame as Parquet and mark partition completed."""
        safe_stem = _sanitize_key(key)
        filename = f"{safe_stem}.parquet"
        target = self.cache_dir / filename
        tmp_path = self.cache_dir / f"{safe_stem}.{uuid.uuid4().hex}.parquet.tmp"
        df.to_parquet(tmp_path)
        os.replace(tmp_path, target)
        with self._lock:
            self._newly_fetched.add(str(key))
            self._set_partition_locked(key, "completed", rel_path=filename, error=None)
        return target

    def save_dataset(
        self,
        key: str,
        ds: xr.Dataset,
        fmt: Literal["netcdf", "zarr"] = "netcdf",
    ) -> Path:
        """Atomically write an xarray Dataset as NetCDF or Zarr and mark partition completed."""
        safe_stem = _sanitize_key(key)
        if fmt == "zarr":
            filename = f"{safe_stem}.zarr"
            target = self.cache_dir / filename
            tmp_path = self.cache_dir / f"{safe_stem}.{uuid.uuid4().hex}.zarr.tmp"
            ds.to_zarr(tmp_path, mode="w", zarr_format=3)
            with self._lock:
                if target.exists():
                    shutil.rmtree(target)
                os.replace(tmp_path, target)
                self._newly_fetched.add(str(key))
                self._set_partition_locked(key, "completed", rel_path=filename, error=None)
            return target

        filename = f"{safe_stem}.nc"
        target = self.cache_dir / filename
        tmp_path = self.cache_dir / f"{safe_stem}.{uuid.uuid4().hex}.nc.tmp"
        ds.to_netcdf(tmp_path)
        os.replace(tmp_path, target)
        with self._lock:
            self._newly_fetched.add(str(key))
            self._set_partition_locked(key, "completed", rel_path=filename, error=None)
        return target

    def save_bytes(self, key: str, data: bytes, suffix: str = ".bin") -> Path:
        """Atomically write raw bytes and mark partition completed."""
        norm_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        safe_stem = _sanitize_key(key)
        filename = f"{safe_stem}{norm_suffix}"
        target = self.cache_dir / filename
        tmp_path = self.cache_dir / f"{safe_stem}.{uuid.uuid4().hex}{norm_suffix}.tmp"
        with open(tmp_path, "wb") as fh:
            fh.write(data)
        os.replace(tmp_path, target)
        with self._lock:
            self._newly_fetched.add(str(key))
            self._set_partition_locked(key, "completed", rel_path=filename, error=None)
        return target

    def load_parquet(self, key: str) -> pd.DataFrame:
        """Load a cached Parquet partition for key."""
        path = self.get_path(key)
        if path is None:
            raise FileNotFoundError(f"No cached Parquet partition found for key '{key}' in {self.cache_dir}")
        return pd.read_parquet(path)

    def load_dataset(self, key: str, **kwargs: Any) -> xr.Dataset:
        """Load a cached NetCDF or Zarr dataset partition for key."""
        path = self.get_path(key)
        if path is None:
            raise FileNotFoundError(f"No cached Dataset partition found for key '{key}' in {self.cache_dir}")
        if path.is_dir() or path.suffix == ".zarr":
            return xr.open_zarr(path, **kwargs)
        return xr.open_dataset(path, **kwargs)

    def mark_empty(self, key: str) -> None:
        """Mark partition as empty (successfully queried, zero records returned)."""
        with self._lock:
            self._set_partition_locked(key, "empty", rel_path=None, error=None)

    def mark_failed(self, key: str, error: Union[str, Exception]) -> None:
        """Mark partition as failed with its error message."""
        with self._lock:
            self._set_partition_locked(key, "failed", rel_path=None, error=str(error))

    def get_failed(self) -> Dict[str, str]:
        """Return a mapping of failed partition keys to their error descriptions."""
        with self._lock:
            return {
                k: str(v.get("error") or "")
                for k, v in self._manifest["partitions"].items()
                if v.get("status") == "failed"
            }

    def record_hit(self, key: str) -> None:
        """Record a cache hit for summary accounting."""
        with self._lock:
            self._cached_hits.add(str(key))

    def summary(self) -> ExtractionManifestSummary:
        """Return an ExtractionManifestSummary snapshot of cache state."""
        with self._lock:
            partitions = self._manifest["partitions"]
            total = len(partitions)
            completed = sum(1 for v in partitions.values() if v.get("status") == "completed")
            empty = sum(1 for v in partitions.values() if v.get("status") == "empty")
            failed_items = {
                k: str(v.get("error") or "")
                for k, v in partitions.items()
                if v.get("status") == "failed"
            }
            return ExtractionManifestSummary(
                source=self.source,
                request_hash=self.request_hash,
                cache_dir=self.cache_dir,
                total=total,
                completed=completed,
                cached_hits=len(self._cached_hits),
                newly_fetched=len(self._newly_fetched),
                empty=empty,
                failed=len(failed_items),
                failed_items=failed_items,
                elapsed_seconds=max(0.0, time.monotonic() - self._start_time),
            )

    def cleanup(self, keep_on_failure: bool = True) -> bool:
        """
        Remove the cache directory if no failures occurred (or keep_on_failure is False).

        Returns:
            True if the cache directory was removed, False otherwise.
        """
        with self._lock:
            failed = self.get_failed()
            if failed and keep_on_failure:
                return False
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir, ignore_errors=True)
                return True
            return False


@dataclass
class ExtractionEstimate:
    """Pre-flight dry-run estimate of extraction workload, size, and duration."""

    source: str
    unit_name: str
    num_units: int
    estimated_requests: int
    estimated_size_mb: float
    estimated_duration_seconds: float
    warnings: List[str] = field(default_factory=list)
    summary_text: str = ""


def _parse_bbox_dims(bbox: Optional[Union[Any, Sequence[float]]]) -> Tuple[float, float]:
    """Return (lat_span, lon_span) in degrees without network calls."""
    if bbox is None:
        return 73.0, 70.0  # Continental Africa default span
    if hasattr(bbox, "lat_min") and hasattr(bbox, "lat_max") and hasattr(bbox, "lon_min") and hasattr(bbox, "lon_max"):
        return max(0.1, float(bbox.lat_max) - float(bbox.lat_min)), max(0.1, float(bbox.lon_max) - float(bbox.lon_min))
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        lat_min, lon_min, lat_max, lon_max = [float(x) for x in bbox]
        return max(0.1, lat_max - lat_min), max(0.1, lon_max - lon_min)
    return 10.0, 10.0


def estimate_extraction(
    source: str,
    start_date: Union[str, datetime, date, pd.Timestamp],
    end_date: Union[str, datetime, date, pd.Timestamp],
    bbox: Optional[Union[Any, Sequence[float]]] = None,
    variables: Optional[Sequence[str]] = None,
    stations: Optional[Sequence[str]] = None,
    levels: Optional[Sequence[int]] = None,
    chunk_days: int = 5,
    use_arco: Optional[bool] = None,
) -> ExtractionEstimate:
    """
    Estimate extraction units, API requests, payload size, and duration without network calls.

    Args:
        source: Data source identifier.
        start_date: Start timestamp of extraction range.
        end_date: End timestamp of extraction range.
        bbox: Optional spatial bounding box or [lat_min, lon_min, lat_max, lon_max].
        variables: Optional requested variable sequence.
        stations: Optional explicit station identifier sequence.
        levels: Optional pressure levels sequence (hPa).
        chunk_days: Temporal chunk size in days for chunked extractors.
        use_arco: Optional flag indicating cloud ARCO preference for ERA5.

    Returns:
        ExtractionEstimate containing pre-flight metrics and warnings.
    """
    norm_source = _normalize_source_name(source)
    t_start = pd.Timestamp(start_date)
    t_end = pd.Timestamp(end_date)
    n_days = max(1, int((t_end - t_start).days) + 1)
    lat_span, lon_span = _parse_bbox_dims(bbox)
    area_deg2 = lat_span * lon_span
    n_vars = max(1, len(variables)) if variables else 5
    n_levels = max(1, len(levels)) if levels else 1
    safe_chunk_days = max(1, int(chunk_days))

    profile = SOURCE_PROGRESS_PROFILES.get(norm_source, {"unit": "chunk", "label": f"{source} chunks"})
    unit_name = profile["unit"]
    warnings_list: List[str] = []

    if norm_source in ("tahmo", "ghcnd"):
        if stations is not None:
            num_units = max(1, len(stations))
        else:
            num_units = max(1, min(600, int(round(area_deg2 * 0.15))))
            warnings_list.append(
                "Station count is estimated from bounding box area because remote station catalog was not queried."
            )
        if norm_source == "tahmo":
            req_per_station = max(1, math.ceil(n_days / 30))
            estimated_requests = num_units * req_per_station
            estimated_size_mb = round(num_units * n_days * n_vars * 0.005, 3)
            estimated_duration_seconds = round(estimated_requests * 0.35, 2)
        else:
            estimated_requests = num_units
            estimated_size_mb = round(num_units * n_days * n_vars * 0.002, 3)
            estimated_duration_seconds = round(estimated_requests * 0.25, 2)

    elif norm_source == "cds_insitu":
        num_units = max(1, math.ceil(n_days / safe_chunk_days))
        estimated_requests = num_units
        estimated_size_mb = round(n_days * n_vars * 0.5, 3)
        estimated_duration_seconds = round(num_units * 15.0, 2)

    elif norm_source in ("era5", "era5_arco"):
        num_units = max(1, math.ceil(n_days / safe_chunk_days))
        estimated_requests = num_units
        grid_pts = max(1.0, (lat_span / 0.25) * (lon_span / 0.25))
        bytes_per_val = 4.0
        estimated_size_mb = round((grid_pts * n_days * 24.0 * n_vars * n_levels * bytes_per_val) / (1024.0 ** 2), 3)
        arco_mode = bool(use_arco) or norm_source == "era5_arco"
        if arco_mode:
            estimated_duration_seconds = round(num_units * 3.0 + estimated_size_mb * 0.1, 2)
        else:
            estimated_duration_seconds = round(num_units * 45.0 + estimated_size_mb * 0.2, 2)
            if num_units > 10:
                warnings_list.append(
                    f"CDS ERA5 request spans {num_units} chunks; queue latency may be significant. "
                    "Consider use_arco=True or a larger chunk_days."
                )

    elif norm_source == "imerg":
        num_units = n_days * 48
        estimated_requests = num_units
        grid_pts = max(1.0, (lat_span / 0.1) * (lon_span / 0.1))
        estimated_size_mb = round((grid_pts * num_units * n_vars * 4.0) / (1024.0 ** 2), 3)
        estimated_duration_seconds = round(num_units * 0.4, 2)

    elif norm_source in ("cams", "merra2"):
        num_units = max(1, math.ceil(n_days / safe_chunk_days))
        estimated_requests = num_units
        grid_pts = max(1.0, (lat_span / 0.5) * (lon_span / 0.5))
        estimated_size_mb = round((grid_pts * n_days * 8.0 * n_vars * n_levels * 4.0) / (1024.0 ** 2), 3)
        estimated_duration_seconds = round(num_units * 12.0, 2)

    elif norm_source in ("open_meteo", "nasa_power", "pvgis"):
        num_units = max(1, math.ceil(lat_span / 2.0) * math.ceil(lon_span / 2.0))
        estimated_requests = num_units * max(1, math.ceil(n_days / 365))
        estimated_size_mb = round(num_units * n_days * n_vars * 0.004, 3)
        estimated_duration_seconds = round(estimated_requests * 0.5, 2)

    else:
        # sarah3, tropomi, landsaf, or generic gridded source
        num_units = n_days
        estimated_requests = num_units
        grid_pts = max(1.0, (lat_span / 0.05) * (lon_span / 0.05))
        estimated_size_mb = round((grid_pts * n_days * n_vars * 4.0) / (1024.0 ** 2), 3)
        estimated_duration_seconds = round(num_units * 2.0, 2)

    if n_days > 365:
        warnings_list.append(f"Requested temporal span ({n_days} days) exceeds 1 year; consider splitting by year.")
    if estimated_size_mb > 2048.0:
        warnings_list.append(
            f"Estimated payload size ({estimated_size_mb:.1f} MB) exceeds 2 GB; verify available disk space."
        )

    lines = [
        f"Extraction dry-run estimate for '{norm_source}':",
        f"  Date range: {t_start.date().isoformat()} to {t_end.date().isoformat()} ({n_days} days)",
        f"  Units: {num_units} {unit_name}(s)",
        f"  Estimated requests: {estimated_requests}",
        f"  Estimated size: {estimated_size_mb:.2f} MB",
        f"  Estimated duration: {estimated_duration_seconds:.1f} s",
    ]
    if warnings_list:
        lines.append("  Warnings:")
        for w in warnings_list:
            lines.append(f"    - {w}")

    return ExtractionEstimate(
        source=norm_source,
        unit_name=unit_name,
        num_units=num_units,
        estimated_requests=estimated_requests,
        estimated_size_mb=estimated_size_mb,
        estimated_duration_seconds=estimated_duration_seconds,
        warnings=warnings_list,
        summary_text="\n".join(lines),
    )


def get_available_memory_mb() -> Optional[float]:
    """
    Return currently available system RAM in megabytes, or None if unavailable.

    Reads ``MemAvailable`` from ``/proc/meminfo`` on Linux and falls back to
    ``psutil.virtual_memory().available`` when available.
    """
    proc_meminfo = Path("/proc/meminfo")
    if proc_meminfo.is_file():
        try:
            with open(proc_meminfo, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            kb = float(parts[1])
                            return kb / 1024.0
        except (OSError, ValueError):
            pass

    try:
        import psutil  # type: ignore[import-not-found]

        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except Exception:
        return None


def allocate_worker_budget(
    num_sources: int,
    requested_max_workers: Optional[int] = None,
    *,
    max_cap: Optional[int] = None,
    default: int = 4,
) -> Tuple[int, int]:
    """
    Split a worker budget across concurrent sources.

    Bounds ``outer_workers`` to ``min(num_sources, 4)`` while allowing
    non-capped sources (e.g., IMERG) to scale beyond 16 when ``max_cap`` is None.

    Args:
        num_sources: Number of concurrent data sources to execute.
        requested_max_workers: Optional user-requested total worker budget.
        max_cap: Optional upper bound on total concurrent workers.
        default: Default worker budget when ``requested_max_workers`` is None.

    Returns:
        Tuple of ``(outer_workers, per_source_workers)``.
    """
    total_budget = clamp_max_workers(
        requested_max_workers,
        default=default,
        max_cap=max_cap,
        source="multi_source",
    )
    outer_workers = max(1, min(int(num_sources), 4))
    per_source_workers = max(1, total_budget // outer_workers)
    return outer_workers, per_source_workers


def _align_table_to_schema(table: Any, target_schema: Any) -> Any:
    """Align a PyArrow Table to a unified target schema, padding missing columns with nulls."""
    import pyarrow as pa

    if table.schema.equals(target_schema):
        return table

    col_names = set(table.column_names)
    arrays = []
    for field in target_schema:
        if field.name in col_names:
            col = table.column(field.name)
            if not col.type.equals(field.type):
                col = col.cast(field.type)
            arrays.append(col)
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(arrays, schema=target_schema)


def stream_merge_parquet(
    parquet_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    *,
    batch_size: int = 20,
    delete_inputs: bool = False,
) -> Path:
    """
    Merge multiple Parquet files into one output file with bounded memory usage.

    Unifies schemas across heterogeneous station/partition files and streams
    row groups in batches via ``pyarrow.parquet.ParquetWriter``, writing
    atomically through a temporary file and invoking ``gc.collect()`` per batch.

    Args:
        parquet_paths: Input Parquet file paths to merge.
        output_path: Destination merged Parquet file path.
        batch_size: Number of input files processed before running garbage collection.
        delete_inputs: If True, removes input files after successful merge.

    Returns:
        Path to the merged Parquet file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid_paths = [Path(p) for p in parquet_paths if Path(p).is_file() and Path(p).stat().st_size > 0]

    tmp_path = out_path.with_name(f"{out_path.name}.{uuid.uuid4().hex}.tmp")
    if not valid_paths:
        pq.write_table(pa.table({}), tmp_path)
        os.replace(tmp_path, out_path)
        return out_path

    schemas = [pq.read_schema(p) for p in valid_paths]
    try:
        unified_schema = pa.unify_schemas(schemas, promote_options="permissive")
    except TypeError:
        unified_schema = pa.unify_schemas(schemas)

    step = max(1, int(batch_size))
    try:
        with pq.ParquetWriter(tmp_path, unified_schema, compression="snappy") as writer:
            for start_idx in range(0, len(valid_paths), step):
                batch = valid_paths[start_idx : start_idx + step]
                for p in batch:
                    tbl = pq.read_table(p)
                    aligned = _align_table_to_schema(tbl, unified_schema)
                    writer.write_table(aligned)
                    del tbl, aligned
                gc.collect()
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    if delete_inputs:
        for p in valid_paths:
            if p != out_path and p.exists():
                p.unlink(missing_ok=True)

    return out_path


def append_to_zarr(
    ds: xr.Dataset,
    store_path: Union[str, Path],
    *,
    append_dim: str = "time",
    chunks: Optional[Dict[str, int]] = None,
) -> Path:
    """
    Write or append an xarray Dataset to a consolidated Zarr store.

    Args:
        ds: Dataset to write or append.
        store_path: Target Zarr store directory path.
        append_dim: Dimension along which to append when the store already exists.
        chunks: Optional chunking dictionary applied before writing.

    Returns:
        Path to the target Zarr store.
    """
    target = Path(store_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ds_to_write = ds.chunk(chunks) if chunks else ds

    if not target.exists() or not any(target.iterdir()):
        ds_to_write.to_zarr(target, mode="w", zarr_format=3)
    else:
        ds_to_write.to_zarr(target, mode="a", append_dim=append_dim)
    return target


__all__ = [
    "TAHMO_MAX_WORKERS_CAP",
    "GHCN_MAX_WORKERS_CAP",
    "MAX_WORKERS_CAP",
    "SOURCE_MAX_WORKERS_CAPS",
    "clamp_max_workers",
    "get_available_memory_mb",
    "allocate_worker_budget",
    "stream_merge_parquet",
    "append_to_zarr",
    "SOURCE_PROGRESS_PROFILES",
    "make_progress_bar",
    "RateLimiter",
    "retry_with_backoff",
    "ExtractionManifestSummary",
    "ExtractionCache",
    "ExtractionEstimate",
    "estimate_extraction",
]
