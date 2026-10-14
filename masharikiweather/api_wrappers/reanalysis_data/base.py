"""
Base Dataset Wrapper Architecture for MasharikiWeather
========================================================
Provides unified caching, rate limiting, HTTP retry handlers,
location normalization (Africa defaults + global coverage),
canonical variable aliasing, and output reshaping (wide / long).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("MasharikiWeather.Wrappers")

# Regional centroids and bounding boxes (Africa is first-class priority, global supported)
REGIONAL_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "kenya": (-1.286389, 36.817223),          # Nairobi
    "east_africa": (-0.5, 36.0),              # East Africa Great Rift
    "west_africa": (9.0820, 8.6753),          # Nigeria / Sahel
    "southern_africa": (-22.5609, 17.0658),   # Southern Africa
    "north_africa": (26.8206, 30.8025),       # North Africa / Egypt
    "central_africa": (3.8480, 11.5021),      # Central Africa / Congo Basin
    "africa": (1.6508, 20.0),                 # Pan-African Centroid
    "europe": (52.10, 5.18),                  # Central Europe (De Bilt, Netherlands)
    "netherlands": (52.1009, 5.1764),         # KNMI De Bilt
}

DEFAULT_CACHE_DIR = "./mw_cache"


def _safe_request(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 45,
    max_retries: int = 4,
    backoff_factor: float = 1.5,
) -> bytes:
    """Fetch URL with exponential backoff and timeout handling."""
    req_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 MasharikiWeather/1.0"}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, headers=req_headers)
    delay = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            if attempt == max_retries:
                logger.error(f"HTTP request failed after {max_retries} attempts: {url} -> {exc}")
                raise
            time.sleep(delay)
            delay *= backoff_factor

    raise RuntimeError(f"Failed to fetch {url}")


def _parse_date(val: Optional[Union[str, datetime]], default: str) -> str:
    """Parse date into 'YYYY-MM-DD' ISO string."""
    if val is None:
        return default
    if isinstance(val, (datetime, pd.Timestamp)):
        return val.strftime("%Y-%m-%d")
    s = str(val).strip()
    return pd.to_datetime(s).strftime("%Y-%m-%d")


def _normalize_locations(
    locations: Optional[Union[Dict[str, float], Tuple[float, float], List[Any], str]] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    region: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Standardize location inputs into a list of dictionaries:
    [{'name': 'loc_0', 'lat': 1.23, 'lon': 36.45}, ...]
    Defaults to Kenya/Africa if none provided.
    """
    targets: List[Dict[str, Any]] = []

    # Direct lat/lon pair
    if lat is not None and lon is not None:
        name = f"{float(lat):.3f}_{float(lon):.3f}"
        return [{"name": name, "lat": float(lat), "lon": float(lon)}]

    # Named region preset
    if region is not None:
        reg_clean = region.strip().lower().replace("-", "_").replace(" ", "_")
        if reg_clean in REGIONAL_CENTROIDS:
            c_lat, c_lon = REGIONAL_CENTROIDS[reg_clean]
            return [{"name": reg_clean, "lat": c_lat, "lon": c_lon}]

    if locations is None:
        # Default priority is Kenya
        c_lat, c_lon = REGIONAL_CENTROIDS["kenya"]
        return [{"name": "kenya_centroid", "lat": c_lat, "lon": c_lon}]

    if isinstance(locations, str):
        loc_clean = locations.strip().lower().replace("-", "_").replace(" ", "_")
        if loc_clean in REGIONAL_CENTROIDS:
            c_lat, c_lon = REGIONAL_CENTROIDS[loc_clean]
            return [{"name": loc_clean, "lat": c_lat, "lon": c_lon}]
        # Try splitting 'lat,lon'
        if "," in locations:
            parts = [float(p.strip()) for p in locations.split(",")]
            return [{"name": f"{parts[0]:.3f}_{parts[1]:.3f}", "lat": parts[0], "lon": parts[1]}]

    if isinstance(locations, tuple) and len(locations) == 2:
        return [{"name": f"{locations[0]:.3f}_{locations[1]:.3f}", "lat": float(locations[0]), "lon": float(locations[1])}]

    if isinstance(locations, dict):
        if "lat" in locations and "lon" in locations:
            name = str(locations.get("name", f"{locations['lat']:.3f}_{locations['lon']:.3f}"))
            return [{"name": name, "lat": float(locations["lat"]), "lon": float(locations["lon"])}]
        # Dictionary of name -> (lat, lon)
        for k, v in locations.items():
            if isinstance(v, (tuple, list)) and len(v) >= 2:
                targets.append({"name": str(k), "lat": float(v[0]), "lon": float(v[1])})
        return targets

    if isinstance(locations, list):
        for idx, item in enumerate(locations):
            if isinstance(item, dict) and "lat" in item and "lon" in item:
                name = str(item.get("name", f"loc_{idx}"))
                targets.append({"name": name, "lat": float(item["lat"]), "lon": float(item["lon"])})
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                name = str(item[2]) if len(item) >= 3 else f"loc_{idx}"
                targets.append({"name": name, "lat": float(item[0]), "lon": float(item[1])})

    if not targets:
        c_lat, c_lon = REGIONAL_CENTROIDS["kenya"]
        targets = [{"name": "kenya_centroid", "lat": c_lat, "lon": c_lon}]

    return targets


class _BaseDatasetWrapper:
    """
    Abstract base class providing common persistence, caching,
    concurrency, and output transformations across weather/solar wrappers.
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        cache_ttl_hours: float = 168.0,  # 7 days
        max_workers: int = 8,
        rate_limit_delay: float = 0.1,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = cache_ttl_hours
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay

    def _get_cache_path(self, prefix: str, params: Dict[str, Any]) -> Path:
        """Create a deterministic on-disk path for query caching."""
        h = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{prefix}_{h}.parquet"

    def _is_cache_valid(self, path: Path) -> bool:
        """Check if cached Parquet file exists and is within TTL."""
        if not path.exists():
            return False
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        return age_hours < self.cache_ttl_hours

    def _reshape_output(
        self,
        records: List[Dict[str, Any]],
        format: str = "wide",
        column_key: str = "location",
    ) -> pd.DataFrame:
        """
        Reshape a list of standardized observation dicts into:
        - 'wide': DatetimeIndex with MultiIndex columns (location, variable) or (variable, location)
        - 'long': Standard spatiotemporal table ['time', 'location', 'lat', 'lon', 'variable', 'value']
        """
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)

        if format.lower() == "long":
            cols_order = [c for c in ["time", "location", "lat", "lon", "variable", "value"] if c in df.columns]
            return df[cols_order].sort_values(by=["time", "location"] if "location" in df.columns else ["time"]).reset_index(drop=True)

        # Wide format pivot
        if "time" in df.columns and "variable" in df.columns and "value" in df.columns:
            pivot_idx = "time"
            if column_key in df.columns and df[column_key].nunique() > 1:
                pivot_cols = [column_key, "variable"]
            else:
                pivot_cols = ["variable"]
            wide = df.pivot_table(index=pivot_idx, columns=pivot_cols, values="value", aggfunc="first")
            wide.index.name = "time"
            return wide

        return df
