"""
GHCN Pipeline for MasharikiWeather (GHCN-Daily & GHCN-Hourly).

Unified API wrapper supporting:
- NOAA GHCN-Daily (GHCN-D) via multi-tiered AWS S3 Open Data & NOAA NCEI Access
- NOAA GHCN-Hourly (GHCNh) via NCEI global archives
- Dynamic African station & country discovery (East, West, Southern, North, Central Africa)
- High-performance caching, wide/long formatting, and lexicon unit standardization
- Automated physical quality control (QC) & outlier sanitization
- Flexible field querying with automatic variable derivations (e.g. t2m from TMAX/TMIN, RH from T/Td)
- Station coverage diagnostics and one-line convenience query functions
- Geostatistical cross-referencing and Haversine alignment (TAHMO, IMERG, ERA5)
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import io
import math
import os
import pathlib
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    pa = None
    pq = None
    HAS_PYARROW = False

try:
    import obstore as obs
    HAS_OBSTORE = True
except ImportError:
    obs = None
    HAS_OBSTORE = False


# =====================================================================
# 1. African Regional Definitions & Lexicons
# =====================================================================

DEFAULT_KENYA_BOUNDS: Dict[str, float] = {
    "min_lat": -4.9,
    "max_lat": 5.0,
    "min_lon": 33.5,
    "max_lon": 42.0,
}

# Regional presets mapping regions to NOAA 2-letter country prefixes
AFRICAN_REGIONS: Dict[str, List[str]] = {
    "kenya": ["KE"],
    "ke": ["KE"],
    "east_africa": ["KE", "TZ", "UG", "RW", "BY", "ET", "ER", "SO", "DJ", "SS"],
    "west_africa": [
        "NI", "GH", "SG", "IV", "ML", "UV", "NG", "GV", "PU",
        "BN", "TO", "LI", "SL", "GA", "CD"
    ],
    "southern_africa": [
        "SF", "ZI", "ZA", "MZ", "BC", "WA", "LT", "WZ", "MI", "AO"
    ],
    "north_africa": ["EG", "AG", "MO", "TS", "LY", "SU", "WI"],
    "central_africa": ["CG", "CF", "CM", "GB", "CT", "CD", "EK"],
}

# Extended continent and global presets (Africa is priority, but global queries supported)
CONTINENT_REGIONS: Dict[str, List[str]] = {
    "africa": list({c for cs in AFRICAN_REGIONS.values() for c in cs}),
    "all_africa": list({c for cs in AFRICAN_REGIONS.values() for c in cs}),
    "europe": [
        "NL", "GM", "FR", "UK", "IT", "SP", "BE", "SW", "SZ", "AU", "PL",
        "NO", "FI", "DA", "EI", "PO", "GR", "HU", "CZ", "LO", "RO", "BU",
        "HR", "SI", "BK", "EN", "LG", "LH", "MD", "AL", "IC", "LU", "MT"
    ],
    "north_america": [
        "US", "CA", "MX", "GL", "CU", "JM", "HA", "DR", "BF", "BH",
        "CS", "GT", "HO", "ES", "NU", "PM"
    ],
    "south_america": [
        "BR", "AR", "CI", "CO", "PE", "VE", "BL", "EC", "PA", "UY",
        "GY", "SR", "GF"
    ],
    "asia": [
        "CH", "IN", "JA", "KS", "KN", "ID", "TH", "VM", "RP", "MY",
        "PK", "BG", "SA", "AE", "TU", "IS", "IR", "IZ", "KU", "QA",
        "OM", "YE", "SY", "LE", "JO"
    ],
    "oceania": ["AS", "NZ", "FJ", "PP", "NC", "VA", "WS", "TO"],
}

GLOBAL_REGIONS: Dict[str, List[str]] = {
    **AFRICAN_REGIONS,
    **CONTINENT_REGIONS,
}


# Standard GHCN-Daily Lexicon mapping canonical aliases to (NOAA element, unit conversion)
GHCNDailyLexicon: Dict[str, Tuple[str, Callable[[Any], Any]]] = {
    # 1. Temperature metrics
    "t2m_max": ("TMAX", lambda x: x / 10.0),
    "tmax": ("TMAX", lambda x: x / 10.0),
    "max_temp": ("TMAX", lambda x: x / 10.0),
    "maximum_temperature": ("TMAX", lambda x: x / 10.0),
    "t2m_min": ("TMIN", lambda x: x / 10.0),
    "tmin": ("TMIN", lambda x: x / 10.0),
    "min_temp": ("TMIN", lambda x: x / 10.0),
    "minimum_temperature": ("TMIN", lambda x: x / 10.0),
    "t2m_avg": ("TAVG", lambda x: x / 10.0),
    "tavg": ("TAVG", lambda x: x / 10.0),
    "t2m": ("TAVG", lambda x: x / 10.0),
    "temperature": ("TAVG", lambda x: x / 10.0),
    "temp": ("TAVG", lambda x: x / 10.0),
    # Observation-time temperature (synoptic reading hour)
    "tobs": ("TOBS", lambda x: x / 10.0),
    "obs_temp": ("TOBS", lambda x: x / 10.0),
    "observation_temp": ("TOBS", lambda x: x / 10.0),

    # 2. Daily Rainfall & Multi-day Precipitation
    "tp": ("PRCP", lambda x: x / 10.0),
    "prcp": ("PRCP", lambda x: x / 10.0),
    "precipitation": ("PRCP", lambda x: x / 10.0),
    "rain": ("PRCP", lambda x: x / 10.0),
    "rainfall": ("PRCP", lambda x: x / 10.0),
    # Multi-day cumulative rainfall total (critical for rural African manual stations)
    "mdpr": ("MDPR", lambda x: x / 10.0),
    "multi_day_prcp": ("MDPR", lambda x: x / 10.0),
    "multi_day_rain": ("MDPR", lambda x: x / 10.0),
    "multiday_rain": ("MDPR", lambda x: x / 10.0),
    # Number of days included in multi-day precipitation total
    "dapr": ("DAPR", lambda x: x),
    "multi_day_days": ("DAPR", lambda x: x),
    "multiday_days": ("DAPR", lambda x: x),

    # 3. Evaporation & Agricultural Water Balance
    "evap": ("EVAP", lambda x: x / 10.0),
    "pan_evap": ("EVAP", lambda x: x / 10.0),
    "pan_evaporation": ("EVAP", lambda x: x / 10.0),

    # 4. Wind Speed & Severe Convective Wind Gusts
    "ws10m": ("AWND", lambda x: x / 10.0),
    "awnd": ("AWND", lambda x: x / 10.0),
    "wind_speed": ("AWND", lambda x: x / 10.0),
    "wind": ("AWND", lambda x: x / 10.0),
    "wsfg": ("WSFG", lambda x: x / 10.0),
    "wind_gust": ("WSFG", lambda x: x / 10.0),
    "gust": ("WSFG", lambda x: x / 10.0),
    "peak_gust": ("WSFG", lambda x: x / 10.0),

    # 5. Tropical Convection & Thunderstorm Activity (Congo/Lake Victoria Basin)
    "wt03": ("WT03", lambda x: 1.0 if pd.notna(x) else 0.0),
    "thunder": ("WT03", lambda x: 1.0 if pd.notna(x) else 0.0),
    "thunderstorm": ("WT03", lambda x: 1.0 if pd.notna(x) else 0.0),

    # 6. Atmospheric Dust, Smoke & Harmattan Haze (Sahel / West & Southern Africa)
    "wt08": ("WT08", lambda x: 1.0 if pd.notna(x) else 0.0),
    "dust_haze": ("WT08", lambda x: 1.0 if pd.notna(x) else 0.0),
    "harmattan_haze": ("WT08", lambda x: 1.0 if pd.notna(x) else 0.0),
    "haze": ("WT08", lambda x: 1.0 if pd.notna(x) else 0.0),
    "smoke": ("WT08", lambda x: 1.0 if pd.notna(x) else 0.0),

    # 7. Derived Biometeorology: Diurnal Temperature Range (Tmax - Tmin)
    "dtr": ("DTR", lambda x: x),
    "diurnal_range": ("DTR", lambda x: x),
    "diurnal_temperature_range": ("DTR", lambda x: x),

    # 8. High-Altitude Cold / Mountain Terrain (Atlas, Drakensberg, Mt Kenya, Kilimanjaro)
    "snow": ("SNOW", lambda x: x),
    "snowfall": ("SNOW", lambda x: x),
    "snwd": ("SNWD", lambda x: x),
    "snow_depth": ("SNWD", lambda x: x),
}

# Standard GHCN-Hourly Lexicon mapping aliases to (field name, unit conversion)
GHCNHourlyLexicon: Dict[str, Tuple[str, Callable[[Any], Any]]] = {
    "t2m": ("temperature", lambda x: x),
    "temperature": ("temperature", lambda x: x),
    "temp": ("temperature", lambda x: x),
    "t": ("temperature", lambda x: x),
    "d2m": ("dew_point_temperature", lambda x: x),
    "dew_point_temperature": ("dew_point_temperature", lambda x: x),
    "dew_point": ("dew_point_temperature", lambda x: x),
    "dewpoint": ("dew_point_temperature", lambda x: x),
    "td": ("dew_point_temperature", lambda x: x),
    "ws10m": ("wind_speed", lambda x: x),
    "wind_speed": ("wind_speed", lambda x: x),
    "wind": ("wind_speed", lambda x: x),
    "ws": ("wind_speed", lambda x: x),
    "wind_direction": ("wind_direction", lambda x: x),
    "direction": ("wind_direction", lambda x: x),
    "wd": ("wind_direction", lambda x: x),
    "wind_gust": ("wind_gust", lambda x: x),
    "gust": ("wind_gust", lambda x: x),
    "tp": ("precipitation", lambda x: x),
    "precipitation": ("precipitation", lambda x: x),
    "prcp": ("precipitation", lambda x: x),
    "precip": ("precipitation", lambda x: x),
    "rain": ("precipitation", lambda x: x),
    "rh": ("relative_humidity", lambda x: x),  # Relative humidity (derived)
    "relative_humidity": ("relative_humidity", lambda x: x),
    "u10m": ("u10m", lambda x: x),
    "v10m": ("v10m", lambda x: x),
}


def resolve_daily_variable(var_name: str) -> Tuple[str, Callable[[Any], Any], str]:
    """
    Resolve any user field name (case-insensitive) to its NOAA element, unit conversion, and canonical name.
    Returns (noaa_element, unit_converter, canonical_var_name).
    """
    clean = str(var_name).strip().lower().replace("-", "_").replace(" ", "_")
    if clean in GHCNDailyLexicon:
        elem, conv = GHCNDailyLexicon[clean]
        return elem, conv, clean
    # Check uppercase NOAA code (e.g. 'TMAX', 'PRCP', 'MDPR', 'EVAP')
    upper = str(var_name).strip().upper()
    tenth_div = ["TMAX", "TMIN", "TAVG", "AWND", "PRCP", "MDPR", "EVAP", "TOBS", "WSFG"]
    if upper in tenth_div:
        return upper, (lambda x: x / 10.0), upper.lower()
    if upper in ["DAPR", "SNOW", "SNWD", "WT03", "WT08"]:
        return upper, (lambda x: x), upper.lower()
    return upper, (lambda x: x), clean


def resolve_hourly_variable(var_name: str) -> Tuple[str, Callable[[Any], Any], str]:
    """
    Resolve any user field name (case-insensitive) to its GHCNh column and canonical name.
    """
    clean = str(var_name).strip().lower().replace("-", "_").replace(" ", "_")
    if clean in GHCNHourlyLexicon:
        col, conv = GHCNHourlyLexicon[clean]
        return col, conv, clean
    return clean, (lambda x: x), clean


def calculate_relative_humidity(temperature: pd.Series, dew_point: pd.Series) -> pd.Series:
    """
    Compute relative humidity (%) from air temperature and dew point in °C
    using the standard Magnus formula (WMO CIMO Guide).
    """
    t = pd.to_numeric(temperature, errors="coerce")
    td = pd.to_numeric(dew_point, errors="coerce")
    # Valid physical computation
    a = 17.625
    b = 243.04
    alpha = (a * t) / (b + t)
    beta = (a * td) / (b + td)
    rh = 100.0 * np.exp(beta - alpha)
    return pd.Series(np.clip(rh, 0.0, 100.0), index=temperature.index)


# =====================================================================
# 2. Quality Control & Physical Plausibility Sanitization
# =====================================================================

def clean_daily_dataframe(df: pd.DataFrame, apply_qc: bool = True) -> pd.DataFrame:
    """
    Apply meteorological plausibility checks to daily data:
    - Filters invalid missing sentinels (-9999, 9999)
    - Valid temperature range: -25°C to 55°C (TMAX, TMIN, TAVG, TOBS)
    - Valid daily precipitation: 0.0 to 500.0 mm/day (PRCP)
    - Valid multi-day precipitation: 0.0 to 1500.0 mm (MDPR)
    - Valid multi-day duration: 1 to 365 days (DAPR)
    - Valid wind speed & gusts: 0.0 to 80.0 m/s (AWND, WSFG)
    - Valid pan evaporation: 0.0 to 50.0 mm/day (EVAP)
    - Valid diurnal temperature range: 0.0 to 40.0 °C (DTR)
    """
    if not apply_qc or df.empty or "observation" not in df.columns:
        return df

    clean = df.copy()
    obs = pd.to_numeric(clean["observation"], errors="coerce")
    var = clean["variable"].astype(str).str.lower() if "variable" in clean.columns else pd.Series("", index=clean.index)

    # Discard sentinel missing values
    invalid_sentinel = obs.isin([-9999, 9999, -999.9, 999.9, -99.9, 99.9])
    clean.loc[invalid_sentinel, "observation"] = np.nan

    # Temperature bounds (TMAX, TMIN, TAVG, TOBS)
    is_temp = var.str.contains("t2m|tmax|tmin|tavg|temp|tobs", regex=True)
    clean.loc[is_temp & ((obs < -25.0) | (obs > 55.0)), "observation"] = np.nan

    # Precipitation bounds (PRCP, MDPR)
    is_mdpr = var.str.contains("mdpr|multi_day_prcp|multi_day_rain", regex=True)
    clean.loc[is_mdpr & ((obs < 0.0) | (obs > 1500.0)), "observation"] = np.nan

    is_prcp = var.str.contains("tp|prcp|rain|precip", regex=True) & ~is_mdpr
    clean.loc[is_prcp & ((obs < 0.0) | (obs > 500.0)), "observation"] = np.nan

    # Multi-day days duration bounds (DAPR)
    is_dapr = var.str.contains("dapr|multi_day_days", regex=True)
    clean.loc[is_dapr & ((obs < 1.0) | (obs > 365.0)), "observation"] = np.nan

    # Wind speed & wind gust bounds (AWND, WSFG)
    is_wind = var.str.contains("ws|wind|awnd|gust", regex=True)
    clean.loc[is_wind & ((obs < 0.0) | (obs > 80.0)), "observation"] = np.nan

    # Evaporation bounds (EVAP)
    is_evap = var.str.contains("evap", regex=True)
    clean.loc[is_evap & ((obs < 0.0) | (obs > 50.0)), "observation"] = np.nan

    # Diurnal temperature range bounds (DTR)
    is_dtr = var.str.contains("dtr|diurnal", regex=True)
    clean.loc[is_dtr & ((obs < 0.0) | (obs > 40.0)), "observation"] = np.nan

    return clean.dropna(subset=["observation"])


def clean_hourly_dataframe(df: pd.DataFrame, apply_qc: bool = True) -> pd.DataFrame:
    """
    Apply meteorological plausibility checks to hourly telemetry:
    - Eliminates telecommunication artifacts (e.g. -18.3°C in Nairobi or 63 m/s wind spikes)
    - Temperature range: -15°C to 55°C
    - Dew point range: -25°C to 35°C
    - Consistency: Dew point cannot exceed air temperature by > 0.5°C
    - Wind speed: 0.0 to 45.0 m/s (162 km/h)
    - Wind direction: 0° to 360°, removes sentinel 999
    - Precipitation: 0.0 to 250.0 mm/hr
    """
    if not apply_qc or df.empty:
        return df

    clean = df.copy()

    # Temperature plausibility
    if "temperature" in clean.columns:
        t = pd.to_numeric(clean["temperature"], errors="coerce")
        clean.loc[(t < -15.0) | (t > 55.0) | t.isin([-9999, 9999, -99.9, 99.9]), "temperature"] = np.nan

    # Dew point plausibility
    if "dew_point_temperature" in clean.columns:
        td = pd.to_numeric(clean["dew_point_temperature"], errors="coerce")
        clean.loc[(td < -25.0) | (td > 35.0) | td.isin([-9999, 9999, -99.9, 99.9]), "dew_point_temperature"] = np.nan

        # Physical consistency: dew point cannot exceed dry-bulb temperature significantly
        if "temperature" in clean.columns:
            t = pd.to_numeric(clean["temperature"], errors="coerce")
            inconsistent = (td > (t + 0.5)) & t.notna() & td.notna()
            # If temp is suspiciously negative while dew point is tropical, flag temperature as corrupted (METAR sign flip)
            corrupted_temp = (t < 0.0) & (td > 5.0)
            clean.loc[corrupted_temp, "temperature"] = np.nan
            clean.loc[inconsistent & (~corrupted_temp), "dew_point_temperature"] = np.nan

    # Wind speed plausibility
    if "wind_speed" in clean.columns:
        ws = pd.to_numeric(clean["wind_speed"], errors="coerce")
        clean.loc[(ws < 0.0) | (ws > 45.0) | ws.isin([-9999, 9999, 99.9]), "wind_speed"] = np.nan

    # Wind direction plausibility
    if "wind_direction" in clean.columns:
        wd = pd.to_numeric(clean["wind_direction"], errors="coerce")
        clean.loc[(wd < 0.0) | (wd > 360.0) | (wd == 999), "wind_direction"] = np.nan

    # Precipitation plausibility
    if "precipitation" in clean.columns:
        p = pd.to_numeric(clean["precipitation"], errors="coerce")
        clean.loc[(p < 0.0) | (p > 250.0) | p.isin([-9999, 9999]), "precipitation"] = np.nan

    return clean


# =====================================================================
# 3. Caching & S3 Listing Helpers
# =====================================================================

_MUTABLE_CACHE_MAX_AGE_S: float = 24 * 60 * 60  # 24 hours


def _is_cache_fresh(path: str, max_age_s: Optional[float] = None) -> bool:
    if not os.path.isfile(path):
        return False
    if max_age_s is None:
        return True
    return (datetime.now().timestamp() - os.path.getmtime(path)) < max_age_s


def _list_s3_prefix_keys(bucket: str, prefix: str) -> List[str]:
    """List S3 keys using AWS public XML REST API without requiring boto3."""
    url = f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}"
    keys = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MasharikiWeather/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            root = ET.fromstring(resp.read())
            ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            for item in root.findall("s3:Contents", ns):
                key = item.find("s3:Key", ns)
                if key is not None and key.text and key.text.endswith(".parquet"):
                    keys.append(key.text)
    except Exception:
        pass
    return keys


def _extract_q_flag(val: Any) -> str:
    """Safely extract quality flag from NOAA attribute string (e.g. ',,S,' -> quality flag is index 1)."""
    if val is None or pd.isna(val):
        return ""
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ("nan", "none"):
        return ""
    parts = val_str.split(",")
    return parts[1].strip() if len(parts) > 1 else ""


# =====================================================================
# 4. Base Pipeline Architecture
# =====================================================================

class _GHCNBase:
    DEFAULT_KENYA_BOUNDS: Dict[str, float] = DEFAULT_KENYA_BOUNDS

    def __init__(
        self,
        stations: Optional[List[str]] = None,
        region: Optional[str] = None,
        countries: Optional[Union[str, List[str]]] = None,
        country_codes: Optional[Union[str, List[str]]] = None,
        bounds: Optional[Union[Dict[str, float], Tuple[float, float, float, float]]] = None,
        cache_dir: str = "./ghcn_cache",
        verbose: bool = True,
        async_workers: int = 16,
        fetch_mode: str = "auto",
        apply_qc: bool = True,
        min_year: Optional[int] = None,
        active_only: bool = False,
    ):
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.async_workers = async_workers
        self.fetch_mode = fetch_mode
        self.apply_qc = apply_qc
        os.makedirs(self.cache_dir, exist_ok=True)

        selected_stations: Set[str] = set(stations or [])

        # Global station discovery (Africa-first default, with global country/continent support)
        if region or countries or country_codes:
            disc_func = getattr(self, "get_stations", self.get_african_stations)
            disc_df = disc_func(
                region=region,
                countries=countries,
                country_codes=country_codes,
                bounds=bounds,
                cache_dir=cache_dir,
                min_year=min_year,
                active_only=active_only,
            )
            id_col = "ID" if "ID" in disc_df.columns else "Station_ID"
            selected_stations.update(disc_df[id_col].dropna().tolist())
        elif bounds is not None:
            bbox_stations = self.get_stations_bbox(bounds, cache_dir=cache_dir)
            selected_stations.update(bbox_stations)

        self.stations = sorted(list(selected_stations))

    def _run_async(self, coro):
        """Execute async tasks safely across scripts and Jupyter/Colab notebooks."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    def get_location_mappings(self) -> Dict[str, str]:
        """
        Generate station ID to 'lat_lon' string mapping dictionary
        matching regional modeling convention (e.g. 'KE000063612': '3.117_35.617').
        """
        df = self.get_stations_dataframe()
        return dict(zip(df["station"], df["lat_lon_label"]))

    def get_stations_dataframe(self, include_inventory: bool = True) -> pd.DataFrame:
        """
        Return DataFrame of configured stations with lat, lon, elevation, name, lat_lon_label,
        and observation lifecycle coverage (start/end years, status, element availability).
        """
        meta = self.get_station_metadata(self.cache_dir)
        id_col = "ID" if "ID" in meta.columns else ("Station_ID" if "Station_ID" in meta.columns else "GHCN_ID")
        lat_col = "LAT" if "LAT" in meta.columns else ("Latitude" if "Latitude" in meta.columns else "LATITUDE")
        lon_col = "LON" if "LON" in meta.columns else ("Longitude" if "Longitude" in meta.columns else "LONGITUDE")
        elev_col = "ELEV" if "ELEV" in meta.columns else ("Elevation" if "Elevation" in meta.columns else "ELEVATION")
        name_col = "NAME" if "NAME" in meta.columns else ("Station_name" if "Station_name" in meta.columns else "NAME")

        meta_sub = meta[meta[id_col].isin(set(self.stations))].copy()
        meta_sub["latitude"] = pd.to_numeric(meta_sub[lat_col], errors="coerce")
        meta_sub["longitude"] = pd.to_numeric(meta_sub[lon_col], errors="coerce")
        meta_sub["elevation"] = pd.to_numeric(meta_sub[elev_col], errors="coerce") if elev_col in meta_sub.columns else np.nan
        meta_sub["name"] = meta_sub[name_col] if name_col in meta_sub.columns else ""
        meta_sub = meta_sub.dropna(subset=["latitude", "longitude"])

        meta_sub["station"] = meta_sub[id_col].astype(str)
        meta_sub["lat_lon_label"] = (
            np.round(meta_sub["latitude"], 3).astype(str)
            + "_"
            + np.round(meta_sub["longitude"], 3).astype(str)
        )

        # Merge rich observation inventory to expose data start/end dates
        if include_inventory and hasattr(self, "get_station_inventory"):
            try:
                inv_df = self.get_station_inventory(cache_dir=self.cache_dir)
                meta_sub = meta_sub.merge(inv_df, left_on="station", right_on="ID", how="left")
            except Exception:
                pass

        cols = [
            "station", "name", "latitude", "longitude", "elevation", "lat_lon_label",
            "data_start_year", "data_end_year", "data_coverage", "status",
            "has_precipitation", "prcp_coverage", "has_temperature", "temp_coverage",
            "elements_available"
        ]
        available_cols = [c for c in cols if c in meta_sub.columns]
        return meta_sub[available_cols].reset_index(drop=True)

    def align(
        self,
        target_meta: pd.DataFrame,
        target_name: str = "tahmo",
        metric: str = "haversine",
    ) -> pd.DataFrame:
        """
        Align the pipeline's configured stations to an external reference grid or station network.
        """
        source_df = self.get_stations_dataframe()
        return align_to_reference_grid(
            source_df=source_df,
            target_meta=target_meta,
            target_name=target_name,
            metric=metric,
        )


# =====================================================================
# 5. GHCN-Daily Engine
# =====================================================================

class GHCNDaily(_GHCNBase):
    S3_BUCKET: str = "noaa-ghcn-pds"
    HTTP_BASE_URL: str = "https://noaa-ghcn-pds.s3.amazonaws.com"
    NCEI_ACCESS_URL: str = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access"
    COUNTRY_CODES_URL: str = "https://noaa-ghcn-pds.s3.amazonaws.com/ghcnd-countries.txt"
    _BY_STATION_THRESHOLD = 200

    @classmethod
    def get_station_metadata(cls, cache_dir: str = "./ghcn_cache") -> pd.DataFrame:
        """Load and cache official NOAA GHCN-D station list."""
        os.makedirs(cache_dir, exist_ok=True)
        meta_path = os.path.join(cache_dir, "ghcnd-stations.txt")

        if not os.path.exists(meta_path) or os.path.getsize(meta_path) == 0:
            url = f"{cls.HTTP_BASE_URL}/ghcnd-stations.txt"
            req = urllib.request.Request(url, headers={"User-Agent": "MasharikiWeather/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(meta_path, "wb") as f:
                    f.write(resp.read())

        df = pd.read_fwf(
            meta_path,
            widths=[11, 9, 10, 7, 3, 31, 4, 4, 6],
            names=["ID", "LAT", "LON", "ELEV", "STATE", "NAME", "GSN", "HCN", "WMO"],
        )
        df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
        df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
        df["ELEV"] = pd.to_numeric(df["ELEV"], errors="coerce")
        return df

    @classmethod
    def get_country_metadata(cls, cache_dir: str = "./ghcn_cache") -> pd.DataFrame:
        """Load and parse official NOAA country code definitions from ghcnd-countries.txt."""
        os.makedirs(cache_dir, exist_ok=True)
        c_path = os.path.join(cache_dir, "ghcnd-countries.txt")

        if not os.path.exists(c_path) or os.path.getsize(c_path) == 0:
            urls = [
                cls.COUNTRY_CODES_URL,
                "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-countries.txt",
            ]
            for url in urls:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "MasharikiWeather/1.0"})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        with open(c_path, "wb") as f:
                            f.write(resp.read())
                    break
                except Exception:
                    continue

        countries = []
        if os.path.exists(c_path):
            with open(c_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if len(line.strip()) >= 3:
                        code = line[:2].strip()
                        name = line[3:].strip()
                        if code and name:
                            countries.append({"country_code": code, "country_name": name})
        return pd.DataFrame(countries)

    @classmethod
    def get_stations_bbox(
        cls,
        bounds: Union[Dict[str, float], Tuple[float, float, float, float]],
        cache_dir: str = "./ghcn_cache",
    ) -> List[str]:
        if isinstance(bounds, dict):
            min_lat, max_lat = bounds["min_lat"], bounds["max_lat"]
            min_lon, max_lon = bounds["min_lon"], bounds["max_lon"]
        else:
            min_lat, min_lon, max_lat, max_lon = bounds

        meta = cls.get_station_metadata(cache_dir)
        sub = meta[
            (meta["LAT"] >= min_lat)
            & (meta["LAT"] <= max_lat)
            & (meta["LON"] >= min_lon)
            & (meta["LON"] <= max_lon)
        ]
        return sub["ID"].tolist()

    @classmethod
    def get_station_inventory(cls, cache_dir: str = "./ghcn_cache") -> pd.DataFrame:
        """
        Load official NOAA GHCN-Daily station inventory containing observation date ranges,
        element availability, and operational lifecycle windows.
        Caches aggregated inventory locally in parquet format for instant subsequent lookups.
        """
        os.makedirs(cache_dir, exist_ok=True)
        summary_parquet = os.path.join(cache_dir, "ghcnd_station_inventory.parquet")
        if os.path.exists(summary_parquet):
            try:
                df_p = pd.read_parquet(summary_parquet)
                # Ensure alias compatibility
                if "first_year" not in df_p.columns and "data_start_year" in df_p.columns:
                    df_p["first_year"] = df_p["data_start_year"]
                if "last_year" not in df_p.columns and "data_end_year" in df_p.columns:
                    df_p["last_year"] = df_p["data_end_year"]
                if "elements" not in df_p.columns and "elements_available" in df_p.columns:
                    df_p["elements"] = df_p["elements_available"]
                return df_p
            except Exception:
                pass

        inv_txt = os.path.join(cache_dir, "ghcnd-inventory.txt")
        if not os.path.exists(inv_txt) or os.path.getsize(inv_txt) == 0:
            url = f"{cls.HTTP_BASE_URL}/ghcnd-inventory.txt"
            req = urllib.request.Request(url, headers={"User-Agent": "MasharikiWeather/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(inv_txt, "wb") as f:
                    f.write(resp.read())

        colspecs = [(0, 11), (12, 20), (21, 30), (31, 35), (36, 40), (41, 45)]
        names = ["ID", "LAT", "LON", "ELEMENT", "FIRST_YEAR", "LAST_YEAR"]
        df = pd.read_fwf(inv_txt, colspecs=colspecs, names=names, dtype={"FIRST_YEAR": int, "LAST_YEAR": int})

        # Overall observation range and element list
        agg_overall = df.groupby("ID").agg(
            data_start_year=("FIRST_YEAR", "min"),
            data_end_year=("LAST_YEAR", "max"),
            elements_available=("ELEMENT", lambda x: ", ".join(sorted(x.unique()))),
        ).reset_index()

        # PRCP (precipitation) lifecycle range
        prcp_df = df[df["ELEMENT"] == "PRCP"].groupby("ID").agg(
            prcp_start_year=("FIRST_YEAR", "min"),
            prcp_end_year=("LAST_YEAR", "max"),
        ).reset_index()
        prcp_df["has_precipitation"] = True

        # TMAX (temperature) lifecycle range
        tmax_df = df[df["ELEMENT"] == "TMAX"].groupby("ID").agg(
            tmax_start_year=("FIRST_YEAR", "min"),
            tmax_end_year=("LAST_YEAR", "max"),
        ).reset_index()
        tmax_df["has_temperature"] = True

        st_summary = agg_overall.merge(prcp_df, on="ID", how="left").merge(tmax_df, on="ID", how="left")
        st_summary["has_precipitation"] = st_summary["has_precipitation"].fillna(False)
        st_summary["has_temperature"] = st_summary["has_temperature"].fillna(False)

        st_summary["data_coverage"] = st_summary["data_start_year"].astype(str) + " - " + st_summary["data_end_year"].astype(str)
        st_summary["status"] = "Historical (ended " + st_summary["data_end_year"].astype(str) + ")"
        st_summary.loc[st_summary["data_end_year"] >= 2024, "status"] = "Active"

        st_summary["prcp_coverage"] = st_summary.apply(
            lambda r: f"{int(r['prcp_start_year'])} - {int(r['prcp_end_year'])}" if r["has_precipitation"] else "None", axis=1
        )
        st_summary["temp_coverage"] = st_summary.apply(
            lambda r: f"{int(r['tmax_start_year'])} - {int(r['tmax_end_year'])}" if r["has_temperature"] else "None", axis=1
        )

        st_summary["first_year"] = st_summary["data_start_year"]
        st_summary["last_year"] = st_summary["data_end_year"]
        st_summary["elements"] = st_summary["elements_available"]

        try:
            st_summary.to_parquet(summary_parquet, index=False)
        except Exception:
            pass
        return st_summary

    @classmethod
    def get_variable_inventory(cls, cache_dir: str = "./ghcn_cache") -> pd.DataFrame:
        """
        Load per-station, per-variable NOAA GHCN-Daily inventory.
        Tracks each meteorological variable independently with its start year, end year,
        coverage window, and individual active status.
        """
        os.makedirs(cache_dir, exist_ok=True)
        var_parquet = os.path.join(cache_dir, "ghcnd_variable_inventory.parquet")
        if os.path.exists(var_parquet):
            try:
                return pd.read_parquet(var_parquet)
            except Exception:
                pass

        inv_txt = os.path.join(cache_dir, "ghcnd-inventory.txt")
        if not os.path.exists(inv_txt) or os.path.getsize(inv_txt) == 0:
            url = f"{cls.HTTP_BASE_URL}/ghcnd-inventory.txt"
            req = urllib.request.Request(url, headers={"User-Agent": "MasharikiWeather/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(inv_txt, "wb") as f:
                    f.write(resp.read())

        colspecs = [(0, 11), (12, 20), (21, 30), (31, 35), (36, 40), (41, 45)]
        names = ["ID", "LAT", "LON", "ELEMENT", "FIRST_YEAR", "LAST_YEAR"]
        df = pd.read_fwf(inv_txt, colspecs=colspecs, names=names, dtype={"FIRST_YEAR": int, "LAST_YEAR": int})

        elem_map = {
            "PRCP": "rain",
            "TMAX": "tmax",
            "TMIN": "tmin",
            "TAVG": "tavg",
            "AWND": "wind",
            "MDPR": "multi_day_prcp",
            "DAPR": "multi_day_days",
            "EVAP": "pan_evaporation",
            "TOBS": "obs_temp",
            "WSFG": "wind_gust",
            "WT03": "thunder",
            "WT08": "dust_haze",
            "SNWD": "snow_depth",
            "SNOW": "snowfall",
        }

        df_var = df[df["ELEMENT"].isin(elem_map.keys())].copy()
        df_var["variable"] = df_var["ELEMENT"].map(lambda x: elem_map.get(x, x.lower()))
        df_var["is_active"] = df_var["LAST_YEAR"] >= 2024
        df_var["coverage_window"] = df_var["FIRST_YEAR"].astype(str) + " - " + df_var["LAST_YEAR"].astype(str)
        df_var["status"] = df_var.apply(
            lambda r: f"Active (thru {r['LAST_YEAR']})" if r["is_active"] else f"Ended {r['LAST_YEAR']}", axis=1
        )
        try:
            df_var.to_parquet(var_parquet, index=False)
        except Exception:
            pass
        return df_var

    @classmethod
    def get_stations_variable_activeness(
        cls,
        region: Optional[Union[str, List[str]]] = None,
        countries: Optional[Union[str, List[str]]] = None,
        stations: Optional[List[str]] = None,
        cache_dir: str = "./ghcn_cache",
    ) -> pd.DataFrame:
        """
        Produce a comprehensive station table with independent activeness columns for each variable.
        Evaluates rainfall, max temperature, min temperature, and average temperature independently.
        """
        if isinstance(region, (list, tuple, set)):
            stations = list(region)
            region = None

        st_meta = cls.get_african_stations(region=region, countries=countries, cache_dir=cache_dir, include_inventory=False)
        if stations:
            st_meta = st_meta[st_meta["ID"].isin(set(stations))].copy()

        var_inv = cls.get_variable_inventory(cache_dir=cache_dir)
        sub_var = var_inv[var_inv["ID"].isin(set(st_meta["ID"]))]

        rows = []
        for _, srow in st_meta.iterrows():
            st = srow["ID"]
            s_vars = sub_var[sub_var["ID"] == st]

            r_rain = s_vars[s_vars["variable"] == "rain"]
            r_tmax = s_vars[s_vars["variable"] == "tmax"]
            r_tmin = s_vars[s_vars["variable"] == "tmin"]
            r_tavg = s_vars[s_vars["variable"] == "tavg"]
            r_wind = s_vars[s_vars["variable"] == "wind"]
            r_mdpr = s_vars[s_vars["variable"] == "multi_day_prcp"]
            r_evap = s_vars[s_vars["variable"] == "pan_evaporation"]
            r_tobs = s_vars[s_vars["variable"] == "obs_temp"]
            r_gust = s_vars[s_vars["variable"] == "wind_gust"]
            r_thun = s_vars[s_vars["variable"] == "thunder"]
            r_dust = s_vars[s_vars["variable"] == "dust_haze"]

            act_vars = s_vars[s_vars["is_active"]]["variable"].tolist()
            ended_vars = [f"{r['variable']} ({r['LAST_YEAR']})" for _, r in s_vars[~s_vars["is_active"]].iterrows()]

            rows.append({
                "ID": st,
                "station": st,
                "name": srow.get("NAME", ""),
                "country": srow.get("country_name", ""),
                "latitude": srow.get("LAT"),
                "longitude": srow.get("LON"),
                "elevation": srow.get("ELEV"),
                "rain_status": r_rain["status"].iloc[0] if not r_rain.empty else "Not Recorded",
                "rain_window": r_rain["coverage_window"].iloc[0] if not r_rain.empty else "None",
                "tmax_status": r_tmax["status"].iloc[0] if not r_tmax.empty else "Not Recorded",
                "tmax_window": r_tmax["coverage_window"].iloc[0] if not r_tmax.empty else "None",
                "tmin_status": r_tmin["status"].iloc[0] if not r_tmin.empty else "Not Recorded",
                "tmin_window": r_tmin["coverage_window"].iloc[0] if not r_tmin.empty else "None",
                "tavg_status": r_tavg["status"].iloc[0] if not r_tavg.empty else "Not Recorded",
                "wind_status": r_wind["status"].iloc[0] if not r_wind.empty else "Not Recorded",
                "wind_window": r_wind["coverage_window"].iloc[0] if not r_wind.empty else "None",
                "mdpr_status": r_mdpr["status"].iloc[0] if not r_mdpr.empty else "Not Recorded",
                "mdpr_window": r_mdpr["coverage_window"].iloc[0] if not r_mdpr.empty else "None",
                "evap_status": r_evap["status"].iloc[0] if not r_evap.empty else "Not Recorded",
                "evap_window": r_evap["coverage_window"].iloc[0] if not r_evap.empty else "None",
                "obs_temp_status": r_tobs["status"].iloc[0] if not r_tobs.empty else "Not Recorded",
                "gust_status": r_gust["status"].iloc[0] if not r_gust.empty else "Not Recorded",
                "active_variables": ", ".join(act_vars) if act_vars else "None",
                "ended_variables": ", ".join(ended_vars) if ended_vars else "None",
            })
        return pd.DataFrame(rows)

    @classmethod
    def get_stations(
        cls,
        region: Optional[str] = None,
        countries: Optional[Union[str, List[str]]] = None,
        country_codes: Optional[Union[str, List[str]]] = None,
        bounds: Optional[Union[Dict[str, float], Tuple[float, float, float, float]]] = None,
        cache_dir: str = "./ghcn_cache",
        include_inventory: bool = True,
        min_year: Optional[int] = None,
        active_only: bool = False,
    ) -> pd.DataFrame:
        """
        Dynamically discover and filter GHCN stations globally or by region.
        Supports any country worldwide (e.g. 'Netherlands', 'Germany', 'Kenya', 'Japan'),
        continent presets ('europe', 'north_america', 'south_america', 'asia', 'oceania', 'africa'),
        or African regional presets ('kenya', 'east_africa', 'west_africa', etc.).

        Parameters
        ----------
        region : str, optional
            Region preset (e.g. 'kenya', 'east_africa', 'africa', 'europe', 'asia', 'north_america').
        countries : str or list of str, optional
            Country names (e.g. 'Netherlands', 'Kenya', ['Germany', 'France']).
        country_codes : str or list of str, optional
            NOAA 2-letter country prefixes (e.g. 'NL', 'GM', 'KE').
        bounds : dict or tuple, optional
            Bounding box coordinates.
        cache_dir : str, default './ghcn_cache'
            Cache directory for metadata and inventories.
        include_inventory : bool, default True
            If True, merges start/end years, status, and observation elements from NOAA inventory.
        min_year : int, optional
            Only return stations active in or after this year (e.g. 2020).
        active_only : bool, default False
            If True, filters to stations with observations in 2020 or later.
        """
        meta = cls.get_station_metadata(cache_dir=cache_dir).copy()
        country_meta = cls.get_country_metadata(cache_dir=cache_dir)

        meta["country_code"] = meta["ID"].str[:2]
        meta = meta.merge(country_meta, on="country_code", how="left")

        target_codes: Set[str] = set()

        if region:
            reg_clean = region.strip().lower().replace("-", "_").replace(" ", "_")
            if reg_clean in GLOBAL_REGIONS:
                target_codes.update(GLOBAL_REGIONS[reg_clean])
            else:
                available = list(GLOBAL_REGIONS.keys())
                raise ValueError(f"Unknown region '{region}'. Available presets: {available}")

        if countries:
            if isinstance(countries, str):
                countries = [countries]
            for c in countries:
                c_clean = c.strip()
                if len(c_clean) == 2:
                    target_codes.add(c_clean.upper())
                else:
                    c_low = c_clean.lower()
                    matched = country_meta[
                        country_meta["country_name"].str.lower().str.contains(c_low, regex=False)
                    ]
                    if not matched.empty:
                        target_codes.update(matched["country_code"].tolist())
                    else:
                        target_codes.add(c_clean.upper())

        if country_codes:
            if isinstance(country_codes, str):
                country_codes = [country_codes]
            target_codes.update([cc.upper() for cc in country_codes])

        if target_codes:
            meta = meta[meta["country_code"].isin(target_codes)].copy()

        if bounds is not None:
            if isinstance(bounds, dict):
                min_lat, max_lat = bounds["min_lat"], bounds["max_lat"]
                min_lon, max_lon = bounds["min_lon"], bounds["max_lon"]
            else:
                min_lat, min_lon, max_lat, max_lon = bounds
            meta = meta[
                (meta["LAT"] >= min_lat)
                & (meta["LAT"] <= max_lat)
                & (meta["LON"] >= min_lon)
                & (meta["LON"] <= max_lon)
            ].copy()

        # Merge NOAA station inventory to expose observation lifecycle
        if include_inventory:
            try:
                inv_df = cls.get_station_inventory(cache_dir=cache_dir)
                meta = meta.merge(inv_df, on="ID", how="left")
                # Classify lifecycle status
                meta["status"] = "Historical (ended " + meta["last_year"].fillna(0).astype(int).astype(str) + ")"
                meta.loc[meta["last_year"] >= 2024, "status"] = "Active"
                meta.loc[meta["last_year"].isna(), "status"] = "Unknown"

                # Enrich with independent per-variable activeness columns
                var_inv = cls.get_variable_inventory(cache_dir=cache_dir)
                sub_var = var_inv[var_inv["ID"].isin(set(meta["ID"]))]
                if not sub_var.empty:
                    grouped = {k: v for k, v in sub_var.groupby("ID")}
                    var_rows = []
                    for st in meta["ID"]:
                        s_vars = grouped.get(st, pd.DataFrame())
                        if not s_vars.empty:
                            r_rain = s_vars[s_vars["variable"] == "rain"]
                            r_tmax = s_vars[s_vars["variable"] == "tmax"]
                            r_tmin = s_vars[s_vars["variable"] == "tmin"]
                            r_tavg = s_vars[s_vars["variable"] == "tavg"]
                            r_wind = s_vars[s_vars["variable"] == "wind"]
                            r_mdpr = s_vars[s_vars["variable"] == "multi_day_prcp"]
                            r_evap = s_vars[s_vars["variable"] == "pan_evaporation"]
                            r_tobs = s_vars[s_vars["variable"] == "obs_temp"]
                            r_gust = s_vars[s_vars["variable"] == "wind_gust"]

                            act_vars = s_vars[s_vars["is_active"]]["variable"].tolist()
                            ended_vars = [f"{r['variable']} ({r['LAST_YEAR']})" for _, r in s_vars[~s_vars["is_active"]].iterrows()]

                            var_rows.append({
                                "ID": st,
                                "rain_status": r_rain["status"].iloc[0] if not r_rain.empty else "Not Recorded",
                                "rain_window": r_rain["coverage_window"].iloc[0] if not r_rain.empty else "None",
                                "tmax_status": r_tmax["status"].iloc[0] if not r_tmax.empty else "Not Recorded",
                                "tmax_window": r_tmax["coverage_window"].iloc[0] if not r_tmax.empty else "None",
                                "tmin_status": r_tmin["status"].iloc[0] if not r_tmin.empty else "Not Recorded",
                                "tmin_window": r_tmin["coverage_window"].iloc[0] if not r_tmin.empty else "None",
                                "tavg_status": r_tavg["status"].iloc[0] if not r_tavg.empty else "Not Recorded",
                                "wind_status": r_wind["status"].iloc[0] if not r_wind.empty else "Not Recorded",
                                "wind_window": r_wind["coverage_window"].iloc[0] if not r_wind.empty else "None",
                                "mdpr_status": r_mdpr["status"].iloc[0] if not r_mdpr.empty else "Not Recorded",
                                "mdpr_window": r_mdpr["coverage_window"].iloc[0] if not r_mdpr.empty else "None",
                                "evap_status": r_evap["status"].iloc[0] if not r_evap.empty else "Not Recorded",
                                "obs_temp_status": r_tobs["status"].iloc[0] if not r_tobs.empty else "Not Recorded",
                                "active_variables": ", ".join(act_vars) if act_vars else "None",
                                "ended_variables": ", ".join(ended_vars) if ended_vars else "None",
                            })
                        else:
                            var_rows.append({
                                "ID": st,
                                "rain_status": "Not Recorded",
                                "rain_window": "None",
                                "tmax_status": "Not Recorded",
                                "tmax_window": "None",
                                "tmin_status": "Not Recorded",
                                "tmin_window": "None",
                                "tavg_status": "Not Recorded",
                                "wind_status": "Not Recorded",
                                "wind_window": "None",
                                "mdpr_status": "Not Recorded",
                                "mdpr_window": "None",
                                "evap_status": "Not Recorded",
                                "obs_temp_status": "Not Recorded",
                                "active_variables": "None",
                                "ended_variables": "None",
                            })
                    var_df = pd.DataFrame(var_rows)
                    meta = meta.merge(var_df, on="ID", how="left")
            except Exception:
                pass

        if active_only:
            min_year = min_year or 2020

        if min_year is not None and "last_year" in meta.columns:
            meta = meta[meta["last_year"] >= min_year].copy()

        if "last_year" in meta.columns:
            meta = meta.sort_values(by=["last_year", "NAME"], ascending=[False, True]).reset_index(drop=True)
        else:
            meta = meta.reset_index(drop=True)

        return meta

    @classmethod
    def get_african_stations(
        cls,
        region: Optional[str] = None,
        countries: Optional[Union[str, List[str]]] = None,
        country_codes: Optional[Union[str, List[str]]] = None,
        bounds: Optional[Union[Dict[str, float], Tuple[float, float, float, float]]] = None,
        cache_dir: str = "./ghcn_cache",
        include_inventory: bool = True,
        min_year: Optional[int] = None,
        active_only: bool = False,
    ) -> pd.DataFrame:
        """
        Dynamically discover African GHCN stations across East, West, North, Southern, and Central Africa.
        Defaults to all Africa if no regional filters are provided.
        """
        if region is None and countries is None and country_codes is None and bounds is None:
            region = "africa"
        return cls.get_stations(
            region=region,
            countries=countries,
            country_codes=country_codes,
            bounds=bounds,
            cache_dir=cache_dir,
            include_inventory=include_inventory,
            min_year=min_year,
            active_only=active_only,
        )

    async def _fetch_parquet_keys(self, keys: List[str]) -> List[pd.DataFrame]:
        frames = []
        for key in keys:
            url = f"{self.HTTP_BASE_URL}/{key}"
            try:
                def _fetch():
                    req = urllib.request.Request(url, headers={"User-Agent": "MasharikiWeather/1.0"})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        return resp.read()

                data = await asyncio.to_thread(_fetch)
                if HAS_PYARROW:
                    tbl = pq.read_table(io.BytesIO(data), columns=["ID", "DATE", "DATA_VALUE", "Q_FLAG"])
                    frames.append(tbl.to_pandas())
                else:
                    frames.append(pd.read_parquet(io.BytesIO(data), columns=["ID", "DATE", "DATA_VALUE", "Q_FLAG"]))
            except Exception:
                continue
        return frames

    async def _fetch_station_element(self, station: str, element: str) -> pd.DataFrame:
        prefix = f"parquet/by_station/STATION={station}/ELEMENT={element}/"
        cache_file = os.path.join(self.cache_dir, f"d_{station}_{element}.parquet")

        if _is_cache_fresh(cache_file, max_age_s=_MUTABLE_CACHE_MAX_AGE_S):
            try:
                return await asyncio.to_thread(pd.read_parquet, cache_file)
            except Exception:
                pass

        keys = await asyncio.to_thread(_list_s3_prefix_keys, self.S3_BUCKET, prefix)
        frames = await self._fetch_parquet_keys(keys)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            await asyncio.to_thread(combined.to_parquet, cache_file, index=False)
            return combined
        return pd.DataFrame()

    async def _fetch_year_element(self, year: int, element: str) -> pd.DataFrame:
        prefix = f"parquet/by_year/YEAR={year}/ELEMENT={element}/"
        max_age = _MUTABLE_CACHE_MAX_AGE_S if year >= datetime.now().year else None
        cache_file = os.path.join(self.cache_dir, f"d_year_{year}_{element}.parquet")

        if _is_cache_fresh(cache_file, max_age_s=max_age):
            try:
                return await asyncio.to_thread(pd.read_parquet, cache_file)
            except Exception:
                pass

        keys = await asyncio.to_thread(_list_s3_prefix_keys, self.S3_BUCKET, prefix)
        frames = await self._fetch_parquet_keys(keys)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            await asyncio.to_thread(combined.to_parquet, cache_file, index=False)
            return combined
        return pd.DataFrame()

    async def _fetch_ncei_station(self, station: str) -> pd.DataFrame:
        """Download comprehensive station history directly from NOAA NCEI access endpoint."""
        cache_file = os.path.join(self.cache_dir, f"ncei_{station}.parquet")
        if _is_cache_fresh(cache_file, max_age_s=_MUTABLE_CACHE_MAX_AGE_S):
            try:
                return await asyncio.to_thread(pd.read_parquet, cache_file)
            except Exception:
                pass

        url = f"{self.NCEI_ACCESS_URL}/{station}.csv"

        def _download():
            req = urllib.request.Request(url, headers={"User-Agent": "MasharikiWeather/1.0"})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = resp.read()
                    if not data:
                        return pd.DataFrame()
                    df = pd.read_csv(io.BytesIO(data), low_memory=False)
                    return df
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        return pd.DataFrame()
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                except Exception:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            return pd.DataFrame()

        df = await asyncio.to_thread(_download)
        if not df.empty:
            try:
                await asyncio.to_thread(df.to_parquet, cache_file, index=False)
            except Exception:
                pass
        return df

    def get_coverage_summary(
        self,
        variables: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """
        Inspect observation date ranges, count of records, and availability status across stations.
        """
        vars_to_check = variables or ["tp", "t2m_max", "t2m_min"]
        st_df = self.get_stations_dataframe()
        dt_start = pd.to_datetime(start_date) if start_date else None
        dt_end = pd.to_datetime(end_date) if end_date else None

        records = []
        for _, row in st_df.iterrows():
            st = row["station"]
            for v in vars_to_check:
                elem, _, can_v = resolve_daily_variable(v)
                cache_file = os.path.join(self.cache_dir, f"d_{st}_{elem}.parquet")
                if os.path.exists(cache_file):
                    try:
                        df_c = pd.read_parquet(cache_file)
                        if not df_c.empty:
                            df_c["time"] = pd.to_datetime(df_c["DATE"].astype(str), format="%Y%m%d", errors="coerce")
                            total_cnt = len(df_c)
                            min_d = df_c["time"].min()
                            max_d = df_c["time"].max()
                            if dt_start and dt_end:
                                sub = df_c[(df_c["time"] >= dt_start) & (df_c["time"] <= dt_end)]
                                win_cnt = len(sub)
                            else:
                                win_cnt = total_cnt
                            status = "Active" if max_d.year >= 2025 else f"Historical (ended {max_d.year})"
                            records.append({
                                "station": st,
                                "name": row.get("name", ""),
                                "variable": can_v,
                                "total_records": total_cnt,
                                "first_recorded": min_d.strftime("%Y-%m-%d") if pd.notna(min_d) else "",
                                "last_recorded": max_d.strftime("%Y-%m-%d") if pd.notna(max_d) else "",
                                "records_in_window": win_cnt,
                                "status": status,
                            })
                            continue
                    except Exception:
                        pass
                records.append({
                    "station": st,
                    "name": row.get("name", ""),
                    "variable": can_v,
                    "total_records": 0,
                    "first_recorded": "None",
                    "last_recorded": "None",
                    "records_in_window": 0,
                    "status": "Not downloaded / Unavailable",
                })
    def validate_station_coverage(
        self,
        variables: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """
        Validate observation date ranges and element availability for all configured stations.
        Ensures users know beforehand whether a station has records in the queried window.

        Returns
        -------
        pd.DataFrame
            Diagnostics table with columns:
            ['station', 'name', 'data_coverage', 'data_start_year', 'data_end_year',
             'status', 'is_reporting_in_window', 'has_requested_elements',
             'prcp_coverage', 'temp_coverage', 'validity_status', 'elements_available']
        """
        st_df = self.get_stations_dataframe(include_inventory=True)
        dt_start_yr = pd.to_datetime(start_date).year if start_date else 1980
        dt_end_yr = pd.to_datetime(end_date).year if end_date else datetime.now().year

        vars_to_check = variables or ["tp", "t2m_max", "t2m_min"]
        needs_rain = any(str(v).lower() in ("tp", "prcp", "rain", "precipitation", "rainfall") for v in vars_to_check)
        needs_temp = any(str(v).lower() in ("t2m", "tmax", "tmin", "t2m_max", "t2m_min", "temperature", "temp", "tavg") for v in vars_to_check)

        reports = []
        for _, row in st_df.iterrows():
            st = row["station"]
            name = row.get("name", "")
            start_yr = row.get("data_start_year")
            end_yr = row.get("data_end_year")
            status = row.get("status", "Unknown")
            cov = row.get("data_coverage", "Unknown")
            elems = str(row.get("elements_available", ""))
            has_p = bool(row.get("has_precipitation", False))
            has_t = bool(row.get("has_temperature", False))
            prcp_cov = str(row.get("prcp_coverage", "None"))
            temp_cov = str(row.get("temp_coverage", "None"))

            in_window = True
            validity = "Valid"

            if pd.isna(end_yr) or pd.isna(start_yr):
                in_window = False
                validity = "No NOAA Inventory Records"
            elif end_yr < dt_start_yr:
                in_window = False
                validity = f"Historical (Ended in {int(end_yr)} < {dt_start_yr})"
            elif start_yr > dt_end_yr:
                in_window = False
                validity = f"Future / Late start (Begins {int(start_yr)} > {dt_end_yr})"

            elem_ok = True
            if needs_rain and not has_p:
                elem_ok = False
                validity = "Precipitation (PRCP) Not Recorded"
            elif needs_temp and not has_t:
                elem_ok = False
                validity = "Temperature (TMAX/TMIN) Not Recorded"

            reports.append({
                "station": st,
                "name": name,
                "data_coverage": cov,
                "data_start_year": int(start_yr) if pd.notna(start_yr) else None,
                "data_end_year": int(end_yr) if pd.notna(end_yr) else None,
                "status": status,
                "is_reporting_in_window": in_window,
                "has_requested_elements": elem_ok,
                "prcp_coverage": prcp_cov,
                "temp_coverage": temp_cov,
                "validity_status": validity,
                "elements_available": elems,
            })
        return pd.DataFrame(reports)

    def validate_variable_activeness(
        self,
        variables: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """
        Validate operational date ranges and sensor activeness for each requested
        variable independently across all configured stations.

        Parameters
        ----------
        variables : list of str, optional
            List of variables to inspect (e.g. ['rain', 'tmax', 'tmin']).
        start_date : str or datetime, optional
            Start date of the queried window.
        end_date : str or datetime, optional
            End date of the queried window.

        Returns
        -------
        pd.DataFrame
            Table detailing for each (station, variable):
            ['station', 'name', 'variable', 'element', 'first_year', 'last_year',
             'coverage_window', 'sensor_lifecycle', 'query_status', 'is_reporting_in_window']
        """
        st_df = self.get_stations_dataframe(include_inventory=False)
        dt_start_yr = pd.to_datetime(start_date).year if start_date else 1980
        dt_end_yr = pd.to_datetime(end_date).year if end_date else datetime.now().year

        vars_to_check = variables or ["tp", "t2m_max", "t2m_min"]
        var_inv = self.get_variable_inventory(cache_dir=self.cache_dir)
        sub_var = var_inv[var_inv["ID"].isin(set(self.stations))]
        grouped = {k: v for k, v in sub_var.groupby("ID")} if not sub_var.empty else {}

        reports = []
        for _, row in st_df.iterrows():
            st = row["station"]
            name = row.get("name", "")
            s_vars = grouped.get(st, pd.DataFrame())

            for v in vars_to_check:
                elem, _, can_v = resolve_daily_variable(v)

                # Special handling for derived Diurnal Temperature Range (DTR)
                if can_v in ["dtr", "diurnal_range", "diurnal_temperature_range"] or elem == "DTR":
                    tmax_m = s_vars[s_vars["ELEMENT"] == "TMAX"] if not s_vars.empty else pd.DataFrame()
                    tmin_m = s_vars[s_vars["ELEMENT"] == "TMIN"] if not s_vars.empty else pd.DataFrame()
                    if not tmax_m.empty and not tmin_m.empty:
                        first_yr = max(int(tmax_m["FIRST_YEAR"].iloc[0]), int(tmin_m["FIRST_YEAR"].iloc[0]))
                        last_yr = min(int(tmax_m["LAST_YEAR"].iloc[0]), int(tmin_m["LAST_YEAR"].iloc[0]))
                        is_active = last_yr >= 2024
                        cov = f"{first_yr} - {last_yr}"
                        lifecycle = f"Derived (thru {last_yr})" if is_active else f"Derived (ended {last_yr})"
                        if last_yr < dt_start_yr:
                            q_status = f"Decommissioned (temp ended {last_yr} < query start {dt_start_yr})"
                            in_window = False
                        else:
                            q_status = f"Active derived (available through {last_yr})"
                            in_window = True
                        reports.append({
                            "station": st,
                            "name": name,
                            "variable": can_v,
                            "element": "DTR",
                            "first_year": first_yr,
                            "last_year": last_yr,
                            "coverage_window": cov,
                            "sensor_lifecycle": lifecycle,
                            "query_status": q_status,
                            "is_reporting_in_window": in_window,
                        })
                        continue

                var_match = pd.DataFrame()
                if not s_vars.empty:
                    var_match = s_vars[s_vars["variable"] == can_v]
                    if var_match.empty:
                        var_match = s_vars[s_vars["ELEMENT"] == elem]

                if not var_match.empty:
                    first_yr = int(var_match["FIRST_YEAR"].iloc[0])
                    last_yr = int(var_match["LAST_YEAR"].iloc[0])
                    is_active = bool(var_match["is_active"].iloc[0])
                    cov = f"{first_yr} - {last_yr}"
                    lifecycle = f"Active (thru {last_yr})" if is_active else f"Ended {last_yr}"

                    if last_yr < dt_start_yr:
                        q_status = f"Decommissioned (ended {last_yr} < query start {dt_start_yr})"
                        in_window = False
                    elif first_yr > dt_end_yr:
                        q_status = f"Late Start (begins {first_yr} > query end {dt_end_yr})"
                        in_window = False
                    elif last_yr < dt_end_yr:
                        q_status = f"Partial (sensor ended {last_yr} during query window)"
                        in_window = True
                    else:
                        q_status = f"Active (available through {last_yr})"
                        in_window = True
                else:
                    first_yr, last_yr = None, None
                    cov = "None"
                    lifecycle = "Not Recorded"
                    q_status = "Variable not observed at this station"
                    in_window = False

                reports.append({
                    "station": st,
                    "name": name,
                    "variable": can_v,
                    "element": elem,
                    "first_year": first_yr,
                    "last_year": last_yr,
                    "coverage_window": cov,
                    "sensor_lifecycle": lifecycle,
                    "query_status": q_status,
                    "is_reporting_in_window": in_window,
                })
        return pd.DataFrame(reports)

    async def fetch(
        self,
        variables: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        format: str = "long",
        column_key: str = "station",
        apply_qc: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Fetch daily observations across configured stations.
        """
        if not self.stations:
            raise ValueError("No stations configured.")

        qc_active = self.apply_qc if apply_qc is None else apply_qc
        dt_start = pd.to_datetime(start_date) if start_date else pd.to_datetime("1980-01-01")
        if end_date:
            dt_end = pd.to_datetime(end_date)
            if isinstance(end_date, str) and len(end_date.strip()) <= 10:
                dt_end = dt_end.replace(hour=23, minute=59, second=59)
        else:
            dt_end = datetime.now()

        semaphore = asyncio.Semaphore(self.async_workers)

        # Resolve requested variables to canonical elements
        elem_map: Dict[str, Tuple[str, Callable[[Any], Any]]] = {}
        requested_vars = []
        needs_derived_t2m = False
        needs_derived_dtr = False

        for v in variables:
            elem, conv, can_v = resolve_daily_variable(v)
            requested_vars.append(can_v)
            if can_v in ["t2m", "t2m_avg", "tavg"]:
                needs_derived_t2m = True
            if can_v in ["dtr", "diurnal_range", "diurnal_temperature_range"]:
                needs_derived_dtr = True
            if elem != "DTR":
                elem_map[can_v] = (elem, conv)

        # If t2m or dtr requested, ensure TMAX and TMIN are also fetched for auto-calculation
        if needs_derived_t2m or needs_derived_dtr:
            if "t2m_max" not in elem_map and "tmax" not in elem_map:
                elem_map["_tmax_helper"] = ("TMAX", lambda x: x / 10.0)
            if "t2m_min" not in elem_map and "tmin" not in elem_map:
                elem_map["_tmin_helper"] = ("TMIN", lambda x: x / 10.0)

        raw_frames = []
        ncei_needed_stations: Set[str] = set()

        if self.fetch_mode != "ncei":
            use_by_station = len(self.stations) <= self._BY_STATION_THRESHOLD
            if use_by_station:
                async def _fetch_st(st, el):
                    async with semaphore:
                        return await self._fetch_station_element(st, el)

                tasks = [(_fetch_st(st, elem), v, st) for v, (elem, _) in elem_map.items() for st in self.stations]
                results = await asyncio.gather(*(t[0] for t in tasks), return_exceptions=True)

                fetched_stations = set()
                for (_, v, st), res in zip(tasks, results):
                    if isinstance(res, pd.DataFrame) and not res.empty:
                        df = res.copy()
                        df["variable"] = v
                        df["DATA_VALUE"] = elem_map[v][1](pd.to_numeric(df["DATA_VALUE"], errors="coerce"))
                        raw_frames.append(df)
                        fetched_stations.add(st)

                ncei_needed_stations = set(self.stations) - fetched_stations
            else:
                years = list(range(dt_start.year, dt_end.year + 1))
                st_set = set(self.stations)

                async def _fetch_yr(yr, el):
                    async with semaphore:
                        return await self._fetch_year_element(yr, el)

                tasks = [(_fetch_yr(yr, elem), v) for v, (elem, _) in elem_map.items() for yr in years]
                results = await asyncio.gather(*(t[0] for t in tasks), return_exceptions=True)

                for (_, v), res in zip(tasks, results):
                    if isinstance(res, pd.DataFrame) and not res.empty:
                        df = res[res["ID"].isin(st_set)].copy()
                        if not df.empty:
                            df["variable"] = v
                            df["DATA_VALUE"] = elem_map[v][1](pd.to_numeric(df["DATA_VALUE"], errors="coerce"))
                            raw_frames.append(df)
        else:
            ncei_needed_stations = set(self.stations)

        # Fallback to NCEI CSV
        if ncei_needed_stations:
            async def _fetch_ncei(st):
                async with semaphore:
                    return await self._fetch_ncei_station(st)

            ncei_tasks = [(_fetch_ncei(st), st) for st in ncei_needed_stations]
            ncei_results = await asyncio.gather(*(t[0] for t in ncei_tasks), return_exceptions=True)

            for (_, st), res in zip(ncei_tasks, ncei_results):
                if isinstance(res, pd.DataFrame) and not res.empty:
                    df = res.copy()
                    df["DATE"] = df["DATE"].astype(str).str.replace("-", "")
                    for v, (elem, unit_conv) in elem_map.items():
                        if elem in df.columns:
                            sub = pd.DataFrame()
                            sub["ID"] = df["STATION"]
                            sub["DATE"] = df["DATE"]
                            sub["DATA_VALUE"] = unit_conv(pd.to_numeric(df[elem], errors="coerce"))
                            flag_col = f"{elem}_ATTRIBUTES"
                            if flag_col in df.columns:
                                sub["Q_FLAG"] = df[flag_col].apply(_extract_q_flag)
                            else:
                                sub["Q_FLAG"] = ""
                            sub["variable"] = v
                            raw_frames.append(sub.dropna(subset=["DATA_VALUE"]))

        if not raw_frames:
            empty_long = pd.DataFrame(columns=["time", "lat", "lon", "elev", "station", "observation", "variable", "lat_lon"])
            if format.lower() == "wide":
                return to_wide(empty_long, column_key=column_key)
            return empty_long

        df = pd.concat(raw_frames, ignore_index=True)
        # Quality control flag filtering (empty or NaN passed all QC checks)
        df = df[df["Q_FLAG"].isna() | (df["Q_FLAG"] == "") | (df["Q_FLAG"] == " ")]
        df["time"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["time", "DATA_VALUE"])

        df = df[(df["time"] >= dt_start) & (df["time"] <= dt_end)]

        meta = self.get_station_metadata(self.cache_dir)[["ID", "LAT", "LON", "ELEV"]].drop_duplicates("ID")
        df = df.merge(meta, on="ID", how="left").rename(columns={
            "LAT": "lat", "LON": "lon", "ELEV": "elev", "ID": "station", "DATA_VALUE": "observation"
        })

        # Smart automatic derivation of t2m from (tmax + tmin) / 2 if TAVG was unpopulated
        if needs_derived_t2m:
            target_t2m_name = next((v for v in requested_vars if v in ["t2m", "t2m_avg", "tavg"]), "t2m")
            has_direct_tavg = not df[df["variable"] == target_t2m_name].empty
            if not has_direct_tavg:
                tmax_sub = df[df["variable"].isin(["t2m_max", "_tmax_helper", "tmax"])][["time", "station", "lat", "lon", "elev", "observation"]].rename(columns={"observation": "tmax_val"})
                tmin_sub = df[df["variable"].isin(["t2m_min", "_tmin_helper", "tmin"])][["time", "station", "observation"]].rename(columns={"observation": "tmin_val"})
                if not tmax_sub.empty and not tmin_sub.empty:
                    derived = tmax_sub.merge(tmin_sub, on=["time", "station"], how="inner")
                    derived["observation"] = (derived["tmax_val"] + derived["tmin_val"]) / 2.0
                    derived["variable"] = target_t2m_name
                    derived = derived[["time", "lat", "lon", "elev", "station", "observation", "variable"]]
                    df = pd.concat([df, derived], ignore_index=True)

        # Smart automatic derivation of dtr (diurnal temperature range) from (tmax - tmin)
        if needs_derived_dtr:
            target_dtr_name = next((v for v in requested_vars if v in ["dtr", "diurnal_range", "diurnal_temperature_range"]), "dtr")
            tmax_sub = df[df["variable"].isin(["t2m_max", "_tmax_helper", "tmax"])][["time", "station", "lat", "lon", "elev", "observation"]].rename(columns={"observation": "tmax_val"})
            tmin_sub = df[df["variable"].isin(["t2m_min", "_tmin_helper", "tmin"])][["time", "station", "observation"]].rename(columns={"observation": "tmin_val"})
            if not tmax_sub.empty and not tmin_sub.empty:
                derived_dtr = tmax_sub.merge(tmin_sub, on=["time", "station"], how="inner")
                derived_dtr["observation"] = derived_dtr["tmax_val"] - derived_dtr["tmin_val"]
                derived_dtr["variable"] = target_dtr_name
                derived_dtr = derived_dtr[["time", "lat", "lon", "elev", "station", "observation", "variable"]]
                df = pd.concat([df, derived_dtr], ignore_index=True)

        # Remove internal helper variables
        df = df[~df["variable"].str.startswith("_helper") & ~df["variable"].str.startswith("_tmax") & ~df["variable"].str.startswith("_tmin")].copy()

        # If user did not explicitly ask for helper variables, filter down to requested variables
        if requested_vars:
            df = df[df["variable"].isin(set(requested_vars))].copy()

        # Quality control filtering
        df = clean_daily_dataframe(df, apply_qc=qc_active)

        if df.empty:
            empty_long = pd.DataFrame(columns=["time", "lat", "lon", "elev", "station", "observation", "variable", "lat_lon"])
            if format.lower() == "wide":
                return to_wide(empty_long, column_key=column_key)
            return empty_long

        df["lat_lon"] = (
            np.round(df["lat"], 3).astype(str)
            + "_"
            + np.round(df["lon"], 3).astype(str)
        )
        long_df = df[["time", "lat", "lon", "elev", "station", "observation", "variable", "lat_lon"]].reset_index(drop=True)

        if format.lower() == "wide":
            return to_wide(long_df, column_key=column_key)
        return long_df

    def fetch_sync(
        self,
        variables: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        format: str = "long",
        column_key: str = "station",
        apply_qc: Optional[bool] = None,
    ) -> pd.DataFrame:
        return self._run_async(self.fetch(variables, start_date, end_date, format=format, column_key=column_key, apply_qc=apply_qc))

    def __call__(
        self,
        variables: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        format: str = "long",
        column_key: str = "station",
        apply_qc: Optional[bool] = None,
    ) -> pd.DataFrame:
        return self.fetch_sync(variables, start_date, end_date, format=format, column_key=column_key, apply_qc=apply_qc)


# =====================================================================
# 6. GHCN-Hourly Engine
# =====================================================================

class GHCNHourly(_GHCNBase):
    BASE_URL: str = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network"
    METADATA_URL: str = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-station-list.csv"
    COUNTRIES_URL: str = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcn-countries.txt"

    def __init__(
        self,
        stations: Optional[List[str]] = None,
        region: Optional[str] = None,
        countries: Optional[Union[str, List[str]]] = None,
        country_codes: Optional[Union[str, List[str]]] = None,
        bounds: Optional[Union[Dict[str, float], Tuple[float, float, float, float]]] = None,
        cache_dir: str = "./ghcn_cache",
        verbose: bool = True,
        async_workers: int = 16,
        fetch_mode: str = "auto",
        access_method: str = "auto",
        apply_qc: bool = True,
        min_year: Optional[int] = None,
        active_only: bool = False,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.async_workers = async_workers
        self.fetch_mode = fetch_mode or access_method
        self.access_method = (access_method or fetch_mode or "auto").lower()
        self.apply_qc = apply_qc
        os.makedirs(self.cache_dir, exist_ok=True)

        self.station_alias_map: Dict[str, str] = {}
        self.requested_stations = list(stations or [])

        # Resolve explicit station inputs (supports ICAO codes like HKJK, WMO IDs, or GHCN_IDs)
        resolved_stations: List[str] = []
        if stations:
            for s in stations:
                canon_id, meta = self.resolve_station(s, cache_dir=cache_dir)
                resolved_stations.append(canon_id)
                self.station_alias_map[s.upper()] = canon_id
                self.station_alias_map[canon_id] = s.upper()
                if meta.get("icao"):
                    self.station_alias_map[meta["icao"].upper()] = canon_id

        selected_stations: Set[str] = set(resolved_stations)

        # Dynamic African region / country / bounding box station discovery
        if region or countries or country_codes:
            african_df = self.get_african_stations(
                region=region,
                countries=countries,
                country_codes=country_codes,
                bounds=bounds,
                cache_dir=cache_dir,
                min_year=min_year,
                active_only=active_only,
            )
            id_col = "ID" if "ID" in african_df.columns else "GHCN_ID"
            selected_stations.update(african_df[id_col].dropna().tolist())
        elif bounds is not None and not stations:
            bbox_stations = self.get_stations_bbox(bounds, cache_dir=cache_dir)
            selected_stations.update(bbox_stations)

        self.stations = sorted(list(selected_stations))

    @classmethod
    def resolve_station(cls, query: str, cache_dir: str = "./ghcn_cache") -> Tuple[str, Dict[str, Any]]:
        """
        Resolve any station identifier (e.g. ICAO airport code 'HKJK', WMO '63686',
        or canonical GHCNh ID 'KEI0000HKJK') to its official NOAA GHCN_ID.
        Returns (canonical_ghcn_id, metadata_dict).
        """
        clean = str(query).strip().upper()
        if clean.startswith("GHCNH_"):
            clean = clean[6:]

        try:
            meta = cls.get_station_metadata(cache_dir=cache_dir)
        except Exception:
            return clean, {}

        id_col = "ID" if "ID" in meta.columns else "GHCN_ID"

        # 1. Direct GHCN_ID / ID match
        m = meta[meta[id_col].astype(str).str.upper() == clean]
        if not m.empty:
            row = m.iloc[0]
            return str(row[id_col]), {
                "name": str(row.get("NAME", "")),
                "icao": str(row.get("ICAO", "")),
                "wmo": str(row.get("WMO_ID", "")),
            }

        # 2. ICAO airport code match (e.g. HKJK, HKNW, HKMO, HKEL)
        if "ICAO" in meta.columns:
            m = meta[meta["ICAO"].astype(str).str.upper() == clean]
            if not m.empty:
                row = m.iloc[0]
                return str(row[id_col]), {
                    "name": str(row.get("NAME", "")),
                    "icao": str(row.get("ICAO", "")),
                    "wmo": str(row.get("WMO_ID", "")),
                }

        # 3. WMO ID match
        if "WMO_ID" in meta.columns:
            clean_wmo = clean.replace(".0", "")
            m = meta[meta["WMO_ID"].astype(str).str.strip().str.replace(".0", "", regex=False) == clean_wmo]
            if not m.empty:
                row = m.iloc[0]
                return str(row[id_col]), {
                    "name": str(row.get("NAME", "")),
                    "icao": str(row.get("ICAO", "")),
                    "wmo": str(row.get("WMO_ID", "")),
                }

        # 4. Partial suffix match (e.g. ID ends with clean query)
        m = meta[meta[id_col].astype(str).str.upper().str.endswith(clean)]
        if not m.empty:
            row = m.iloc[0]
            return str(row[id_col]), {
                "name": str(row.get("NAME", "")),
                "icao": str(row.get("ICAO", "")),
                "wmo": str(row.get("WMO_ID", "")),
            }

        return clean, {}

    @classmethod
    def get_station_metadata(cls, cache_dir: str = "./ghcn_cache") -> pd.DataFrame:
        """Load and cache official NOAA GHCN-Hourly station list."""
        os.makedirs(cache_dir, exist_ok=True)
        meta_path = os.path.join(cache_dir, "ghcnh-station-list.csv")

        if not os.path.exists(meta_path) or os.path.getsize(meta_path) == 0:
            req = urllib.request.Request(cls.METADATA_URL, headers={"User-Agent": "MasharikiWeather/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(meta_path, "wb") as f:
                    f.write(resp.read())

        df = pd.read_csv(meta_path)
        col_rename = {
            "GHCN_ID": "ID", "Station_ID": "ID",
            "NAME": "NAME", "Station_name": "NAME",
            "LATITUDE": "LAT", "Latitude": "LAT",
            "LONGITUDE": "LON", "Longitude": "LON",
            "ELEVATION": "ELEV", "Elevation": "ELEV",
        }
        for old_c, new_c in col_rename.items():
            if old_c in df.columns and new_c not in df.columns:
                df[new_c] = df[old_c]
        if "LAT" not in df.columns and "LATITUDE" in df.columns:
            df["LAT"] = df["LATITUDE"]
        if "LON" not in df.columns and "LONGITUDE" in df.columns:
            df["LON"] = df["LONGITUDE"]
        if "ELEV" not in df.columns and "ELEVATION" in df.columns:
            df["ELEV"] = df["ELEVATION"]
        if "ID" not in df.columns and "GHCN_ID" in df.columns:
            df["ID"] = df["GHCN_ID"]

        df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
        df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
        df["ELEV"] = pd.to_numeric(df["ELEV"], errors="coerce")

        if "ISO_CODE" in df.columns:
            df["country_code"] = df["ISO_CODE"].fillna(df["ID"].str[:2])
        else:
            df["country_code"] = df["ID"].str[:2]
        return df

    @classmethod
    def get_stations_bbox(
        cls,
        bounds: Union[Dict[str, float], Tuple[float, float, float, float]],
        cache_dir: str = "./ghcn_cache",
    ) -> List[str]:
        if isinstance(bounds, dict):
            min_lat, max_lat = bounds["min_lat"], bounds["max_lat"]
            min_lon, max_lon = bounds["min_lon"], bounds["max_lon"]
        else:
            min_lat, min_lon, max_lat, max_lon = bounds

        meta = cls.get_station_metadata(cache_dir)
        sub = meta[
            (meta["LAT"] >= min_lat)
            & (meta["LAT"] <= max_lat)
            & (meta["LON"] >= min_lon)
            & (meta["LON"] <= max_lon)
        ]
        return sub["ID"].tolist()

    @classmethod
    def get_stations(
        cls,
        region: Optional[str] = None,
        countries: Optional[Union[str, List[str]]] = None,
        country_codes: Optional[Union[str, List[str]]] = None,
        bounds: Optional[Union[Dict[str, float], Tuple[float, float, float, float]]] = None,
        cache_dir: str = "./ghcn_cache",
        active_only: bool = False,
        min_year: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Dynamically discover and filter GHCN-Hourly stations worldwide or by region.
        Supports global presets ('europe', 'north_america', 'south_america', 'asia', 'oceania', 'africa'),
        country names (e.g. 'Netherlands', 'Germany', 'Kenya'), or NOAA country codes.
        """
        meta = cls.get_station_metadata(cache_dir=cache_dir).copy()
        if "country_code" not in meta.columns:
            if "ISO_CODE" in meta.columns:
                meta["country_code"] = meta["ISO_CODE"].fillna(meta["ID"].str[:2])
            else:
                meta["country_code"] = meta["ID"].str[:2]

        country_meta = GHCNDaily.get_country_metadata(cache_dir=cache_dir)

        target_codes: Set[str] = set()
        if region:
            reg_clean = region.strip().lower().replace("-", "_").replace(" ", "_")
            if reg_clean in GLOBAL_REGIONS:
                target_codes.update(GLOBAL_REGIONS[reg_clean])
            else:
                available = list(GLOBAL_REGIONS.keys())
                raise ValueError(f"Unknown region '{region}'. Available presets: {available}")

        if country_codes:
            if isinstance(country_codes, str):
                country_codes = [country_codes]
            target_codes.update([cc.upper() for cc in country_codes])

        if countries:
            if isinstance(countries, str):
                countries = [countries]
            for c in countries:
                c_clean = c.strip()
                if len(c_clean) == 2:
                    target_codes.add(c_clean.upper())
                else:
                    c_low = c_clean.lower()
                    matched = country_meta[
                        country_meta["country_name"].str.lower().str.contains(c_low, regex=False)
                    ]
                    if not matched.empty:
                        target_codes.update(matched["country_code"].tolist())
                    matched_ids = meta[meta["NAME"].astype(str).str.upper().str.contains(c_clean.upper(), regex=False)]["ID"].tolist()
                    target_codes.update([m[:2] for m in matched_ids])

        if target_codes:
            iso_col = meta["ISO_CODE"] if "ISO_CODE" in meta.columns else pd.Series(index=meta.index, dtype=str)
            meta = meta[
                meta["country_code"].isin(target_codes)
                | meta["ID"].str[:2].isin(target_codes)
                | iso_col.isin(target_codes)
            ].copy()

        if bounds is not None:
            if isinstance(bounds, dict):
                min_lat, max_lat = bounds["min_lat"], bounds["max_lat"]
                min_lon, max_lon = bounds["min_lon"], bounds["max_lon"]
            else:
                min_lat, min_lon, max_lat, max_lon = bounds
            meta = meta[
                (meta["LAT"] >= min_lat)
                & (meta["LAT"] <= max_lat)
                & (meta["LON"] >= min_lon)
                & (meta["LON"] <= max_lon)
            ].copy()

        return meta.reset_index(drop=True)

    @classmethod
    def get_african_stations(
        cls,
        region: Optional[str] = None,
        countries: Optional[Union[str, List[str]]] = None,
        country_codes: Optional[Union[str, List[str]]] = None,
        bounds: Optional[Union[Dict[str, float], Tuple[float, float, float, float]]] = None,
        cache_dir: str = "./ghcn_cache",
        active_only: bool = False,
        min_year: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Dynamically discover African GHCN-Hourly stations across East, West, North, Southern, and Central Africa.
        Defaults to all Africa if no regional filters are provided.
        """
        if region is None and countries is None and country_codes is None and bounds is None:
            region = "africa"
        return cls.get_stations(
            region=region,
            countries=countries,
            country_codes=country_codes,
            bounds=bounds,
            cache_dir=cache_dir,
            active_only=active_only,
            min_year=min_year,
            **kwargs,
        )

    async def _fetch_station_year(self, station: str, year: int, access_method: str = "auto") -> pd.DataFrame:
        """
        Fetch hourly observations for a station in a specific year.
        Supports by-year Parquet and by-year PSV with local caching.
        """
        cache_pq = os.path.join(self.cache_dir, f"h_{station}_{year}.parquet")
        cache_psv = os.path.join(self.cache_dir, f"h_{station}_{year}.psv")
        max_age = _MUTABLE_CACHE_MAX_AGE_S if year >= datetime.now().year else None

        # Check local cache
        if _is_cache_fresh(cache_pq, max_age_s=max_age):
            try:
                return await asyncio.to_thread(pd.read_parquet, cache_pq)
            except Exception:
                pass
        if _is_cache_fresh(cache_psv, max_age_s=max_age):
            try:
                return await asyncio.to_thread(pd.read_csv, cache_psv, sep="|", low_memory=False)
            except Exception:
                pass

        # Build list of URLs to try
        pq_url = f"{self.BASE_URL}/hourly/access/by-year/{year}/parquet/GHCNh_{station}_{year}.parquet"
        psv_url = f"{self.BASE_URL}/hourly/access/by-year/{year}/psv/GHCNh_{station}_{year}.psv"

        if access_method in ["by_year_psv", "psv"]:
            urls_to_try = [(psv_url, False)]
        elif access_method in ["by_year_parquet", "parquet"]:
            urls_to_try = [(pq_url, True)]
        else:  # "auto" or "by_year"
            if HAS_PYARROW:
                urls_to_try = [(pq_url, True), (psv_url, False)]
            else:
                urls_to_try = [(psv_url, False), (pq_url, True)]

        def _fetch_sync():
            for url, is_parquet in urls_to_try:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "MasharikiWeather/1.0"})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = resp.read()
                    if data:
                        if is_parquet:
                            try:
                                return pd.read_parquet(io.BytesIO(data))
                            except Exception:
                                continue
                        else:
                            return pd.read_csv(io.BytesIO(data), sep="|", low_memory=False)
                except Exception:
                    continue
            return pd.DataFrame()

        df = await asyncio.to_thread(_fetch_sync)
        if not df.empty:
            def _write_cache():
                saved = False
                if HAS_PYARROW:
                    try:
                        df.to_parquet(cache_pq, index=False)
                        saved = True
                    except Exception:
                        pass
                if not saved:
                    try:
                        df.to_csv(cache_psv, sep="|", index=False)
                    except Exception:
                        pass
            await asyncio.to_thread(_write_cache)
        return df

    async def _fetch_station_por(self, station: str) -> pd.DataFrame:
        """
        Fetch complete Period-of-Record (POR) hourly telemetry for a station.
        Uses NCEI by-station access endpoint:
        hourly/access/by-station/GHCNh_{station}_por.psv
        """
        cache_pq = os.path.join(self.cache_dir, f"h_por_{station}.parquet")
        cache_psv = os.path.join(self.cache_dir, f"h_por_{station}.psv")

        # Check local cache
        if _is_cache_fresh(cache_pq, max_age_s=_MUTABLE_CACHE_MAX_AGE_S):
            try:
                return await asyncio.to_thread(pd.read_parquet, cache_pq)
            except Exception:
                pass
        if _is_cache_fresh(cache_psv, max_age_s=_MUTABLE_CACHE_MAX_AGE_S):
            try:
                return await asyncio.to_thread(pd.read_csv, cache_psv, sep="|", low_memory=False)
            except Exception:
                pass

        url = f"{self.BASE_URL}/hourly/access/by-station/GHCNh_{station}_por.psv"

        def _fetch_por_sync():
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "MasharikiWeather/1.0"})
                with urllib.request.urlopen(req, timeout=90) as resp:
                    return pd.read_csv(resp, sep="|", low_memory=False)
            except Exception:
                return pd.DataFrame()

        df = await asyncio.to_thread(_fetch_por_sync)
        if not df.empty:
            def _write_cache():
                saved = False
                if HAS_PYARROW:
                    try:
                        df.to_parquet(cache_pq, index=False)
                        saved = True
                    except Exception:
                        pass
                if not saved:
                    try:
                        df.to_csv(cache_psv, sep="|", index=False)
                    except Exception:
                        pass
            await asyncio.to_thread(_write_cache)
        return df

    async def fetch(
        self,
        variables: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        format: str = "long",
        column_key: str = "station",
        apply_qc: Optional[bool] = None,
        access_method: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch hourly observations across configured stations.

        Parameters
        ----------
        variables : list of str
            Hourly fields to query (e.g. ['t2m', 'dew_point', 'rh', 'ws10m', 'wd', 'tp']).
        start_date : str or datetime, optional
            Start date ('YYYY-MM-DD'). Default '2020-01-01'.
        end_date : str or datetime, optional
            End date ('YYYY-MM-DD'). Default today.
        format : str, default 'long'
            'long' (Earth2Studio schema) or 'wide' (Date x Stations matrix).
        column_key : str, default 'station'
            Header name in wide format ('station' or 'lat_lon').
        apply_qc : bool, optional
            Whether to sanitize unphysical artifacts.
        access_method : str, optional
            Ingestion method:
            - 'auto' (default): by-year Parquet with PSV and by-station fallbacks
            - 'by_year' / 'by_year_parquet': NOAA annual Parquet slices
            - 'by_year_psv': NOAA annual pipe-separated values
            - 'by_station' / 'by_station_psv': NOAA full period-of-record (POR)
        """
        if not self.stations:
            raise ValueError("No stations configured.")

        method = (access_method or self.access_method or "auto").lower()
        qc_active = self.apply_qc if apply_qc is None else apply_qc
        dt_start = pd.to_datetime(start_date) if start_date else pd.to_datetime("2020-01-01")
        if end_date:
            dt_end = pd.to_datetime(end_date)
            if isinstance(end_date, str) and len(end_date.strip()) <= 10:
                dt_end = dt_end.replace(hour=23, minute=59, second=59)
        else:
            dt_end = datetime.now()
        years = list(range(dt_start.year, dt_end.year + 1))

        semaphore = asyncio.Semaphore(self.async_workers)
        frames = []

        if method in ["by_station", "by_station_psv"]:
            async def _bounded_por(st):
                async with semaphore:
                    return await self._fetch_station_por(st)

            tasks = [(_bounded_por(st), st) for st in self.stations]
            results = await asyncio.gather(*(t[0] for t in tasks), return_exceptions=True)
            frames = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
        else:
            async def _bounded_year(st, yr):
                async with semaphore:
                    return await self._fetch_station_year(st, yr, access_method=method)

            tasks = [(_bounded_year(st, yr), st) for st in self.stations for yr in years]
            results = await asyncio.gather(*(t[0] for t in tasks), return_exceptions=True)
            frames = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]

            # In auto mode, if any stations yielded no records, fallback to by-station POR
            if method == "auto":
                seen_stations = {r["STATION"].iloc[0] for r in frames if "STATION" in r.columns and not r.empty}
                missing_stations = set(self.stations) - seen_stations
                if missing_stations:
                    async def _fallback_por(st):
                        async with semaphore:
                            return await self._fetch_station_por(st)

                    por_tasks = [(_fallback_por(st), st) for st in missing_stations]
                    por_results = await asyncio.gather(*(t[0] for t in por_tasks), return_exceptions=True)
                    for r in por_results:
                        if isinstance(r, pd.DataFrame) and not r.empty:
                            frames.append(r)

        if not frames:
            empty_long = pd.DataFrame(columns=["time", "lat", "lon", "elev", "station", "observation", "variable", "lat_lon"])
            if format.lower() == "wide":
                return to_wide(empty_long, column_key=column_key)
            return empty_long

        df = pd.concat(frames, ignore_index=True)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df[(df["DATE"] >= dt_start) & (df["DATE"] <= dt_end)]

        # Apply meteorological quality control sanitization to raw fields
        df = clean_hourly_dataframe(df, apply_qc=qc_active)

        # Vectorized wind vector decomposition
        if "wind_direction" in df.columns and "wind_speed" in df.columns:
            rad = np.radians(pd.to_numeric(df["wind_direction"], errors="coerce"))
            ws = pd.to_numeric(df["wind_speed"], errors="coerce")
            df["u10m"] = -np.sin(rad) * ws
            df["v10m"] = -np.cos(rad) * ws

        # Derive relative humidity if requested or possible
        needs_rh = any(resolve_hourly_variable(v)[2] in ["rh", "relative_humidity"] for v in variables)
        if needs_rh:
            if "relative_humidity" not in df.columns or df["relative_humidity"].isna().all():
                if "temperature" in df.columns and "dew_point_temperature" in df.columns:
                    df["relative_humidity"] = calculate_relative_humidity(df["temperature"], df["dew_point_temperature"])

        # Melt requested variables to long standard schema
        long_frames = []
        for v in variables:
            col_name, _, can_v = resolve_hourly_variable(v)
            target_col = col_name if col_name in df.columns else (can_v if can_v in df.columns else None)
            if target_col is not None and target_col in df.columns:
                sub = df[["DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "STATION", target_col]].copy()
                sub = sub.rename(columns={
                    "DATE": "time", "LATITUDE": "lat", "LONGITUDE": "lon",
                    "ELEVATION": "elev", "STATION": "station", target_col: "observation"
                })
                sub["observation"] = pd.to_numeric(sub["observation"], errors="coerce")
                sub["variable"] = can_v
                long_frames.append(sub.dropna(subset=["observation"]))

        if not long_frames:
            empty_long = pd.DataFrame(columns=["time", "lat", "lon", "elev", "station", "observation", "variable", "lat_lon"])
            if format.lower() == "wide":
                return to_wide(empty_long, column_key=column_key)
            return empty_long

        res_df = pd.concat(long_frames, ignore_index=True).reset_index(drop=True)
        res_df["lat_lon"] = (
            np.round(res_df["lat"], 3).astype(str)
            + "_"
            + np.round(res_df["lon"], 3).astype(str)
        )

        # Store canonical ID and alias metadata
        res_df.attrs["station_aliases"] = self.station_alias_map
        res_df["ghcn_id"] = res_df["station"]

        # If user queried using specific aliases (e.g. HKJK, HKNW), map station label accordingly
        if self.station_alias_map:
            res_df["station"] = res_df["station"].apply(lambda s: self.station_alias_map.get(s, s))

        if format.lower() == "wide":
            return to_wide(res_df, column_key=column_key)
        return res_df

    def fetch_sync(
        self,
        variables: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        format: str = "long",
        column_key: str = "station",
        apply_qc: Optional[bool] = None,
        access_method: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._run_async(
            self.fetch(
                variables,
                start_date,
                end_date,
                format=format,
                column_key=column_key,
                apply_qc=apply_qc,
                access_method=access_method,
            )
        )

    def __call__(
        self,
        variables: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        format: str = "long",
        column_key: str = "station",
        apply_qc: Optional[bool] = None,
        access_method: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.fetch_sync(
            variables,
            start_date,
            end_date,
            format=format,
            column_key=column_key,
            apply_qc=apply_qc,
            access_method=access_method,
        )


# =====================================================================
# 7. Geostatistical, Wide Reshaping, and Alignment Helpers
# =====================================================================

def haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Computes pairwise great-circle distance between coordinates in kilometers."""
    R = 6371.0  # Earth's radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def to_wide(
    df: pd.DataFrame,
    index_col: str = "time",
    column_key: str = "station",
    value_col: str = "observation",
    aggfunc: Any = "mean",
) -> pd.DataFrame:
    """
    Converts a long observation DataFrame into a wide pivot table
    indexed by timestamp and columnized by station or lat_lon.
    """
    if df.empty:
        return pd.DataFrame()

    df_clean = df.copy()
    if index_col not in df_clean.columns and "date" in df_clean.columns:
        index_col = "date"
    df_clean[index_col] = pd.to_datetime(df_clean[index_col])

    col_target = "lat_lon" if column_key == "lat_lon" and "lat_lon" in df_clean.columns else "station"
    if col_target not in df_clean.columns and "station_id" in df_clean.columns:
        col_target = "station_id"

    if isinstance(aggfunc, dict):
        res_dict = {}
        for var_name, func in aggfunc.items():
            sub_df = df_clean[df_clean["variable"] == var_name] if "variable" in df_clean.columns else df_clean
            if not sub_df.empty:
                res_dict[var_name] = to_wide(
                    sub_df, index_col=index_col, column_key=column_key, value_col=value_col, aggfunc=func
                )
            else:
                res_dict[var_name] = pd.DataFrame()
        return res_dict

    if "variable" in df_clean.columns and df_clean["variable"].nunique() > 1:
        wide = df_clean.pivot_table(
            index=index_col,
            columns=[col_target, "variable"],
            values=value_col,
            aggfunc=aggfunc,
        )
    else:
        wide = df_clean.pivot_table(
            index=index_col,
            columns=col_target,
            values=value_col if value_col in df_clean.columns else df_clean.columns[-1],
            aggfunc=aggfunc,
        )
    return wide


def align_to_reference_grid(
    source_df: pd.DataFrame,
    target_meta: pd.DataFrame,
    target_name: str = "reference",
    metric: str = "haversine",
) -> pd.DataFrame:
    """
    Spatially align source stations to reference stations (e.g. TAHMO) or gridded models (ERA5).
    """
    def _find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    source_lat_col = _find_col(source_df, ["latitude", "LAT", "lat"])
    source_lon_col = _find_col(source_df, ["longitude", "LON", "lon"])
    source_id_col = _find_col(source_df, ["station", "station_id", "ID", "GHCN_ID"])

    target_lat_col = _find_col(target_meta, ["latitude", "LAT", "lat"])
    target_lon_col = _find_col(target_meta, ["longitude", "LON", "lon"])
    target_id_col = _find_col(target_meta, ["lat_lon_label", "station", "station_id", "ID", "grid_id"])

    if not all([source_lat_col, source_lon_col, target_lat_col, target_lon_col]):
        raise ValueError("Both source and target dataframes must contain latitude and longitude columns.")

    src_clean = source_df.dropna(subset=[source_lat_col, source_lon_col]).copy()
    tgt_clean = target_meta.dropna(subset=[target_lat_col, target_lon_col]).copy()

    dist_col_name = "distance_km" if metric == "haversine" else "distance_deg"

    if src_clean.empty or tgt_clean.empty:
        return pd.DataFrame(columns=[
            "ghcn_station", "lat", "lon", f"{target_name}_nearest_label", dist_col_name
        ])

    src_lat = src_clean[source_lat_col].to_numpy(dtype=float)
    src_lon = src_clean[source_lon_col].to_numpy(dtype=float)
    tgt_lat = tgt_clean[target_lat_col].to_numpy(dtype=float)
    tgt_lon = tgt_clean[target_lon_col].to_numpy(dtype=float)

    if metric == "haversine":
        dist_matrix = haversine_distance(src_lat[:, None], src_lon[:, None], tgt_lat[None, :], tgt_lon[None, :])
    else:
        source_coords = np.column_stack([src_lat, src_lon])
        target_coords = np.column_stack([tgt_lat, tgt_lon])
        dist_matrix = cdist(source_coords, target_coords)

    nearest_idx = np.argmin(dist_matrix, axis=1)

    results = []
    tgt_labels = tgt_clean[target_id_col].to_numpy() if target_id_col else np.arange(len(tgt_clean))
    for i, (_, row) in enumerate(src_clean.iterrows()):
        best_match = nearest_idx[i]
        results.append({
            "ghcn_station": row[source_id_col] if source_id_col else f"ST_{i}",
            "lat": row[source_lat_col],
            "lon": row[source_lon_col],
            f"{target_name}_nearest_label": tgt_labels[best_match],
            dist_col_name: float(dist_matrix[i, best_match]),
        })

    return pd.DataFrame(results)


def find_nearest_stations(
    source_stations_df: pd.DataFrame,
    target_metadata_df: pd.DataFrame,
    target_name: str = "tahmo",
) -> pd.DataFrame:
    """Direct drop-in replacement for nearest stations helper."""
    return align_to_reference_grid(
        source_df=source_stations_df,
        target_meta=target_metadata_df,
        target_name=target_name,
        metric="haversine",
    )


def find_nearest_grid_point(
    source_lat: float,
    source_lon: float,
    gridded_df: pd.DataFrame,
    product_name: str = "gridded",
) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Finds the grid point closest to source_lat, source_lon.
    Assumes gridded_df has columns formatted as 'latitude_longitude' and date as index.
    """
    min_dist = float("inf")
    nearest_col = None

    for col in gridded_df.columns:
        try:
            parts = str(col).split("_")
            if len(parts) >= 2:
                grid_lat, grid_lon = float(parts[0]), float(parts[1])
                dist = (source_lat - grid_lat)**2 + (source_lon - grid_lon)**2
                if dist < min_dist:
                    min_dist = dist
                    nearest_col = col
        except ValueError:
            continue

    if nearest_col is not None:
        return gridded_df[nearest_col], nearest_col
    return None, None


# =====================================================================
# 8. Unified Factory & High-Level Convenience Query Functions
# =====================================================================

def GHCN(
    resolution: str = "daily",
    stations: Optional[List[str]] = None,
    region: Optional[str] = None,
    countries: Optional[Union[str, List[str]]] = None,
    country_codes: Optional[Union[str, List[str]]] = None,
    bounds: Optional[Union[Dict[str, float], Tuple[float, float, float, float]]] = None,
    cache_dir: str = "./ghcn_cache",
    fetch_mode: str = "auto",
    apply_qc: bool = True,
    **kwargs,
) -> Union[GHCNDaily, GHCNHourly]:
    """
    Factory creating a Daily or Hourly GHCN pipeline.
    """
    if resolution.lower().startswith("hour"):
        return GHCNHourly(
            stations=stations,
            region=region,
            countries=countries,
            country_codes=country_codes,
            bounds=bounds,
            cache_dir=cache_dir,
            fetch_mode=fetch_mode,
            apply_qc=apply_qc,
            **kwargs,
        )
    return GHCNDaily(
        stations=stations,
        region=region,
        countries=countries,
        country_codes=country_codes,
        bounds=bounds,
        cache_dir=cache_dir,
        fetch_mode=fetch_mode,
        apply_qc=apply_qc,
        **kwargs,
    )


def query_ghcn_daily(
    region: Optional[str] = None,
    stations: Optional[List[str]] = None,
    countries: Optional[Union[str, List[str]]] = None,
    variables: Optional[List[str]] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    format: str = "wide",
    column_key: str = "station",
    apply_qc: bool = True,
    cache_dir: str = "./ghcn_cache",
    bounds: Optional[Union[Dict[str, float], Tuple[float, float, float, float]]] = None,
    active_only: bool = False,
    min_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    High-level, user-friendly function to query cleaned NOAA GHCN-Daily data in one line.

    Parameters
    ----------
    region : str, optional
        Regional preset ('kenya', 'east_africa', 'west_africa', etc.). Defaults to 'kenya'
        if no stations, countries, or bounds are specified.
    stations : list of str, optional
        Explicit station IDs.
    countries : list of str, optional
        Country names to search (e.g. ['Nigeria', 'Kenya']).
    variables : list of str, optional
        Meteorological fields (e.g. ['tp', 't2m_max', 't2m_min', 't2m', 'rain']).
    start_date : str or datetime, optional
        Start date ('YYYY-MM-DD'). Default '2020-01-01'.
    end_date : str or datetime, optional
        End date ('YYYY-MM-DD'). Default today.
    format : str, default 'wide'
        'wide' (Date x Stations matrix) or 'long' (Earth2Studio schema).
    column_key : str, default 'station'
        Column headers in wide format ('station' or 'lat_lon').
    apply_qc : bool, default True
        Automatically clean physical impossibilities and missing value sentinels.
    active_only : bool, default False
        If True, only queries stations with active telemetry (2020 or later).
    min_year : int, optional
        Filter stations with observations recorded in or after this year.

    Examples
    --------
    >>> import masharikiweather as mw
    >>> df = mw.query_ghcn_daily(countries='Nigeria', variables=['rain'], start_date='2024-07-01')
    """
    if stations is None and region is None and countries is None and bounds is None:
        region = "kenya"

    vars_to_fetch = variables or ["tp", "t2m_max", "t2m_min"]
    pipeline = GHCNDaily(
        stations=stations,
        region=region,
        countries=countries,
        bounds=bounds,
        cache_dir=cache_dir,
        apply_qc=apply_qc,
        active_only=active_only,
        min_year=min_year,
    )

    # Validate observation lifecycle date ranges & variable coverage
    cov_df = pd.DataFrame()
    try:
        cov_df = pipeline.validate_station_coverage(
            variables=vars_to_fetch,
            start_date=start_date,
            end_date=end_date,
        )
        invalid_st = cov_df[~cov_df["is_reporting_in_window"] | ~cov_df["has_requested_elements"]]
        for _, row in invalid_st.iterrows():
            print(
                f"[GHCN-Daily Notice] Station '{row['station']}' ({row.get('name', '')}): "
                f"{row['validity_status']}. Operational window: {row['data_coverage']} "
                f"[{row['elements_available']}]. Requested: {start_date or '1980-01-01'} to {end_date or 'today'}."
            )
    except Exception:
        pass

    # Validate independent variable activeness lifecycle
    var_lifecycle_df = pd.DataFrame()
    try:
        var_lifecycle_df = pipeline.validate_variable_activeness(
            variables=vars_to_fetch,
            start_date=start_date,
            end_date=end_date,
        )
        problem_vars = var_lifecycle_df[~var_lifecycle_df["is_reporting_in_window"]]
        if not problem_vars.empty:
            for st, st_group in problem_vars.groupby("station"):
                st_name = st_group["name"].iloc[0]
                lines = [f"  • '{r['variable']}': {r['query_status']}" for _, r in st_group.iterrows()]
                print(
                    f"[GHCN Sensor Notice] Station '{st}' ({st_name}):\n"
                    + "\n".join(lines)
                )
    except Exception:
        pass

    res = pipeline(
        variables=vars_to_fetch,
        start_date=start_date,
        end_date=end_date,
        format=format,
        column_key=column_key,
        apply_qc=apply_qc,
    )

    # Attach rich station coverage report to DataFrame attributes (JSON-serializable for Parquet)
    if not cov_df.empty:
        res.attrs["station_coverage"] = cov_df.to_dict(orient="records")
        try:
            st_meta_df = pipeline.get_stations_dataframe(include_inventory=True)
            res.attrs["stations_metadata"] = st_meta_df.to_dict(orient="records")
        except Exception:
            pass

    # Enrich variable lifecycle with actual returned data observations and last date
    if not var_lifecycle_df.empty:
        try:
            enriched_rows = []
            for _, r in var_lifecycle_df.iterrows():
                st = r["station"]
                var = r["variable"]
                s = get_station_series(res, station=st, variable=var)
                rec_count = len(s)
                last_date = s.index.max().strftime("%Y-%m-%d") if rec_count > 0 else "None"
                r_dict = dict(r)
                r_dict["records_returned"] = rec_count
                r_dict["last_recorded_date"] = last_date
                enriched_rows.append(r_dict)
            res.attrs["variable_lifecycle"] = enriched_rows
        except Exception:
            res.attrs["variable_lifecycle"] = var_lifecycle_df.to_dict(orient="records")

    return res


def check_station_coverage(
    stations: List[str],
    variables: Optional[List[str]] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    cache_dir: str = "./ghcn_cache",
) -> pd.DataFrame:
    """
    Check operational observation date ranges and element availability for target stations.
    Returns a diagnostics DataFrame with validity status, operational windows, and recommendations.
    """
    pipeline = GHCNDaily(stations=stations, cache_dir=cache_dir)
    return pipeline.validate_station_coverage(variables=variables, start_date=start_date, end_date=end_date)


def check_variable_activeness(
    region: Optional[str] = None,
    countries: Optional[Union[str, List[str]]] = None,
    stations: Optional[List[str]] = None,
    cache_dir: str = "./ghcn_cache",
) -> pd.DataFrame:
    """
    Check independent operational status and date windows for every meteorological
    variable (rain, tmax, tmin, tavg) across stations in a country or region.
    """
    return GHCNDaily.get_stations_variable_activeness(
        region=region,
        countries=countries,
        stations=stations,
        cache_dir=cache_dir,
    )


def get_stations_variable_activeness(
    region: Optional[str] = None,
    countries: Optional[Union[str, List[str]]] = None,
    stations: Optional[List[str]] = None,
    cache_dir: str = "./ghcn_cache",
) -> pd.DataFrame:
    """
    Convenience alias for check_variable_activeness.
    """
    return GHCNDaily.get_stations_variable_activeness(
        region=region,
        countries=countries,
        stations=stations,
        cache_dir=cache_dir,
    )


def get_variable_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieve the independent variable lifecycle and sensor operational status table
    from a GHCN DataFrame. Shows which sensors are active vs ended and their exact last dates.
    """
    if df is not None and "variable_lifecycle" in getattr(df, "attrs", {}):
        vl = df.attrs["variable_lifecycle"]
        if isinstance(vl, pd.DataFrame):
            return vl
        if isinstance(vl, list):
            return pd.DataFrame(vl)
    return pd.DataFrame()


def get_station_series(
    df: pd.DataFrame,
    station: str,
    variable: Optional[str] = None,
) -> pd.Series:
    """
    Safely extract a 1D observation Series for a station and variable from a GHCN DataFrame.
    Guarantees no KeyError if a station was decommissioned or not present in results.
    Supports station aliases (e.g. matching 'HKJK' to 'KEI0000HKJK').

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by query_ghcn_daily or query_ghcn_hourly (wide or long format).
    station : str
        Target station ID or alias (case-insensitive, e.g. 'HKJK', 'KEI0000HKJK', 'NIM00065201', 'BC000068032').
    variable : str, optional
        Target variable name (e.g. 'rain', 'prcp', 'tp', 'tmax', 'tmin', 't2m', 'rh').

    Returns
    -------
    pd.Series
        Cleaned, non-empty pandas Series indexed by DatetimeIndex, or an empty Series if unavailable.
    """
    st_clean = str(station).strip().upper()
    aliases: Set[str] = {st_clean}

    # Inspect station alias metadata dictionary attached to DataFrame
    if df is not None and "station_aliases" in getattr(df, "attrs", {}):
        alias_map = df.attrs["station_aliases"]
        if isinstance(alias_map, dict):
            if st_clean in alias_map:
                aliases.add(str(alias_map[st_clean]).upper())
            for k, v in alias_map.items():
                if str(v).upper() == st_clean:
                    aliases.add(str(k).upper())

    res_s = None

    if df is None or df.empty:
        res_s = pd.Series(dtype=float, name=f"{st_clean}_{variable}" if variable else st_clean)
    elif isinstance(df.columns, pd.MultiIndex):
        matched_stations = [c for c in df.columns.levels[0] if str(c).upper() in aliases]
        if not matched_stations:
            res_s = pd.Series(dtype=float, name=f"{st_clean}_{variable}" if variable else st_clean)
        else:
            st_sub = df[matched_stations[0]]
            if variable:
                try:
                    _, _, can_var = resolve_daily_variable(variable)
                except Exception:
                    can_var = str(variable).lower()

                target_cols = [c for c in st_sub.columns if str(c).lower() in (str(variable).lower(), can_var)]
                if target_cols:
                    res_s = st_sub[target_cols[0]].dropna()
                elif st_sub.shape[1] == 1:
                    res_s = st_sub.iloc[:, 0].dropna()
                else:
                    res_s = pd.Series(dtype=float, name=f"{st_clean}_{variable}" if variable else st_clean)
            else:
                res_s = st_sub.iloc[:, 0].dropna()
    else:
        st_matches = [c for c in df.columns if str(c).upper() in aliases]
        if st_matches:
            res_s = df[st_matches[0]].dropna()
        else:
            st_col = "station" if "station" in df.columns else ("station_id" if "station_id" in df.columns else None)
            if st_col and "observation" in df.columns:
                sub = df[df[st_col].astype(str).str.upper().isin(aliases)]
                if sub.empty and "ghcn_id" in df.columns:
                    sub = df[df["ghcn_id"].astype(str).str.upper().isin(aliases)]

                if variable and "variable" in sub.columns:
                    try:
                        _, _, can_var = resolve_daily_variable(variable)
                    except Exception:
                        can_var = str(variable).lower()
                    sub = sub[sub["variable"].str.lower().isin([str(variable).lower(), can_var])]

                time_col = "time" if "time" in sub.columns else ("date" if "date" in sub.columns else None)
                if time_col and not sub.empty:
                    res_s = pd.Series(
                        sub["observation"].values,
                        index=pd.to_datetime(sub[time_col]),
                        name=f"{st_clean}_{variable}" if variable else st_clean,
                    ).dropna()

    if res_s is None:
        res_s = pd.Series(dtype=float, name=f"{st_clean}_{variable}" if variable else st_clean)

    # Attach rich variable lifecycle metadata attributes
    if df is not None and "variable_lifecycle" in getattr(df, "attrs", {}):
        vl = df.attrs["variable_lifecycle"]
        if isinstance(vl, list):
            vl = pd.DataFrame(vl)
        if isinstance(vl, pd.DataFrame) and not vl.empty:
            sub_vl = vl[vl["station"].astype(str).str.upper().isin(aliases)]
            if variable:
                try:
                    _, _, can_v = resolve_daily_variable(variable)
                except Exception:
                    can_v = str(variable).lower()
                sub_vl = sub_vl[sub_vl["variable"].str.lower().isin([str(variable).lower(), can_v])]
            if not sub_vl.empty:
                v_row = sub_vl.iloc[0]
                res_s.attrs["variable_status"] = v_row.get("sensor_lifecycle", "Unknown")
                res_s.attrs["query_status"] = v_row.get("query_status", "Unknown")
                res_s.attrs["coverage_window"] = v_row.get("coverage_window", "Unknown")
                res_s.attrs["last_recorded_date"] = v_row.get("last_recorded_date", "None")

    return res_s


def query_ghcn_hourly(
    stations: Optional[List[str]] = None,
    region: Optional[str] = None,
    countries: Optional[Union[str, List[str]]] = None,
    variables: Optional[List[str]] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    format: str = "wide",
    column_key: str = "station",
    apply_qc: bool = True,
    access_method: str = "auto",
    fetch_mode: str = "auto",
    cache_dir: str = "./ghcn_cache",
    **kwargs,
) -> pd.DataFrame:
    """
    High-level, user-friendly function to query cleaned NOAA GHCN-Hourly (GHCNh) data.

    Parameters
    ----------
    stations : list of str, optional
        Hourly station IDs or airport ICAO codes (e.g. ['HKJK', 'HKNW'] or ['KEI0000HKJK', 'KEI0000HKNW']).
    region : str, optional
        Region to discover hourly stations in (e.g. 'kenya', 'east_africa').
    countries : list of str, optional
        Country names or ISO codes to discover stations in.
    variables : list of str, optional
        Target fields: ['t2m', 'dew_point', 'rh', 'ws10m', 'wd', 'tp'].
    start_date : str or datetime, optional
        Start date. Default '2024-01-01'.
    end_date : str or datetime, optional
        End date.
    format : str, default 'wide'
        'wide' (Date x Stations matrix) or 'long' (standard spatiotemporal table).
    column_key : str, default 'station'
        Header name in wide format ('station' or 'lat_lon').
    apply_qc : bool, default True
        Automatically filter negative tropical temperature anomalies, impossible dew points, and wind spikes.
    access_method : str, default 'auto'
        NCEI ingestion method:
        - 'auto': by-year Parquet with PSV and by-station fallbacks
        - 'by_year' / 'by_year_parquet': NOAA annual Parquet slices
        - 'by_year_psv': NOAA annual pipe-separated values
        - 'by_station' / 'by_station_psv': NOAA full period-of-record (POR)
    cache_dir : str, default './ghcn_cache'
        Local directory for caching downloaded NOAA archives.
    """
    if stations is None and region is None and countries is None:
        region = "kenya"

    vars_to_fetch = variables or ["t2m", "ws10m", "dew_point"]
    pipeline = GHCNHourly(
        stations=stations,
        region=region,
        countries=countries,
        cache_dir=cache_dir,
        apply_qc=apply_qc,
        access_method=access_method,
        fetch_mode=fetch_mode,
        **kwargs,
    )
    return pipeline(
        variables=vars_to_fetch,
        start_date=start_date,
        end_date=end_date,
        format=format,
        column_key=column_key,
        apply_qc=apply_qc,
        access_method=access_method,
    )


def check_ghcn_availability(
    resolution: str = "daily",
    region: Optional[str] = None,
    stations: Optional[List[str]] = None,
    countries: Optional[Union[str, List[str]]] = None,
    variables: Optional[List[str]] = None,
    cache_dir: str = "./ghcn_cache",
) -> pd.DataFrame:
    """
    Diagnostics function checking observation date ranges, record counts, and latest active status.
    """
    if stations is None and region is None and countries is None:
        region = "kenya"
    pipeline = GHCN(resolution=resolution, region=region, stations=stations, countries=countries, cache_dir=cache_dir)
    if hasattr(pipeline, "validate_station_coverage"):
        return pipeline.validate_station_coverage(variables=variables)
    return pipeline.get_stations_dataframe()


def get_stations(
    resolution: str = "daily",
    region: Optional[str] = None,
    countries: Optional[Union[str, List[str]]] = None,
    country_codes: Optional[Union[str, List[str]]] = None,
    bounds: Optional[Union[Dict[str, float], Tuple[float, float, float, float]]] = None,
    cache_dir: str = "./ghcn_cache",
    include_inventory: bool = True,
    min_year: Optional[int] = None,
    active_only: bool = False,
) -> pd.DataFrame:
    """
    Dynamically discover weather stations anywhere in the world or by region/country.
    Defaults to African coverage when no location parameters are specified, while fully
    supporting global discovery for any country (e.g. 'Netherlands', 'Germany', 'USA'),
    continent presets ('europe', 'north_america', 'south_america', 'asia', 'oceania', 'africa'),
    or bounding box coordinates worldwide.

    Parameters
    ----------
    resolution : str, default 'daily'
        'daily' (GHCN-Daily) or 'hourly' (GHCN-Hourly).
    region : str, optional
        Preset region or continent ('kenya', 'east_africa', 'africa', 'europe', 'north_america', 'asia', etc.).
    countries : str or list of str, optional
        Country name(s) (e.g. 'Netherlands', 'Kenya', ['Germany', 'France']).
    country_codes : str or list of str, optional
        NOAA 2-letter country code(s) (e.g. 'NL', 'GM', 'KE').
    bounds : dict or tuple, optional
        Bounding box coordinates (min_lat, min_lon, max_lat, max_lon) or dict with keys.
    cache_dir : str, default './ghcn_cache'
        Local directory for metadata and station list caching.
    include_inventory : bool, default True
        If True, enriches daily stations with observation lifecycle (active status, start/end years).
    min_year : int, optional
        Only return stations active in or after this year (e.g. 2020).
    active_only : bool, default False
        Filter to actively reporting stations.

    Returns
    -------
    pd.DataFrame
        DataFrame of matching stations with metadata and operational coordinates.
    """
    res = str(resolution).lower().strip()
    if res in ("hourly", "ghcnh"):
        return GHCNHourly.get_stations(
            region=region,
            countries=countries,
            country_codes=country_codes,
            bounds=bounds,
            cache_dir=cache_dir,
            active_only=active_only,
            min_year=min_year,
        )
    return GHCNDaily.get_stations(
        region=region,
        countries=countries,
        country_codes=country_codes,
        bounds=bounds,
        cache_dir=cache_dir,
        include_inventory=include_inventory,
        min_year=min_year,
        active_only=active_only,
    )


def get_african_stations(
    resolution: str = "daily",
    region: Optional[str] = None,
    countries: Optional[Union[str, List[str]]] = None,
    country_codes: Optional[Union[str, List[str]]] = None,
    bounds: Optional[Union[Dict[str, float], Tuple[float, float, float, float]]] = None,
    cache_dir: str = "./ghcn_cache",
    include_inventory: bool = True,
    min_year: Optional[int] = None,
    active_only: bool = False,
) -> pd.DataFrame:
    """
    Discover African weather stations across East, West, North, Southern, and Central Africa.
    Defaults to all Africa if no regional filters are provided.
    """
    return get_stations(
        resolution=resolution,
        region=region or "africa",
        countries=countries,
        country_codes=country_codes,
        bounds=bounds,
        cache_dir=cache_dir,
        include_inventory=include_inventory,
        min_year=min_year,
        active_only=active_only,
    )



