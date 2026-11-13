"""
EUMETSAT CM SAF SARAH-3 Geostationary Surface Solar Radiation API Wrapper for MasharikiWeather.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_CACHE_DIR,
    _BaseDatasetWrapper,
    _normalize_locations,
    _parse_date,
    _safe_request,
)

logger = logging.getLogger("MasharikiWeather.SARAH3")

# ==============================================================================
# Parameter Definitions & Metadata Dictionary
# ==============================================================================

SARAH3_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "shortwave_radiation": {
        "api_name": "shortwave_radiation",
        "aliases": ["ghi", "sis", "solar_rad", "global_horizontal", "shortwave"],
        "unit": "W/m²",
        "category": "Solar Radiation",
        "description": "Surface Incoming Shortwave Radiation (Global Horizontal Irradiance, GHI). Total hemispherical solar flux reaching the surface.",
    },
    "direct_radiation": {
        "api_name": "direct_radiation",
        "aliases": ["direct", "sid", "beam_horizontal", "direct_horizontal"],
        "unit": "W/m²",
        "category": "Solar Radiation",
        "description": "Surface Incoming Direct Radiation (Direct Horizontal Irradiance, BHI). Direct beam solar flux projected onto a horizontal surface.",
    },
    "diffuse_radiation": {
        "api_name": "diffuse_radiation",
        "aliases": ["diffuse", "dhi", "diffuse_horizontal"],
        "unit": "W/m²",
        "category": "Solar Radiation",
        "description": "Surface Diffuse Horizontal Irradiance (DHI). Solar radiation scattered by atmospheric molecules, aerosols, and clouds.",
    },
    "direct_normal_irradiance": {
        "api_name": "direct_normal_irradiance",
        "aliases": ["dni", "direct_normal", "beam_normal"],
        "unit": "W/m²",
        "category": "Solar Radiation",
        "description": "Direct Normal Irradiance (DNI). Solar flux per unit area on a surface perpendicular to the sun's rays. Crucial for CSP and solar trackers.",
    },
    "terrestrial_radiation": {
        "api_name": "terrestrial_radiation",
        "aliases": ["toa", "extraterrestrial", "g0", "top_of_atmosphere"],
        "unit": "W/m²",
        "category": "Extraterrestrial Solar",
        "description": "Extraterrestrial horizontal solar irradiance at the top of the atmosphere (TOA). Theoretical maximum solar flux prior to atmospheric attenuation.",
    },
    "clearness_index": {
        "api_name": "clearness_index",
        "aliases": ["kt", "cloud_clearness"],
        "unit": "dimensionless (0–1)",
        "category": "Derived Atmospheric Metric",
        "description": "Atmospheric Clearness Index (Kt = GHI / TOA). Measures relative atmospheric transparency and cloud attenuation.",
    },
    "diffuse_fraction": {
        "api_name": "diffuse_fraction",
        "aliases": ["kd", "diffuse_ratio"],
        "unit": "dimensionless (0–1)",
        "category": "Derived Atmospheric Metric",
        "description": "Diffuse Fraction (Kd = DHI / GHI). Ratio of scattered diffuse radiation to total global horizontal radiation.",
    },
}

# Mapping table from user aliases to canonical Open-Meteo API parameter names
SARAH3_LEXICON: Dict[str, str] = {}
for param_key, meta in SARAH3_PARAMETERS.items():
    api_name = meta["api_name"]
    SARAH3_LEXICON[param_key] = api_name
    SARAH3_LEXICON[api_name] = api_name
    for alias in meta.get("aliases", []):
        SARAH3_LEXICON[alias.lower().strip()] = api_name

DEFAULT_SARAH3_VARS = ["ghi", "direct", "diffuse", "dni", "terrestrial_radiation"]


def search_sarah3_parameters(query: Optional[str] = None) -> pd.DataFrame:
    """
    Search available CM SAF SARAH-3 parameters by keyword or category.

    Parameters
    ----------
    query : str, optional
        Keyword to match against parameter name, aliases, or description.
        If None, returns all available parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Variable', 'API Name', 'Category', 'Unit', 'Aliases', 'Description'].
    """
    rows = []
    q = query.lower().strip() if query else None

    for name, meta in SARAH3_PARAMETERS.items():
        aliases_str = ", ".join(meta.get("aliases", []))
        desc = meta.get("description", "")
        cat = meta.get("category", "")
        api_n = meta.get("api_name", name)

        if q:
            match = (
                q in name.lower()
                or q in api_n.lower()
                or q in aliases_str.lower()
                or q in desc.lower()
                or q in cat.lower()
            )
            if not match:
                continue

        rows.append({
            "Variable": name,
            "API Name": api_n,
            "Category": cat,
            "Unit": meta.get("unit", ""),
            "Aliases": aliases_str,
            "Description": desc,
        })

    return pd.DataFrame(rows)


def list_sarah3_parameters(category: Optional[str] = None) -> pd.DataFrame:
    """
    List all SARAH-3 parameters, optionally filtered by category.
    """
    df = search_sarah3_parameters(query=None)
    if category:
        cat_lower = category.lower().strip()
        df = df[df["Category"].str.lower().str.contains(cat_lower)]
    return df.reset_index(drop=True)


def get_sarah3_metadata() -> Dict[str, Any]:
    """
    Retrieve comprehensive scientific metadata regarding the EUMETSAT CM SAF SARAH-3 climate record.
    """
    return {
        "dataset_name": "Surface Solar Radiation Data Set - Heliosat (SARAH-3)",
        "agency": "EUMETSAT Satellite Application Facility on Climate Monitoring (CM SAF)",
        "host_institute": "Deutscher Wetterdienst (DWD, Germany)",
        "satellites": [
            "Meteosat-2 to Meteosat-7 (Meteosat First Generation, MFG MVIRI, 1983–2005)",
            "Meteosat-8 to Meteosat-11 (Meteosat Second Generation, MSG SEVIRI, 2004–present)",
        ],
        "spatial_resolution": "0.05° x 0.05° (~5.5 km at equator)",
        "spatial_coverage": "Europe, Africa, Atlantic Ocean, Middle East (-65° to +65° lat/lon)",
        "temporal_resolution": "Hourly (continuous 24-hr profile), daily, and monthly averages",
        "temporal_coverage": "1983 through present (40+ continuous years)",
        "api_service": "Open-Meteo EUMETSAT Satellite Archive (models=eumetsat_sarah3)",
        "api_endpoint": "https://satellite-api.open-meteo.com/v1/archive",
        "authentication": "Zero-Key (Free and open public REST API)",
        "licensing": "EUMETSAT / CM SAF Open Data Policy (free for academic, scientific, and commercial use)",
        "citation": "Pfeifroth, U., Trentmann, J., Kothe, S., et al. (2023). Surface Radiation Data Set - Heliosat, Edition 3 (SARAH-3). Satellite Application Facility on Climate Monitoring. DOI: 10.5676/EUM_SAF_CM/SARAH/V003.",
    }


# ==============================================================================
# SARAH-3 Wrapper Client
# ==============================================================================

class SARAH3(_BaseDatasetWrapper):
    """
    Client for querying EUMETSAT CM SAF SARAH-3 geostationary satellite surface solar radiation.
    """

    API_URL = "https://satellite-api.open-meteo.com/v1/archive"

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        rate_limit_delay: float = 0.2,
        max_workers: int = 4,
    ) -> None:
        super().__init__(
            cache_dir=cache_dir,
            rate_limit_delay=rate_limit_delay,
            max_workers=max_workers,
        )

    def fetch_point(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None,
        location_name: Optional[str] = None,
        compute_derived_metrics: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Fetch SARAH-3 hourly time series for a single (lat, lon) coordinate.

        Parameters
        ----------
        lat : float
            Latitude in degrees (-65 to 65).
        lon : float
            Longitude in degrees (-65 to 65).
        start_date : str
            Start date string ('YYYY-MM-DD').
        end_date : str
            End date string ('YYYY-MM-DD').
        variables : list of str, optional
            Requested variables or aliases.
        location_name : str, optional
            Human-readable label.
        compute_derived_metrics : bool, default True
            If True, automatically calculates Clearness Index (Kt) and Diffuse Fraction (Kd).

        Returns
        -------
        list of dict
            Standard observation records.
        """
        loc_label = location_name or f"{lat:.3f}_{lon:.3f}"
        if variables == "all" or (isinstance(variables, (list, tuple)) and "all" in variables):
            vars_requested = list(SARAH3_PARAMETERS.keys())
        else:
            vars_requested = variables or DEFAULT_SARAH3_VARS

        api_vars: List[str] = []
        var_to_canon: Dict[str, str] = {}
        include_kt = False
        include_kd = False

        for v in vars_requested:
            canon = v.lower().strip()
            if canon in ("clearness_index", "kt"):
                include_kt = True
                continue
            if canon in ("diffuse_fraction", "kd"):
                include_kd = True
                continue

            mapped = SARAH3_LEXICON.get(canon, canon)
            if mapped in ("clearness_index", "diffuse_fraction"):
                continue

            api_vars.append(mapped)
            var_to_canon[mapped] = canon

        # If derived metrics are requested, ensure necessary inputs are queried
        if (include_kt or compute_derived_metrics) and "terrestrial_radiation" not in api_vars:
            api_vars.append("terrestrial_radiation")
            var_to_canon["terrestrial_radiation"] = "terrestrial_radiation"
        if (include_kd or compute_derived_metrics) and "diffuse_radiation" not in api_vars:
            api_vars.append("diffuse_radiation")
            var_to_canon["diffuse_radiation"] = "diffuse"
        if (include_kt or include_kd or compute_derived_metrics) and "shortwave_radiation" not in api_vars:
            api_vars.append("shortwave_radiation")
            var_to_canon["shortwave_radiation"] = "ghi"

        api_vars = sorted(list(set(api_vars)))

        # Cache key construction
        cache_key = {
            "source": "sarah3",
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "start": start_date,
            "end": end_date,
            "vars": api_vars,
            "derived": compute_derived_metrics or include_kt or include_kd,
        }
        cache_file = self._get_cache_path("sarah3", cache_key)
        if self._is_cache_valid(cache_file):
            try:
                cached_df = pd.read_parquet(cache_file)
                return cached_df.to_dict(orient="records")
            except Exception:
                pass

        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(api_vars),
            "models": "eumetsat_sarah3",
            "timezone": "UTC",
        }
        url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"

        time.sleep(self.rate_limit_delay)
        raw_bytes = _safe_request(url)
        data = json.loads(raw_bytes.decode("utf-8"))

        records: List[Dict[str, Any]] = []
        hourly_data = data.get("hourly", {})
        times = pd.to_datetime(hourly_data.get("time", []), utc=True)

        ghi_vals = hourly_data.get("shortwave_radiation", [])
        diff_vals = hourly_data.get("diffuse_radiation", [])
        toa_vals = hourly_data.get("terrestrial_radiation", [])

        for api_var in api_vars:
            canon_name = var_to_canon.get(api_var, api_var)
            vals = hourly_data.get(api_var, [])
            for ts, val in zip(times, vals):
                if val is not None and not pd.isna(val):
                    records.append({
                        "time": ts,
                        "location": loc_label,
                        "lat": lat,
                        "lon": lon,
                        "variable": canon_name,
                        "value": float(val),
                    })

        # Calculate derived metrics: Clearness Index (Kt) and Diffuse Fraction (Kd)
        if compute_derived_metrics or include_kt:
            for ts, g, t_toa in zip(times, ghi_vals, toa_vals):
                if g is not None and t_toa is not None and t_toa > 5.0:
                    kt = float(np.clip(g / t_toa, 0.0, 1.0))
                    records.append({
                        "time": ts,
                        "location": loc_label,
                        "lat": lat,
                        "lon": lon,
                        "variable": "clearness_index",
                        "value": round(kt, 4),
                    })

        if compute_derived_metrics or include_kd:
            for ts, g, d in zip(times, ghi_vals, diff_vals):
                if g is not None and d is not None and g > 5.0:
                    kd = float(np.clip(d / g, 0.0, 1.0))
                    records.append({
                        "time": ts,
                        "location": loc_label,
                        "lat": lat,
                        "lon": lon,
                        "variable": "diffuse_fraction",
                        "value": round(kd, 4),
                    })

        if records:
            try:
                pd.DataFrame(records).to_parquet(cache_file, index=False)
            except Exception:
                pass

        return records

    def query(
        self,
        locations: Optional[Union[Dict[str, float], Tuple[float, float], List[Any], str]] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        region: Optional[str] = None,
        variables: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        compute_derived_metrics: bool = True,
        format: str = "wide",
    ) -> pd.DataFrame:
        """
        Query SARAH-3 solar irradiance data across one or more locations.

        Parameters
        ----------
        locations : dict, tuple, list, or str, optional
            Target location or locations.
        lat, lon : float, optional
            Direct coordinates.
        region : str, optional
            Preset region name (e.g. 'kenya').
        variables : list of str, optional
            Target variables: ['ghi', 'direct', 'diffuse', 'dni', 'terrestrial_radiation', 'clearness_index'].
        start_date, end_date : str or datetime, optional
            Date range ('YYYY-MM-DD'). Defaults to a 7-day window.
        compute_derived_metrics : bool, default True
            If True, calculates Clearness Index (Kt) and Diffuse Fraction (Kd).
        format : str, default 'wide'
            'wide' (datetime index with variables as columns) or 'long'.

        Returns
        -------
        pd.DataFrame
            Clean hourly solar radiation time series.
        """
        locs = _normalize_locations(locations=locations, lat=lat, lon=lon, region=region)
        s_date = _parse_date(start_date, "2020-01-01")
        e_date = _parse_date(end_date, "2020-01-07")

        all_records: List[Dict[str, Any]] = []

        if len(locs) == 1:
            loc = locs[0]
            all_records = self.fetch_point(
                lat=loc["lat"],
                lon=loc["lon"],
                start_date=s_date,
                end_date=e_date,
                variables=variables,
                location_name=loc["name"],
                compute_derived_metrics=compute_derived_metrics,
            )
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self.fetch_point,
                        lat=loc["lat"],
                        lon=loc["lon"],
                        start_date=s_date,
                        end_date=e_date,
                        variables=variables,
                        location_name=loc["name"],
                        compute_derived_metrics=compute_derived_metrics,
                    )
                    for loc in locs
                ]
                for f in as_completed(futures):
                    try:
                        all_records.extend(f.result())
                    except Exception as e:
                        logger.warning(f"SARAH-3 point query failed: {e}")

        return self._reshape_output(all_records, format=format)


# ==============================================================================
# Functional Public API
# ==============================================================================

def query_sarah3(
    locations: Optional[Union[Dict[str, float], Tuple[float, float], List[Any], str]] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    region: Optional[str] = None,
    variables: Optional[List[str]] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    compute_derived_metrics: bool = True,
    format: str = "wide",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Query EUMETSAT CM SAF SARAH-3 satellite solar irradiance in a single function call.

    Examples
    --------
    >>> # 1. Query Dedan Kimathi University (Nyeri) for GHI, DNI, and Diffuse radiation
    >>> df = query_sarah3(lat=-0.398, lon=36.958, start_date='2023-01-01', end_date='2023-01-07')

    >>> # 2. Multi-decadal historical query for Kenya
    >>> df_clim = query_sarah3(lat=-0.398, lon=36.958, start_date='1990-01-01', end_date='1990-01-31')
    """
    client = SARAH3(cache_dir=cache_dir)
    return client.query(
        locations=locations,
        lat=lat,
        lon=lon,
        region=region,
        variables=variables,
        start_date=start_date,
        end_date=end_date,
        compute_derived_metrics=compute_derived_metrics,
        format=format,
    )



# Aliases
SARAH3Wrapper = SARAH3
get_sarah3_data = query_sarah3
