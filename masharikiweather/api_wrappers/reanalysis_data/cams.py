"""
Copernicus Atmosphere Monitoring Service (CAMS) API Wrapper
============================================================
Queries atmospheric composition, aerosol optical depth (AOD), desert dust,
particulate matter (PM2.5, PM10), trace gases (NO2, SO2, CO, O3), and solar UV
radiation from the European Copernicus Earth observation program.

Data Sources & Tiers:
1. Tier 1 (Zero-Key Operational Global Service):
   Open-Meteo CAMS Global Air Quality API
   (AOD 550nm, Saharan/Somali Dust, PM10, PM2.5, CO, NO2, SO2, O3, NH3, UV Index).
   Free public REST API, no registration, no API key required.
   Provides continuous hourly operational data (2023–present) and rolling 5-day forecasts.
2. Tier 2 (Solar & McClear Radiation):
   SoDa CAMS Radiation Service (via pvlib iotools) providing all-sky and McClear clear-sky
   GHI, DNI, DHI, and TOA (optional, requires free SoDa email).
3. Tier 3 (Historical Reanalysis Bulk):
   ECMWF Copernicus Atmosphere Data Store (ADS EAC4 reanalysis via cdsapi).
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

import pandas as pd

from .base import (
    DEFAULT_CACHE_DIR,
    _BaseDatasetWrapper,
    _normalize_locations,
    _parse_date,
    _safe_request,
)

logger = logging.getLogger("MasharikiWeather.CAMS")

# ==============================================================================
# Parameter Definitions & Metadata Dictionary
# ==============================================================================

CAMS_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "aerosol_optical_depth": {
        "api_name": "aerosol_optical_depth",
        "aliases": ["aod", "aod_550", "aod550", "total_aod"],
        "unit": "dimensionless",
        "category": "Aerosols & Dust",
        "description": "Total atmospheric Aerosol Optical Depth at 550 nm (extinction across entire vertical atmospheric column).",
    },
    "dust": {
        "api_name": "dust",
        "aliases": ["saharan_dust", "desert_dust", "mineral_dust"],
        "unit": "μg/m³",
        "category": "Aerosols & Dust",
        "description": "Surface mass concentration of mineral/desert dust particles (e.g., Saharan and Chalbi/Somali dust plumes).",
    },
    "pm2_5": {
        "api_name": "pm2_5",
        "aliases": ["pm25", "pm_2_5", "particulate_matter_2_5"],
        "unit": "μg/m³",
        "category": "Particulate Matter",
        "description": "Fine particulate matter with aerodynamic diameter <= 2.5 μm (primary public health risk metric).",
    },
    "pm10": {
        "api_name": "pm10",
        "aliases": ["pm_10", "particulate_matter_10", "coarse_pm"],
        "unit": "μg/m³",
        "category": "Particulate Matter",
        "description": "Inhalable particulate matter with aerodynamic diameter <= 10 μm.",
    },
    "nitrogen_dioxide": {
        "api_name": "nitrogen_dioxide",
        "aliases": ["no2", "no_2"],
        "unit": "μg/m³",
        "category": "Trace Gases",
        "description": "Nitrogen dioxide surface concentration (key marker of fossil fuel combustion and urban traffic).",
    },
    "sulphur_dioxide": {
        "api_name": "sulphur_dioxide",
        "aliases": ["so2", "so_2", "sulfur_dioxide"],
        "unit": "μg/m³",
        "category": "Trace Gases",
        "description": "Sulphur dioxide surface concentration (volcanic emissions and industrial combustion).",
    },
    "carbon_monoxide": {
        "api_name": "carbon_monoxide",
        "aliases": ["co"],
        "unit": "μg/m³",
        "category": "Trace Gases",
        "description": "Carbon monoxide surface concentration (biomass burning, wildfires, and incomplete combustion).",
    },
    "ozone": {
        "api_name": "ozone",
        "aliases": ["o3", "surface_ozone", "tropospheric_ozone"],
        "unit": "μg/m³",
        "category": "Trace Gases",
        "description": "Ground-level surface ozone concentration (secondary photochemical pollutant formed from NOx and VOCs under sunlight).",
    },
    "ammonia": {
        "api_name": "ammonia",
        "aliases": ["nh3"],
        "unit": "μg/m³",
        "category": "Trace Gases",
        "description": "Ammonia surface concentration (primarily from agricultural livestock and fertilizer applications).",
    },
    "uv_index": {
        "api_name": "uv_index",
        "aliases": ["uv", "uvi", "all_sky_uv"],
        "unit": "index (0–12+)",
        "category": "Radiation & UV",
        "description": "All-sky surface Ultraviolet Index, taking into account clouds, ozone, and aerosol attenuation.",
    },
    "uv_index_clear_sky": {
        "api_name": "uv_index_clear_sky",
        "aliases": ["uv_clear", "clear_sky_uv", "uv_cs"],
        "unit": "index (0–12+)",
        "category": "Radiation & UV",
        "description": "Theoretical maximum Clear-Sky UV Index in the absence of clouds.",
    },
    "us_aqi": {
        "api_name": "us_aqi",
        "aliases": ["aqi", "us_air_quality_index"],
        "unit": "AQI index (0–500)",
        "category": "Air Quality Indices",
        "description": "United States EPA Air Quality Index based on multi-pollutant maximum risk calculation.",
    },
    "european_aqi": {
        "api_name": "european_aqi",
        "aliases": ["eaqi", "eu_aqi", "european_air_quality_index"],
        "unit": "EAQI index (0–100)",
        "category": "Air Quality Indices",
        "description": "European Air Quality Index rating overall air cleanliness according to EU standards.",
    },
    "alder_pollen": {
        "api_name": "alder_pollen",
        "aliases": ["pollen_alder"],
        "unit": "grains/m³",
        "category": "Pollen & Allergens",
        "description": "Alder tree pollen count.",
    },
    "birch_pollen": {
        "api_name": "birch_pollen",
        "aliases": ["pollen_birch"],
        "unit": "grains/m³",
        "category": "Pollen & Allergens",
        "description": "Birch tree pollen count.",
    },
    "grass_pollen": {
        "api_name": "grass_pollen",
        "aliases": ["pollen_grass"],
        "unit": "grains/m³",
        "category": "Pollen & Allergens",
        "description": "Grass pollen count.",
    },
    "olive_pollen": {
        "api_name": "olive_pollen",
        "aliases": ["pollen_olive"],
        "unit": "grains/m³",
        "category": "Pollen & Allergens",
        "description": "Olive tree pollen count.",
    },
    "ragweed_pollen": {
        "api_name": "ragweed_pollen",
        "aliases": ["pollen_ragweed"],
        "unit": "grains/m³",
        "category": "Pollen & Allergens",
        "description": "Ragweed pollen count.",
    },
}

# Build flat lookup table: maps any alias or api_name to canonical api_name
CAMS_AQ_LEXICON: Dict[str, str] = {}
for param_key, meta in CAMS_PARAMETERS.items():
    api_name = meta["api_name"]
    CAMS_AQ_LEXICON[param_key] = api_name
    CAMS_AQ_LEXICON[api_name] = api_name
    for alias in meta.get("aliases", []):
        CAMS_AQ_LEXICON[alias.lower().strip()] = api_name

DEFAULT_CAMS_AQ_VARS = [
    "aerosol_optical_depth",
    "dust",
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "uv_index",
]

# US EPA AQI Health Categories & Breakpoints
AQI_CATEGORIES = [
    {"min": 0, "max": 50, "label": "Good", "color": "#00e400", "advisory": "Air quality is satisfactory; air pollution poses little or no risk."},
    {"min": 51, "max": 100, "label": "Moderate", "color": "#ffff00", "advisory": "Air quality is acceptable; sensitive individuals may experience minor symptoms."},
    {"min": 101, "max": 150, "label": "Unhealthy for Sensitive Groups", "color": "#ff7e00", "advisory": "Members of sensitive groups may experience health effects; general public unlikely affected."},
    {"min": 151, "max": 200, "label": "Unhealthy", "color": "#ff0000", "advisory": "Everyone may begin to experience health effects; sensitive groups may experience more serious effects."},
    {"min": 201, "max": 300, "label": "Very Unhealthy", "color": "#8f3f97", "advisory": "Health alert: risk of health effects is increased for everyone."},
    {"min": 301, "max": 500, "label": "Hazardous", "color": "#7e0023", "advisory": "Health warning of emergency conditions: everyone is more likely to be affected."},
]


def classify_aqi(aqi_value: float) -> Dict[str, Any]:
    """
    Classify a US EPA Air Quality Index value into its official health category,
    display color, and health advisory recommendation.
    """
    if pd.isna(aqi_value):
        return {"label": "Unknown", "color": "#95a5a6", "advisory": "No data available."}
    v = float(aqi_value)
    for cat in AQI_CATEGORIES:
        if cat["min"] <= v <= cat["max"]:
            return {
                "value": v,
                "label": cat["label"],
                "color": cat["color"],
                "advisory": cat["advisory"],
            }
    if v > 500:
        return {
            "value": v,
            "label": "Beyond Index (Hazardous)",
            "color": "#7e0023",
            "advisory": "Emergency air quality hazard.",
        }
    return {"value": v, "label": "Good", "color": "#00e400", "advisory": "Air quality is satisfactory."}


def search_cams_parameters(query: Optional[str] = None) -> pd.DataFrame:
    """
    Search available CAMS and atmospheric air quality parameters by keyword or category.

    Parameters
    ----------
    query : str, optional
        Keyword to search in parameter name, aliases, or description.
        If None, returns all available parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Variable', 'API Name', 'Category', 'Unit', 'Aliases', 'Description'].
    """
    rows = []
    q = query.lower().strip() if query else None

    for name, meta in CAMS_PARAMETERS.items():
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


def list_cams_parameters(category: Optional[str] = None) -> pd.DataFrame:
    """
    List all CAMS variables, optionally filtered by category.
    Available categories: 'Aerosols & Dust', 'Particulate Matter', 'Trace Gases',
    'Radiation & UV', 'Air Quality Indices', 'Pollen & Allergens'.
    """
    df = search_cams_parameters(query=None)
    if category:
        cat_lower = category.lower().strip()
        df = df[df["Category"].str.lower().str.contains(cat_lower)]
    return df.reset_index(drop=True)


def get_cams_metadata() -> Dict[str, Any]:
    """
    Retrieve metadata regarding the CAMS dataset architecture, resolution, and citation.
    """
    return {
        "provider": "European Union Copernicus Earth Observation Programme / ECMWF",
        "program": "Copernicus Atmosphere Monitoring Service (CAMS)",
        "api_service": "Open-Meteo Air Quality REST API (Zero-Key)",
        "api_endpoint": "https://air-quality-api.open-meteo.com/v1/air-quality",
        "spatial_resolution": "0.4° (~40 km grid cell) Global CAMS model",
        "temporal_resolution": "Hourly (continuous 24-hr resolution)",
        "temporal_coverage": "2023 through present (operational reanalysis) + 5-day rolling forecast",
        "primary_applications": [
            "Solar radiation atmospheric attenuation (Aerosol Optical Depth)",
            "Desert dust transport and particulate matter modeling (PM2.5, PM10)",
            "Urban photochemical air quality monitoring (NO2, O3, CO, SO2)",
            "Surface solar Ultraviolet Index forecasting",
            "Public health epidemiology and WHO air quality benchmark compliance",
        ],
        "license": "Copernicus open data policy (free and open access for commercial and academic purposes)",
        "citation": "Inness, A., et al. (2019). The CAMS reanalysis of atmospheric composition. Atmos. Chem. Phys., 19(6), 3515-3556.",
    }


# ==============================================================================
# CAMS Wrapper Client
# ==============================================================================

class CAMS(_BaseDatasetWrapper):
    """
    Client for querying Copernicus Atmosphere Monitoring Service (CAMS)
    aerosols, particulate matter, trace gases, and solar UV radiation.
    """

    OPENMETEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def __init__(
        self,
        soda_email: Optional[str] = None,
        cache_dir: str = DEFAULT_CACHE_DIR,
        rate_limit_delay: float = 0.2,
        max_workers: int = 4,
    ):
        super().__init__(
            cache_dir=cache_dir,
            rate_limit_delay=rate_limit_delay,
            max_workers=max_workers,
        )
        self.soda_email = soda_email or os.environ.get("SODA_EMAIL") or os.environ.get("CAMS_EMAIL")

    def fetch_point(
        self,
        lat: float,
        lon: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        past_days: Optional[int] = None,
        forecast_days: Optional[int] = None,
        variables: Optional[List[str]] = None,
        location_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch CAMS global atmospheric data for a single latitude/longitude coordinate.

        Parameters
        ----------
        lat : float
            Latitude in degrees (-90 to 90).
        lon : float
            Longitude in degrees (-180 to 180).
        start_date : str, optional
            ISO start date string ('YYYY-MM-DD').
        end_date : str, optional
            ISO end date string ('YYYY-MM-DD').
        past_days : int, optional
            Number of past days (0 to 92) relative to current time.
        forecast_days : int, optional
            Number of forecast days (1 to 7) relative to current time.
        variables : list of str, optional
            List of variable names or aliases. Defaults to standard aerosol & AQ variables.
        location_name : str, optional
            Human-readable location label.

        Returns
        -------
        list of dict
            List of hourly data records.
        """
        loc_label = location_name or f"{lat:.3f}_{lon:.3f}"
        vars_requested = variables or DEFAULT_CAMS_AQ_VARS

        api_vars: List[str] = []
        var_to_canon: Dict[str, str] = {}
        for v in vars_requested:
            v_clean = v.lower().strip()
            mapped = CAMS_AQ_LEXICON.get(v_clean, v_clean)
            api_vars.append(mapped)
            var_to_canon[mapped] = v_clean

        api_vars = sorted(list(set(api_vars)))

        # Cache key construction
        cache_key = {
            "source": "cams_aq",
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "start": start_date or "",
            "end": end_date or "",
            "past": past_days or 0,
            "forecast": forecast_days or 0,
            "vars": api_vars,
        }
        cache_file = self._get_cache_path("cams_aq", cache_key)
        if self._is_cache_valid(cache_file):
            try:
                cached_df = pd.read_parquet(cache_file)
                return cached_df.to_dict(orient="records")
            except Exception:
                pass

        # Build query parameters
        params: Dict[str, Any] = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "hourly": ",".join(api_vars),
            "timezone": "UTC",
        }

        if past_days is not None or forecast_days is not None:
            if past_days is not None:
                params["past_days"] = int(past_days)
            if forecast_days is not None:
                params["forecast_days"] = int(forecast_days)
        elif start_date and end_date:
            params["start_date"] = start_date
            params["end_date"] = end_date
        else:
            # Default to 3 past days + 3 forecast days
            params["past_days"] = 3
            params["forecast_days"] = 3

        url = f"{self.OPENMETEO_AQ_URL}?{urllib.parse.urlencode(params)}"

        time.sleep(self.rate_limit_delay)
        raw_bytes = _safe_request(url)
        data = json.loads(raw_bytes.decode("utf-8"))

        records: List[Dict[str, Any]] = []
        hourly_data = data.get("hourly", {})
        times = pd.to_datetime(hourly_data.get("time", []), utc=True)

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

        if records:
            try:
                pd.DataFrame(records).to_parquet(cache_file, index=False)
            except Exception:
                pass

        return records

    def fetch_solar_point(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        altitude: float = 0.0,
        time_step: str = "15min",
        location_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch CAMS Solar & McClear radiation time series (Tier 2, requires SoDa email).
        """
        loc_label = location_name or f"{lat:.3f}_{lon:.3f}"
        if not self.soda_email:
            raise ValueError(
                "CAMS Solar Radiation requires a free SoDa user email. "
                "Pass soda_email='your_email@domain.com' or set CAMS_EMAIL environment variable."
            )

        cache_key = {
            "source": "cams_solar",
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "start": start_date,
            "end": end_date,
            "step": time_step,
        }
        cache_file = self._get_cache_path("cams_solar", cache_key)
        if self._is_cache_valid(cache_file):
            try:
                cached_df = pd.read_parquet(cache_file)
                return cached_df.to_dict(orient="records")
            except Exception:
                pass

        try:
            import pvlib
        except ImportError:
            raise ImportError("pvlib is required for CAMS Solar & McClear. Install with: pip install pvlib")

        df_cams, _ = pvlib.iotools.get_cams(
            latitude=lat,
            longitude=lon,
            altitude=max(altitude, 0.0),
            start=pd.Timestamp(start_date),
            end=pd.Timestamp(end_date),
            email=self.soda_email,
            identifier="mcclear",
            time_step=time_step,
            time_ref="UT",
            verbose=False,
            map_variables=True,
            timeout=90,
        )

        if df_cams.index.tz is None:
            df_cams.index = df_cams.index.tz_localize("UTC")
        else:
            df_cams.index = df_cams.index.tz_convert("UTC")

        records: List[Dict[str, Any]] = []
        for col in df_cams.columns:
            for ts, val in zip(df_cams.index, df_cams[col]):
                if pd.notna(val):
                    records.append({
                        "time": ts,
                        "location": loc_label,
                        "lat": lat,
                        "lon": lon,
                        "variable": col.lower(),
                        "value": float(val),
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
        past_days: Optional[int] = None,
        forecast_days: Optional[int] = None,
        format: str = "wide",
    ) -> pd.DataFrame:
        """
        Query CAMS atmospheric data for one or multiple locations.

        Parameters
        ----------
        locations : dict, tuple, list, or str, optional
            Target site or list of target sites.
        lat, lon : float, optional
            Direct coordinates.
        region : str, optional
            Named region (e.g. 'kenya', 'nairobi').
        variables : list of str, optional
            Variables to retrieve (e.g., ['aod', 'pm2_5', 'dust', 'uv_index']).
        start_date, end_date : str or datetime, optional
            Date range ('YYYY-MM-DD').
        past_days : int, optional
            Number of past days (0 to 92).
        forecast_days : int, optional
            Number of forecast days (1 to 7).
        format : str, default 'wide'
            'wide' (datetime index with variables as columns) or 'long' (time, location, variable, value).

        Returns
        -------
        pd.DataFrame
            Clean atmospheric time series.
        """
        locs = _normalize_locations(locations=locations, lat=lat, lon=lon, region=region)

        s_date = None
        e_date = None
        if start_date is not None:
            s_date = _parse_date(start_date, "2024-01-01")
            e_date = _parse_date(end_date, s_date) if end_date else s_date

        all_records: List[Dict[str, Any]] = []

        if len(locs) == 1:
            loc = locs[0]
            all_records = self.fetch_point(
                lat=loc["lat"],
                lon=loc["lon"],
                start_date=s_date,
                end_date=e_date,
                past_days=past_days,
                forecast_days=forecast_days,
                variables=variables,
                location_name=loc["name"],
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
                        past_days=past_days,
                        forecast_days=forecast_days,
                        variables=variables,
                        location_name=loc["name"],
                    )
                    for loc in locs
                ]
                for f in as_completed(futures):
                    try:
                        all_records.extend(f.result())
                    except Exception as e:
                        logger.warning(f"CAMS fetch point failed: {e}")

        return self._reshape_output(all_records, format=format)

    def query_aerosols(
        self,
        locations: Optional[Union[Dict[str, float], Tuple[float, float], List[Any], str]] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        region: Optional[str] = None,
        variables: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        past_days: Optional[int] = None,
        forecast_days: Optional[int] = None,
        format: str = "wide",
    ) -> pd.DataFrame:
        """Query CAMS aerosol optical depth, desert dust, and particulate matter."""
        vars_to_query = variables or ["aod", "dust", "pm2_5", "pm10"]
        return self.query(
            locations=locations,
            lat=lat,
            lon=lon,
            region=region,
            variables=vars_to_query,
            start_date=start_date,
            end_date=end_date,
            past_days=past_days,
            forecast_days=forecast_days,
            format=format,
        )

    def query_air_quality(
        self,
        locations: Optional[Union[Dict[str, float], Tuple[float, float], List[Any], str]] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        region: Optional[str] = None,
        variables: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        past_days: Optional[int] = None,
        forecast_days: Optional[int] = None,
        format: str = "wide",
    ) -> pd.DataFrame:
        """Query CAMS air quality parameters (US AQI, PM2.5, PM10, NO2, O3, CO, SO2)."""
        vars_to_query = variables or ["us_aqi", "pm2_5", "pm10", "no2", "ozone", "co", "so2"]
        return self.query(
            locations=locations,
            lat=lat,
            lon=lon,
            region=region,
            variables=vars_to_query,
            start_date=start_date,
            end_date=end_date,
            past_days=past_days,
            forecast_days=forecast_days,
            format=format,
        )

    def query_forecast(
        self,
        lat: float,
        lon: float,
        forecast_days: int = 5,
        variables: Optional[List[str]] = None,
        format: str = "wide",
    ) -> pd.DataFrame:
        """
        Query rolling air quality forecast (1 to 7 days ahead) for a given location.
        """
        vars_to_query = variables or ["us_aqi", "pm2_5", "pm10", "aod", "dust", "uv_index", "no2", "ozone"]
        return self.query(
            lat=lat,
            lon=lon,
            forecast_days=min(max(forecast_days, 1), 7),
            past_days=0,
            variables=vars_to_query,
            format=format,
        )


# ==============================================================================
# Functional Public API
# ==============================================================================

def query_cams(
    locations: Optional[Union[Dict[str, float], Tuple[float, float], List[Any], str]] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    region: Optional[str] = None,
    variables: Optional[List[str]] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    past_days: Optional[int] = None,
    forecast_days: Optional[int] = None,
    format: str = "wide",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Query Copernicus Atmosphere Monitoring Service (CAMS) data in a single line.

    Examples
    --------
    >>> # 1. Query DeKUT, Nyeri for AOD, PM2.5, and Dust
    >>> df = query_cams(lat=-0.398, lon=36.958, start_date='2024-01-01', end_date='2024-01-05')

    >>> # 2. Query 5-day Air Quality forecast for Nairobi
    >>> df = query_cams(lat=-1.286, lon=36.817, forecast_days=5)
    """
    client = CAMS(cache_dir=cache_dir)
    return client.query(
        locations=locations,
        lat=lat,
        lon=lon,
        region=region,
        variables=variables,
        start_date=start_date,
        end_date=end_date,
        past_days=past_days,
        forecast_days=forecast_days,
        format=format,
    )


def query_cams_aerosols(
    locations: Optional[Union[Dict[str, float], Tuple[float, float], List[Any], str]] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    region: Optional[str] = None,
    variables: Optional[List[str]] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    past_days: Optional[int] = None,
    forecast_days: Optional[int] = None,
    format: str = "wide",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Convenience function to query CAMS aerosol optical depth (AOD 550nm),
    desert dust, PM2.5, and PM10 in one line.
    """
    client = CAMS(cache_dir=cache_dir)
    return client.query_aerosols(
        locations=locations,
        lat=lat,
        lon=lon,
        region=region,
        variables=variables,
        start_date=start_date,
        end_date=end_date,
        past_days=past_days,
        forecast_days=forecast_days,
        format=format,
    )


def query_cams_air_quality(
    locations: Optional[Union[Dict[str, float], Tuple[float, float], List[Any], str]] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    region: Optional[str] = None,
    variables: Optional[List[str]] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    past_days: Optional[int] = None,
    forecast_days: Optional[int] = None,
    format: str = "wide",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Convenience function to query CAMS multi-pollutant Air Quality (AQI, PM2.5, PM10, NO2, O3, CO, SO2).
    """
    client = CAMS(cache_dir=cache_dir)
    return client.query_air_quality(
        locations=locations,
        lat=lat,
        lon=lon,
        region=region,
        variables=variables,
        start_date=start_date,
        end_date=end_date,
        past_days=past_days,
        forecast_days=forecast_days,
        format=format,
    )


def query_cams_forecast(
    lat: float,
    lon: float,
    forecast_days: int = 5,
    variables: Optional[List[str]] = None,
    format: str = "wide",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Convenience function to retrieve a rolling air quality forecast (1 to 7 days).
    """
    client = CAMS(cache_dir=cache_dir)
    return client.query_forecast(
        lat=lat,
        lon=lon,
        forecast_days=forecast_days,
        variables=variables,
        format=format,
    )
