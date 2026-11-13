"""
Open-Meteo High-Resolution Global Weather and Historical Reanalysis Wrapper for MasharikiWeather.

Provides unified interfaces for querying Open-Meteo global meteorological data,
including ECMWF IFS, NOAA GFS, DWD ICON, and ECMWF ERA5/ERA5-Land reanalysis.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .common import (
    DEFAULT_CACHE_DIR,
    REGIONAL_CENTROIDS,
    _BaseDatasetWrapper,
    _normalize_locations,
    _parse_date,
    _safe_request,
)

logger = logging.getLogger("MasharikiWeather.OpenMeteo")

# ==============================================================================
# WMO Weather Interpretation Codes Dictionary
# ==============================================================================

WMO_WEATHER_CODES: Dict[int, Dict[str, str]] = {
    0: {"description": "Clear sky", "icon": "clear_sky"},
    1: {"description": "Mainly clear", "icon": "mainly_clear"},
    2: {"description": "Partly cloudy", "icon": "partly_cloudy"},
    3: {"description": "Overcast", "icon": "overcast"},
    45: {"description": "Fog", "icon": "fog"},
    48: {"description": "Depositing rime fog", "icon": "rime_fog"},
    51: {"description": "Light drizzle", "icon": "light_drizzle"},
    53: {"description": "Moderate drizzle", "icon": "moderate_drizzle"},
    55: {"description": "Dense drizzle", "icon": "dense_drizzle"},
    56: {"description": "Light freezing drizzle", "icon": "light_freezing_drizzle"},
    57: {"description": "Dense freezing drizzle", "icon": "dense_freezing_drizzle"},
    61: {"description": "Slight rain", "icon": "slight_rain"},
    63: {"description": "Moderate rain", "icon": "moderate_rain"},
    65: {"description": "Heavy rain", "icon": "heavy_rain"},
    66: {"description": "Light freezing rain", "icon": "light_freezing_rain"},
    67: {"description": "Heavy freezing rain", "icon": "heavy_freezing_rain"},
    71: {"description": "Slight snow fall", "icon": "slight_snow"},
    73: {"description": "Moderate snow fall", "icon": "moderate_snow"},
    75: {"description": "Heavy snow fall", "icon": "heavy_snow"},
    77: {"description": "Snow grains", "icon": "snow_grains"},
    80: {"description": "Slight rain showers", "icon": "slight_rain_showers"},
    81: {"description": "Moderate rain showers", "icon": "moderate_rain_showers"},
    82: {"description": "Violent rain showers", "icon": "violent_rain_showers"},
    85: {"description": "Slight snow showers", "icon": "slight_snow_showers"},
    86: {"description": "Heavy snow showers", "icon": "heavy_snow_showers"},
    95: {"description": "Thunderstorm", "icon": "thunderstorm"},
    96: {"description": "Thunderstorm with slight hail", "icon": "thunderstorm_slight_hail"},
    99: {"description": "Thunderstorm with heavy hail", "icon": "thunderstorm_heavy_hail"},
}


def decode_weather_code(code: Union[int, float]) -> Dict[str, str]:
    """
    Translate a numeric WMO weather interpretation code into a human-readable
    description and weather condition key.

    Parameters
    ----------
    code : int or float
        Numeric WMO code value.

    Returns
    -------
    dict
        Dictionary with 'description' and 'icon' string fields.
    """
    if pd.isna(code):
        return {"description": "Unknown", "icon": "unknown"}
    c = int(code)
    return WMO_WEATHER_CODES.get(c, {"description": f"Weather code {c}", "icon": "unknown"})


# Supported Numerical Weather Prediction (NWP) and Reanalysis Models
OPEN_METEO_MODELS: Dict[str, str] = {
    "auto": "Seamless multi-model blend optimized by location",
    "ecmwf_ifs025": "ECMWF Integrated Forecasting System (9 km global operational model)",
    "gfs_seamless": "NOAA Global Forecast System (13 km / 25 km)",
    "icon_seamless": "DWD ICON (11 km global nonhydrostatic model)",
    "era5_seamless": "ECMWF ERA5 / ERA5-Land reanalysis (1940 to present)",
}

# ==============================================================================
# Parameter Definitions & Metadata Dictionary
# ==============================================================================

OPENMETEO_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "temperature_2m": {
        "api_name": "temperature_2m",
        "aliases": ["temp", "t2m", "temperature", "air_temp"],
        "unit": "°C",
        "category": "Temperature & Humidity",
        "description": "Air temperature at 2 meters above ground level.",
    },
    "relative_humidity_2m": {
        "api_name": "relative_humidity_2m",
        "aliases": ["rh", "rh2m", "humidity", "relative_humidity"],
        "unit": "%",
        "category": "Temperature & Humidity",
        "description": "Relative humidity at 2 meters above ground level.",
    },
    "dew_point_2m": {
        "api_name": "dew_point_2m",
        "aliases": ["dew_point", "td2m", "dewpoint"],
        "unit": "°C",
        "category": "Temperature & Humidity",
        "description": "Dew point temperature at 2 meters above ground level.",
    },
    "apparent_temperature": {
        "api_name": "apparent_temperature",
        "aliases": ["feels_like", "heat_index", "wind_chill"],
        "unit": "°C",
        "category": "Temperature & Humidity",
        "description": "Apparent temperature (feels-like temperature combining wind chill and humidity index).",
    },
    "precipitation": {
        "api_name": "precipitation",
        "aliases": ["precip", "rain_total", "rainfall"],
        "unit": "mm",
        "category": "Precipitation & Hydrology",
        "description": "Total liquid precipitation (rain, showers) accumulated during the preceding hour.",
    },
    "rain": {
        "api_name": "rain",
        "aliases": ["large_scale_rain"],
        "unit": "mm",
        "category": "Precipitation & Hydrology",
        "description": "Large-scale stratiform rain accumulation.",
    },
    "showers": {
        "api_name": "showers",
        "aliases": ["convective_rain", "convective_showers"],
        "unit": "mm",
        "category": "Precipitation & Hydrology",
        "description": "Convective rain showers from cumulus/cumulonimbus clouds.",
    },
    "precipitation_probability": {
        "api_name": "precipitation_probability",
        "aliases": ["pop", "precip_prob", "rain_chance"],
        "unit": "%",
        "category": "Precipitation & Hydrology",
        "description": "Probability of precipitation exceeding 0.1 mm during the hour.",
    },
    "et0_fao_evapotranspiration": {
        "api_name": "et0_fao_evapotranspiration",
        "aliases": ["et0", "evapotranspiration", "reference_et", "fao_et0"],
        "unit": "mm",
        "category": "Precipitation & Hydrology",
        "description": "Reference potential evapotranspiration calculated using the FAO-56 Penman-Monteith equation.",
    },
    "surface_pressure": {
        "api_name": "surface_pressure",
        "aliases": ["sp", "station_pressure"],
        "unit": "hPa",
        "category": "Atmospheric Pressure",
        "description": "Atmospheric pressure measured at surface elevation.",
    },
    "pressure_msl": {
        "api_name": "pressure_msl",
        "aliases": ["mslp", "msl", "barometer", "sea_level_pressure"],
        "unit": "hPa",
        "category": "Atmospheric Pressure",
        "description": "Atmospheric pressure reduced to Mean Sea Level (MSL).",
    },
    "wind_speed_10m": {
        "api_name": "wind_speed_10m",
        "aliases": ["wind_speed", "ws10m", "ws", "wind"],
        "unit": "m/s",
        "category": "Wind",
        "description": "Horizontal wind speed at 10 meters above ground level.",
    },
    "wind_direction_10m": {
        "api_name": "wind_direction_10m",
        "aliases": ["wind_direction", "wd10m", "wd"],
        "unit": "degrees (°)",
        "category": "Wind",
        "description": "Wind direction at 10 meters (0° = North, 90° = East, 180° = South, 270° = West).",
    },
    "wind_gusts_10m": {
        "api_name": "wind_gusts_10m",
        "aliases": ["gusts", "wind_gusts", "max_wind"],
        "unit": "m/s",
        "category": "Wind",
        "description": "Maximum 3-second wind gust observed during the preceding hour at 10m.",
    },
    "wind_speed_100m": {
        "api_name": "wind_speed_100m",
        "aliases": ["wind_100m", "ws100m", "turbine_wind"],
        "unit": "m/s",
        "category": "Wind",
        "description": "Wind speed at 100 meters above ground level (representative hub height for wind turbines).",
    },
    "cloud_cover": {
        "api_name": "cloud_cover",
        "aliases": ["clouds", "cloudcover", "tcc", "total_clouds"],
        "unit": "%",
        "category": "Cloud Cover",
        "description": "Total cloud cover across all atmospheric levels (0 = completely clear, 100 = completely overcast).",
    },
    "cloud_cover_low": {
        "api_name": "cloud_cover_low",
        "aliases": ["lcc", "low_clouds"],
        "unit": "%",
        "category": "Cloud Cover",
        "description": "Low cloud cover (altitude < 2 km above ground level).",
    },
    "cloud_cover_mid": {
        "api_name": "cloud_cover_mid",
        "aliases": ["mcc", "mid_clouds"],
        "unit": "%",
        "category": "Cloud Cover",
        "description": "Mid-level cloud cover (altitude 2 km to 6 km above ground level).",
    },
    "cloud_cover_high": {
        "api_name": "cloud_cover_high",
        "aliases": ["hcc", "high_clouds", "cirrus"],
        "unit": "%",
        "category": "Cloud Cover",
        "description": "High-altitude cloud cover (altitude > 6 km above ground level).",
    },
    "shortwave_radiation": {
        "api_name": "shortwave_radiation",
        "aliases": ["ghi", "solar", "solar_radiation", "solar_rad"],
        "unit": "W/m²",
        "category": "Solar Radiation",
        "description": "Global Horizontal Irradiance (GHI). Total incoming solar shortwave flux reaching the surface.",
    },
    "direct_normal_irradiance": {
        "api_name": "direct_normal_irradiance",
        "aliases": ["dni", "beam_normal"],
        "unit": "W/m²",
        "category": "Solar Radiation",
        "description": "Direct Normal Irradiance (DNI). Beam radiation on a surface tracking the sun.",
    },
    "soil_temperature_0_to_7cm": {
        "api_name": "soil_temperature_0_to_7cm",
        "aliases": ["soil_temp_0_7", "topsoil_temp"],
        "unit": "°C",
        "category": "Soil & Agricultural",
        "description": "Soil temperature in the 0 to 7 cm topsoil layer.",
    },
    "soil_temperature_7_to_28cm": {
        "api_name": "soil_temperature_7_to_28cm",
        "aliases": ["soil_temp_7_28", "subsoil_temp"],
        "unit": "°C",
        "category": "Soil & Agricultural",
        "description": "Soil temperature in the 7 to 28 cm rootzone layer.",
    },
    "soil_moisture_0_to_7cm": {
        "api_name": "soil_moisture_0_to_7cm",
        "aliases": ["soil_moisture", "soil_moist_0_7", "topsoil_moisture"],
        "unit": "m³/m³",
        "category": "Soil & Agricultural",
        "description": "Volumetric soil water content in the 0 to 7 cm layer (fraction from 0.0 to 1.0).",
    },
    "soil_moisture_7_to_28cm": {
        "api_name": "soil_moisture_7_to_28cm",
        "aliases": ["soil_moist_7_28", "rootzone_moisture"],
        "unit": "m³/m³",
        "category": "Soil & Agricultural",
        "description": "Volumetric soil water content in the 7 to 28 cm rootzone layer.",
    },
    "weather_code": {
        "api_name": "weather_code",
        "aliases": ["weather", "wmo_code", "condition_code"],
        "unit": "WMO Code",
        "category": "Weather Interpretation",
        "description": "World Meteorological Organization numeric weather interpretation code (0=clear, 61-65=rain, 95=thunderstorm).",
    },
}

# Lookup dictionary mapping user aliases to canonical API parameter names
OPENMETEO_LEXICON: Dict[str, str] = {}
for param_key, meta in OPENMETEO_PARAMETERS.items():
    api_name = meta["api_name"]
    OPENMETEO_LEXICON[param_key] = api_name
    OPENMETEO_LEXICON[api_name] = api_name
    for alias in meta.get("aliases", []):
        OPENMETEO_LEXICON[alias.lower().strip()] = api_name

DEFAULT_OPENMETEO_VARS: List[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
    "et0_fao_evapotranspiration",
    "soil_moisture_0_to_7cm",
    "weather_code",
]


def search_open_meteo_parameters(query: Optional[str] = None) -> pd.DataFrame:
    """
    Search available Open-Meteo weather, soil, and radiation parameters by keyword or category.

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
    rows: List[Dict[str, Any]] = []
    q = query.lower().strip() if query else None

    for name, meta in OPENMETEO_PARAMETERS.items():
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


def list_open_meteo_parameters(category: Optional[str] = None) -> pd.DataFrame:
    """
    List all Open-Meteo variables, optionally filtered by category.
    Available categories: 'Temperature & Humidity', 'Precipitation & Hydrology',
    'Atmospheric Pressure', 'Wind', 'Cloud Cover', 'Solar Radiation',
    'Soil & Agricultural', 'Weather Interpretation'.

    Parameters
    ----------
    category : str, optional
        Category filter string.

    Returns
    -------
    pd.DataFrame
        Filtered parameters table.
    """
    df = search_open_meteo_parameters(query=None)
    if category:
        cat_lower = category.lower().strip()
        df = df[df["Category"].str.lower().str.contains(cat_lower)]
    return df.reset_index(drop=True)


def get_open_meteo_metadata() -> Dict[str, Any]:
    """
    Retrieve comprehensive scientific metadata regarding Open-Meteo global weather and reanalysis.

    Returns
    -------
    dict
        Metadata configuration parameters and citation references.
    """
    return {
        "service_name": "Open-Meteo Global Weather & Reanalysis API",
        "provider": "Open-Meteo GmbH (Germany) open-source weather initiative",
        "api_endpoints": {
            "forecast": "https://api.open-meteo.com/v1/forecast",
            "historical_archive": "https://archive-api.open-meteo.com/v1/archive",
            "ensemble": "https://ensemble-api.open-meteo.com/v1/ensemble",
        },
        "supported_models": OPEN_METEO_MODELS,
        "temporal_resolution": "Hourly continuous 24-hr profile",
        "temporal_span": "1940 to present (historical) + up to 16 days forward (operational forecast)",
        "authentication": "Zero-Key (Free and open public REST API, no key required)",
        "licensing": "Creative Commons Attribution 4.0 International (CC BY 4.0) / Open Database License (ODbL)",
        "citation": "Open-Meteo Weather API (2024). High-resolution open weather data platform. https://open-meteo.com",
    }


# ==============================================================================
# OpenMeteo Wrapper Client
# ==============================================================================

class OpenMeteo(_BaseDatasetWrapper):
    """
    Client for querying Open-Meteo high-resolution global weather forecasts and historical reanalysis.
    """

    FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(
        self,
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
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
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        past_days: Optional[int] = None,
        forecast_days: Optional[int] = None,
        variables: Optional[Union[List[str], Sequence[str], str]] = None,
        model: Optional[str] = None,
        location_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch hourly weather time series for a single (lat, lon) coordinate.

        Mathematical docstring:
        Input:
            lat: scalar latitude [-90.0, 90.0]
            lon: scalar longitude [-180.0, 180.0]
            temporal: interval [T_start, T_end]
        Output:
            List of N records, each containing keys:
            ['time', 'location', 'lat', 'lon', 'variable', 'value']

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
            Number of past days (0 to 92).
        forecast_days : int, optional
            Number of forecast days (1 to 16).
        variables : list of str, optional
            List of variable names or aliases.
        model : str, optional
            Target NWP model ('ecmwf_ifs025', 'gfs_seamless', 'icon_seamless', 'era5_seamless', or None).
        location_name : str, optional
            Human-readable label.

        Returns
        -------
        list of dict
            Standard observation records.
        """
        loc_label = location_name or f"{lat:.3f}_{lon:.3f}"
        if variables == "all" or (isinstance(variables, (list, tuple)) and "all" in variables):
            vars_requested = list(OPENMETEO_PARAMETERS.keys())
        elif isinstance(variables, (list, tuple)):
            vars_requested = list(variables)
        elif isinstance(variables, str):
            vars_requested = [variables]
        else:
            vars_requested = list(DEFAULT_OPENMETEO_VARS)

        api_vars: List[str] = []
        var_to_canon: Dict[str, str] = {}
        for v in vars_requested:
            v_clean = v.lower().strip()
            mapped = OPENMETEO_LEXICON.get(v_clean, v_clean)
            api_vars.append(mapped)
            var_to_canon[mapped] = v_clean

        api_vars = sorted(list(set(api_vars)))

        # Cache key construction
        cache_key = {
            "source": "open_meteo",
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "start": start_date or "",
            "end": end_date or "",
            "past": past_days or 0,
            "forecast": forecast_days or 0,
            "model": model or "auto",
            "vars": api_vars,
        }
        cache_file = self._get_cache_path("open_meteo", cache_key)
        if self._is_cache_valid(cache_file):
            try:
                cached_df = pd.read_parquet(cache_file)
                return cached_df.to_dict(orient="records")
            except Exception:
                pass

        # Decide whether to query Forecast or Historical Archive endpoint
        is_archive = False
        if start_date and end_date:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                now_dt = datetime.now(timezone.utc)
                if (now_dt - end_dt).days > 85:
                    is_archive = True
            except Exception:
                pass

        endpoint = self.ARCHIVE_URL if is_archive else self.FORECAST_URL

        params: Dict[str, Any] = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "hourly": ",".join(api_vars),
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }

        if model and model != "auto":
            params["models"] = model

        if is_archive:
            params["start_date"] = start_date
            params["end_date"] = end_date
        elif past_days is not None or forecast_days is not None:
            if past_days is not None:
                params["past_days"] = int(past_days)
            if forecast_days is not None:
                params["forecast_days"] = int(forecast_days)
        elif start_date and end_date:
            params["start_date"] = start_date
            params["end_date"] = end_date
        else:
            params["past_days"] = 3
            params["forecast_days"] = 5

        url = f"{endpoint}?{urllib.parse.urlencode(params)}"

        time.sleep(self.rate_limit_delay)
        raw_bytes = _safe_request(url)
        data = json.loads(raw_bytes.decode("utf-8"))

        records: List[Dict[str, Any]] = []
        hourly_data = data.get("hourly", {})
        times = pd.to_datetime(hourly_data.get("time", []), utc=True)

        for api_var in api_vars:
            canon_name = var_to_canon.get(api_var, api_var)
            vals = hourly_data.get(api_var)
            if vals is None and model:
                vals = hourly_data.get(f"{api_var}_{model}")
            vals = vals or []

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

    def query(
        self,
        locations: Optional[Union[Dict[str, float], Tuple[float, float], List[Any], str]] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        region: Optional[str] = None,
        variables: Optional[Union[List[str], Sequence[str], str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        past_days: Optional[int] = None,
        forecast_days: Optional[int] = None,
        model: Optional[str] = None,
        format: str = "wide",
    ) -> pd.DataFrame:
        """
        Query Open-Meteo weather parameters across one or more locations.

        Mathematical docstring:
        Input:
            locations: sequence of N point coordinates or presets
            variables: sequence of C variable names
            temporal: [start_date, end_date]
        Output:
            - Wide format: (T_timestamps, N_locations * C_variables) or (T_timestamps, C_variables)
            - Long format: (N_records, 6) with columns ['time', 'location', 'lat', 'lon', 'variable', 'value']

        Parameters
        ----------
        locations : dict, tuple, list, or str, optional
            Target location or locations.
        lat, lon : float, optional
            Direct coordinates.
        region : str, optional
            Preset region name (e.g. 'kenya').
        variables : list of str, optional
            List of variables or aliases.
        start_date, end_date : str or datetime, optional
            Date range ('YYYY-MM-DD').
        past_days : int, optional
            Number of past days (0 to 92).
        forecast_days : int, optional
            Number of forecast days (1 to 16).
        model : str, optional
            NWP model ('ecmwf_ifs025', 'gfs_seamless', 'icon_seamless', or None).
        format : str, default 'wide'
            'wide' (datetime index with variables as columns) or 'long'.

        Returns
        -------
        pd.DataFrame
            Clean hourly meteorological time series.
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
                model=model,
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
                        model=model,
                        location_name=loc["name"],
                    )
                    for loc in locs
                ]
                for f in as_completed(futures):
                    try:
                        all_records.extend(f.result())
                    except Exception as e:
                        logger.warning("Open-Meteo point query failed: %s", e)

        return self._reshape_output(all_records, format=format)

    def query_forecast(
        self,
        lat: float,
        lon: float,
        forecast_days: int = 7,
        variables: Optional[Union[List[str], Sequence[str]]] = None,
        model: Optional[str] = None,
        format: str = "wide",
    ) -> pd.DataFrame:
        """
        Query rolling operational weather forecast (1 to 16 days ahead) for a coordinate.
        """
        vars_to_query = list(variables) if variables else [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "precipitation_probability",
            "wind_speed_10m",
            "wind_gusts_10m",
            "cloud_cover",
            "et0_fao_evapotranspiration",
            "weather_code",
        ]
        return self.query(
            lat=lat,
            lon=lon,
            forecast_days=min(max(forecast_days, 1), 16),
            past_days=0,
            variables=vars_to_query,
            model=model,
            format=format,
        )

    def query_soil(
        self,
        lat: float,
        lon: float,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        past_days: Optional[int] = None,
        forecast_days: Optional[int] = None,
        format: str = "wide",
    ) -> pd.DataFrame:
        """
        Query vertical soil temperature and moisture profile (0-7cm and 7-28cm) for agricultural modeling.
        """
        soil_vars = [
            "soil_temperature_0_to_7cm",
            "soil_temperature_7_to_28cm",
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm",
            "et0_fao_evapotranspiration",
            "precipitation",
        ]
        return self.query(
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            past_days=past_days,
            forecast_days=forecast_days,
            variables=soil_vars,
            format=format,
        )


# ==============================================================================
# Functional Public API
# ==============================================================================

def query_open_meteo(
    locations: Optional[Union[Dict[str, float], Tuple[float, float], List[Any], str]] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    region: Optional[str] = None,
    variables: Optional[Union[List[str], Sequence[str]]] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    past_days: Optional[int] = None,
    forecast_days: Optional[int] = None,
    model: Optional[str] = None,
    format: str = "wide",
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Query Open-Meteo high-resolution global weather in a single function call.
    """
    client = OpenMeteo(cache_dir=cache_dir)
    return client.query(
        locations=locations,
        lat=lat,
        lon=lon,
        region=region,
        variables=list(variables) if variables else None,
        start_date=start_date,
        end_date=end_date,
        past_days=past_days,
        forecast_days=forecast_days,
        model=model,
        format=format,
    )


def query_open_meteo_forecast(
    lat: float,
    lon: float,
    forecast_days: int = 7,
    variables: Optional[Union[List[str], Sequence[str]]] = None,
    model: Optional[str] = None,
    format: str = "wide",
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Convenience function to retrieve a rolling operational weather forecast (1 to 16 days ahead).
    """
    client = OpenMeteo(cache_dir=cache_dir)
    return client.query_forecast(
        lat=lat,
        lon=lon,
        forecast_days=forecast_days,
        variables=list(variables) if variables else None,
        model=model,
        format=format,
    )


def query_open_meteo_soil(
    lat: float,
    lon: float,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    past_days: Optional[int] = None,
    forecast_days: Optional[int] = None,
    format: str = "wide",
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Convenience function to query vertical soil temperature and moisture profile for agricultural modeling.
    """
    client = OpenMeteo(cache_dir=cache_dir)
    return client.query_soil(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        past_days=past_days,
        forecast_days=forecast_days,
        format=format,
    )


# Aliases
OpenMeteoWrapper = OpenMeteo
get_open_meteo_data = query_open_meteo

__all__ = [
    "OpenMeteo",
    "OpenMeteoWrapper",
    "get_open_meteo_data",
    "OPEN_METEO_MODELS",
    "WMO_WEATHER_CODES",
    "decode_weather_code",
    "OPENMETEO_PARAMETERS",
    "OPENMETEO_LEXICON",
    "DEFAULT_OPENMETEO_VARS",
    "search_open_meteo_parameters",
    "list_open_meteo_parameters",
    "get_open_meteo_metadata",
    "query_open_meteo",
    "query_open_meteo_forecast",
    "query_open_meteo_soil",
]
