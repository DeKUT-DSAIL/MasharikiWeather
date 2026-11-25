"""
TAHMO DataHub Weather Station Extraction Engine.

Downloads high-resolution ground station observations from the Trans-African
Hydro-Meteorological Observatory (TAHMO) DataHub (https://datahub.tahmo.org).

Key Design Principles:
1. Non-Destructive Storage: Multi-sensor observations (e.g. rh_S001402, rh_S001403)
   are never averaged or collapsed; each sensor stream is preserved in dedicated columns.
2. Quality Control: Observations with quality flag != 1 are converted to NaN for controlled feeds.
3. Spatiotemporal Subsetting: Automated station discovery and filtering by bounding box.
4. Output Persistence: Stored as standardized Apache Parquet files on local disk.
"""

from __future__ import annotations

import concurrent.futures
import gc
import logging
import os
import random
import re
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth

try:
    import orjson
except ImportError:
    orjson = None

from .base import (
    AFRICA_BBOX,
    EAST_AFRICA_BBOX,
    KENYA_BBOX,
    BaseExtractor,
    clamp_max_workers,
    SpatialBoundingBox,
    TemporalRange,
)
from .credentials import CredentialsError, CredentialsManager, parse_date, parse_date_range
from .resilience import ExtractionEstimate, retry_with_backoff

logger = logging.getLogger("MasharikiWeather.TAHMO")

API_BASE_URL = "https://datahub.tahmo.org"
MAX_REQUEST_DAYS = 365
DATASETS = ("controlled", "raw")
SENSOR_COL_PATTERN = re.compile(r"^(?P<variable>[a-z]{2,3})_(?P<sensor>\w+)$")

# Bidirectional Cross-Source Harmonization Registry for TAHMO (33 Variables)
TAHMO_SHORT_TO_CANONICAL: Dict[str, str] = {
    # 33 primary short codes
    "ap": "surface_pressure",
    "cp": "convective_precipitation",
    "dw": "water_depth",
    "ec": "electrical_conductivity",
    "ew": "pore_water_electrical_conductivity",
    "ht": "humidity_sensor_temperature",
    "lb": "logger_battery_voltage",
    "ld": "lightning_distance",
    "le": "lightning_strike_count",
    "lp": "logger_pressure",
    "lt": "logger_temperature",
    "mp": "matric_potential",
    "pd": "precipitation_drop_count",
    "pr": "total_precipitation",
    "pt": "precipitation_tip_count",
    "pw": "permittivity_pore_water",
    "rh": "relative_humidity",
    "rn": "rain_rate",
    "sd": "sunshine_duration",
    "sm": "soil_moisture",
    "sr": "surface_solar_radiation_downwards",
    "tc": "canopy_temperature",
    "td": "2m_dewpoint_temperature",
    "te": "2m_temperature",
    "tp": "precipitation_temperature",
    "uv": "uv_index",
    "vp": "vapor_pressure",
    "wd": "wind_from_direction",
    "wg": "wind_speed_of_gust",
    "wl": "water_level",
    "wq": "water_discharge",
    "ws": "wind_speed",
    "wv": "water_velocity",
    # Backward compatibility and common aliases
    "temp": "2m_temperature",
    "temperature": "2m_temperature",
    "precip": "total_precipitation",
    "precipitation": "total_precipitation",
    "cumulative_precip": "convective_precipitation",
    "cumulative_precipitation": "convective_precipitation",
    "rad": "surface_solar_radiation_downwards",
    "solar_radiation": "surface_solar_radiation_downwards",
    "pres": "surface_pressure",
    "pressure": "surface_pressure",
    "dewpoint": "2m_dewpoint_temperature",
    "dew_point_temperature": "2m_dewpoint_temperature",
    "wind_gust": "wind_speed_of_gust",
    "gust": "wind_speed_of_gust",
    "lightning_events": "lightning_strike_count",
    "logger_battery": "logger_battery_voltage",
}

TAHMO_CANONICAL_TO_SHORT: Dict[str, str] = {
    # 33 canonical names to primary short codes
    "surface_pressure": "ap",
    "convective_precipitation": "cp",
    "cumulative_precipitation": "cp",
    "water_depth": "dw",
    "electrical_conductivity": "ec",
    "soil_electrical_conductivity": "ec",
    "pore_water_electrical_conductivity": "ew",
    "water_electrical_conductivity": "ew",
    "humidity_sensor_temperature": "ht",
    "relative_humidity_sensor_temperature": "ht",
    "logger_battery_voltage": "lb",
    "logger_battery": "lb",
    "lightning_distance": "ld",
    "lightning_average_distance": "ld",
    "lightning_strike_count": "le",
    "lightning_event_count": "le",
    "lightning_events": "le",
    "logger_pressure": "lp",
    "logger_internal_pressure": "lp",
    "logger_temperature": "lt",
    "logger_internal_temperature": "lt",
    "matric_potential": "mp",
    "soil_matric_potential": "mp",
    "precipitation_drop_count": "pd",
    "total_precipitation": "pr",
    "precipitation": "pr",
    "precip": "pr",
    "precipitation_tip_count": "pt",
    "tipping_bucket_tip_count": "pt",
    "permittivity_pore_water": "pw",
    "relative_humidity": "rh",
    "rain_rate": "rn",
    "rainfall_amount": "rn",
    "sunshine_duration": "sd",
    "soil_moisture": "sm",
    "volumetric_soil_moisture": "sm",
    "surface_solar_radiation_downwards": "sr",
    "solar_radiation": "sr",
    "rad": "sr",
    "canopy_temperature": "tc",
    "2m_dewpoint_temperature": "td",
    "dewpoint": "td",
    "dew_point_temperature": "td",
    "2m_temperature": "te",
    "temperature": "te",
    "temp": "te",
    "precipitation_temperature": "tp",
    "temperature_probe": "tp",
    "uv_index": "uv",
    "ultraviolet_radiation": "uv",
    "vapor_pressure": "vp",
    "wind_from_direction": "wd",
    "wind_speed_of_gust": "wg",
    "wind_gust": "wg",
    "gust": "wg",
    "water_level": "wl",
    "water_discharge": "wq",
    "water_volume_transport_in_river": "wq",
    "wind_speed": "ws",
    "water_velocity": "wv",
}

# Standard parameters catalog inventory for TAHMO (33 Variables)
DEFAULT_TAHMO_PARAMETERS: List[Dict[str, Any]] = [
    {
        "parameter": "surface_pressure",
        "canonical_name": "surface_pressure",
        "short_code": "ap",
        "standard_name": "surface_air_pressure",
        "units": "hPa",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Atmospheric surface pressure.",
    },
    {
        "parameter": "convective_precipitation",
        "canonical_name": "convective_precipitation",
        "short_code": "cp",
        "standard_name": "convective_precipitation_amount",
        "units": "mm",
        "physical_type": "PRECIPITATION",
        "resampling_method": "sum",
        "description": "Cumulative or convective precipitation amount.",
    },
    {
        "parameter": "water_depth",
        "canonical_name": "water_depth",
        "short_code": "dw",
        "standard_name": "depth_to_water_table",
        "units": "mm",
        "physical_type": "HYDROLOGICAL",
        "resampling_method": "mean",
        "description": "Surface water depth or depth to water level.",
    },
    {
        "parameter": "electrical_conductivity",
        "canonical_name": "electrical_conductivity",
        "short_code": "ec",
        "standard_name": "electrical_conductivity",
        "units": "mS cm-1",
        "physical_type": "ELECTRICAL",
        "resampling_method": "mean",
        "description": "Bulk soil electrical conductivity.",
    },
    {
        "parameter": "pore_water_electrical_conductivity",
        "canonical_name": "pore_water_electrical_conductivity",
        "short_code": "ew",
        "standard_name": "electrical_conductivity_of_water",
        "units": "mS cm-1",
        "physical_type": "ELECTRICAL",
        "resampling_method": "mean",
        "description": "Electrical conductivity of soil pore water.",
    },
    {
        "parameter": "humidity_sensor_temperature",
        "canonical_name": "humidity_sensor_temperature",
        "short_code": "ht",
        "standard_name": "temperature",
        "units": "degC",
        "physical_type": "DIAGNOSTIC",
        "resampling_method": "mean",
        "description": "Internal temperature of the relative humidity sensor element.",
    },
    {
        "parameter": "logger_battery_voltage",
        "canonical_name": "logger_battery_voltage",
        "short_code": "lb",
        "standard_name": "battery_voltage",
        "units": "V",
        "physical_type": "DIAGNOSTIC",
        "resampling_method": "mean",
        "description": "Data logger battery voltage or state of charge.",
    },
    {
        "parameter": "lightning_distance",
        "canonical_name": "lightning_distance",
        "short_code": "ld",
        "standard_name": "lightning_strike_distance",
        "units": "km",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Estimated distance to detected lightning strike event.",
    },
    {
        "parameter": "lightning_strike_count",
        "canonical_name": "lightning_strike_count",
        "short_code": "le",
        "standard_name": "number_of_lightning_flashes",
        "units": "counts",
        "physical_type": "DIAGNOSTIC",
        "resampling_method": "sum",
        "description": "Number of detected lightning strike events.",
    },
    {
        "parameter": "logger_pressure",
        "canonical_name": "logger_pressure",
        "short_code": "lp",
        "standard_name": "air_pressure",
        "units": "hPa",
        "physical_type": "DIAGNOSTIC",
        "resampling_method": "mean",
        "description": "Barometric pressure inside logger enclosure.",
    },
    {
        "parameter": "logger_temperature",
        "canonical_name": "logger_temperature",
        "short_code": "lt",
        "standard_name": "temperature",
        "units": "degC",
        "physical_type": "DIAGNOSTIC",
        "resampling_method": "mean",
        "description": "Internal temperature of data logger enclosure.",
    },
    {
        "parameter": "matric_potential",
        "canonical_name": "matric_potential",
        "short_code": "mp",
        "standard_name": "soil_matric_potential",
        "units": "kPa",
        "physical_type": "HYDROLOGICAL",
        "resampling_method": "mean",
        "description": "Soil water matric potential or suction tension.",
    },
    {
        "parameter": "precipitation_drop_count",
        "canonical_name": "precipitation_drop_count",
        "short_code": "pd",
        "standard_name": "number_of_hydrometeors",
        "units": "counts",
        "physical_type": "PRECIPITATION",
        "resampling_method": "sum",
        "description": "Optical disdrometer raindrop count.",
    },
    {
        "parameter": "total_precipitation",
        "canonical_name": "total_precipitation",
        "short_code": "pr",
        "standard_name": "precipitation_amount",
        "units": "mm",
        "physical_type": "PRECIPITATION",
        "resampling_method": "sum",
        "description": "Cumulative liquid precipitation measured by tipping bucket or optical rain gauge.",
    },
    {
        "parameter": "precipitation_tip_count",
        "canonical_name": "precipitation_tip_count",
        "short_code": "pt",
        "standard_name": "number_of_tipping_bucket_tips",
        "units": "counts",
        "physical_type": "PRECIPITATION",
        "resampling_method": "sum",
        "description": "Number of tipping bucket rain gauge tips.",
    },
    {
        "parameter": "permittivity_pore_water",
        "canonical_name": "permittivity_pore_water",
        "short_code": "pw",
        "standard_name": "relative_permittivity",
        "units": "1",
        "physical_type": "ELECTRICAL",
        "resampling_method": "mean",
        "description": "Relative dielectric permittivity of soil pore water.",
    },
    {
        "parameter": "relative_humidity",
        "canonical_name": "relative_humidity",
        "short_code": "rh",
        "standard_name": "relative_humidity",
        "units": "%",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Near-surface relative humidity measured at ~2m.",
    },
    {
        "parameter": "rain_rate",
        "canonical_name": "rain_rate",
        "short_code": "rn",
        "standard_name": "rainfall_rate",
        "units": "mm h-1",
        "physical_type": "PRECIPITATION",
        "resampling_method": "sum",
        "description": "Instantaneous rainfall rate or precipitation intensity.",
    },
    {
        "parameter": "sunshine_duration",
        "canonical_name": "sunshine_duration",
        "short_code": "sd",
        "standard_name": "duration_of_sunshine",
        "units": "s",
        "physical_type": "RADIATION",
        "resampling_method": "sum",
        "description": "Duration of sunshine exceeding direct solar irradiance threshold.",
    },
    {
        "parameter": "soil_moisture",
        "canonical_name": "soil_moisture",
        "short_code": "sm",
        "standard_name": "volume_fraction_of_water_in_soil",
        "units": "m3 m-3",
        "physical_type": "HYDROLOGICAL",
        "resampling_method": "mean",
        "description": "Volumetric soil water content.",
    },
    {
        "parameter": "surface_solar_radiation_downwards",
        "canonical_name": "surface_solar_radiation_downwards",
        "short_code": "sr",
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "units": "W m-2",
        "physical_type": "RADIATION",
        "resampling_method": "mean",
        "description": "Downwelling shortwave solar radiation flux.",
    },
    {
        "parameter": "canopy_temperature",
        "canonical_name": "canopy_temperature",
        "short_code": "tc",
        "standard_name": "canopy_temperature",
        "units": "degC",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Vegetation canopy or surface radiometric temperature.",
    },
    {
        "parameter": "2m_dewpoint_temperature",
        "canonical_name": "2m_dewpoint_temperature",
        "short_code": "td",
        "standard_name": "dew_point_temperature",
        "units": "degC",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Near-surface dew point temperature at ~2m elevation.",
    },
    {
        "parameter": "2m_temperature",
        "canonical_name": "2m_temperature",
        "short_code": "te",
        "standard_name": "air_temperature",
        "units": "degC",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Near-surface dry bulb air temperature at ~2m elevation.",
    },
    {
        "parameter": "precipitation_temperature",
        "canonical_name": "precipitation_temperature",
        "short_code": "tp",
        "standard_name": "temperature",
        "units": "degC",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Temperature of hydrometeors / precipitation sensor element.",
    },
    {
        "parameter": "uv_index",
        "canonical_name": "uv_index",
        "short_code": "uv",
        "standard_name": "ultraviolet_index",
        "units": "1",
        "physical_type": "RADIATION",
        "resampling_method": "mean",
        "description": "Solar ultraviolet radiation index.",
    },
    {
        "parameter": "vapor_pressure",
        "canonical_name": "vapor_pressure",
        "short_code": "vp",
        "standard_name": "water_vapor_pressure",
        "units": "hPa",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Atmospheric water vapor partial pressure.",
    },
    {
        "parameter": "wind_from_direction",
        "canonical_name": "wind_from_direction",
        "short_code": "wd",
        "standard_name": "wind_from_direction",
        "units": "degrees",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Direction from which the wind is blowing (0-360 degrees).",
    },
    {
        "parameter": "wind_speed_of_gust",
        "canonical_name": "wind_speed_of_gust",
        "short_code": "wg",
        "standard_name": "wind_speed_of_gust",
        "units": "m s-1",
        "physical_type": "PHYSICAL",
        "resampling_method": "max",
        "description": "Peak instantaneous horizontal wind gust speed.",
    },
    {
        "parameter": "water_level",
        "canonical_name": "water_level",
        "short_code": "wl",
        "standard_name": "water_surface_height_above_reference_datum",
        "units": "m",
        "physical_type": "HYDROLOGICAL",
        "resampling_method": "mean",
        "description": "Surface water level or water table elevation.",
    },
    {
        "parameter": "water_discharge",
        "canonical_name": "water_discharge",
        "short_code": "wq",
        "standard_name": "water_volume_transport_in_river_channel",
        "units": "m3 s-1",
        "physical_type": "HYDROLOGICAL",
        "resampling_method": "mean",
        "description": "Volumetric stream or river water discharge.",
    },
    {
        "parameter": "wind_speed",
        "canonical_name": "wind_speed",
        "short_code": "ws",
        "standard_name": "wind_speed",
        "units": "m s-1",
        "physical_type": "PHYSICAL",
        "resampling_method": "mean",
        "description": "Horizontal scalar wind speed at ~2m mast.",
    },
    {
        "parameter": "water_velocity",
        "canonical_name": "water_velocity",
        "short_code": "wv",
        "standard_name": "water_velocity",
        "units": "m s-1",
        "physical_type": "HYDROLOGICAL",
        "resampling_method": "mean",
        "description": "Surface or channel water flow velocity.",
    },
]


def canonicalize_tahmo_variable(var: str) -> str:
    """Map raw TAHMO short code to canonical descriptive name."""
    return TAHMO_SHORT_TO_CANONICAL.get(var, var)


def tahmo_canonical_to_short(var: str) -> str:
    """Map canonical descriptive name to primary TAHMO short code."""
    return TAHMO_CANONICAL_TO_SHORT.get(var, var)


class TAHMOExtractor(BaseExtractor):
    """
    Client and extraction engine for TAHMO ground station observations.
    """

    extractor_name: str = "tahmo"
    network_name: str = "tahmo"

    def __init__(
        self,
        output_dir: Union[str, Path] = "./mashariki-data/obs/tahmo",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        env_file: Optional[Union[str, Path]] = None,
        dataset: str = "controlled",
        request_delay: float = 0.1,
        retry_delay: float = 120.0,
        transient_retry_delay: float = 5.0,
        retries: int = 3,
        verify_ssl: bool = True,
        max_workers: Optional[int] = 4,
    ) -> None:
        """
        Initialize TAHMO ground station extraction engine with authentication and output cache.

        Parameters
        ----------
        output_dir : Union[str, Path], default "./mashariki-data/obs/tahmo"
            Directory path where extracted station observations and metadata will be saved.
        api_key : Optional[str]
            TAHMO DataHub API key.
        api_secret : Optional[str]
            TAHMO DataHub API secret.
        env_file : Optional[Union[str, Path]]
            Optional path to custom .env file for BYOK credential resolution.
        dataset : str, default "controlled"
            Dataset tier ("controlled" or "raw").
        request_delay : float, default 0.1
            Inter-request pacing delay in seconds between HTTP queries to protect legacy servers.
        retry_delay : float, default 120.0
            Delay in seconds to wait before retrying a failed HTTP request across failure modes.
            Can also be set via the TAHMO_RETRY_DELAY environment variable.
        retries : int, default 3
            Maximum number of retry attempts for failed HTTP requests. Can also be set
            via the TAHMO_RETRIES environment variable.
        verify_ssl : bool, default True
            Whether to enforce SSL/TLS certificate verification. Can also be overridden
            via the TAHMO_VERIFY_SSL environment variable ('false', '0', 'no').
        max_workers : Optional[int], default 4
            Maximum number of concurrent worker threads for batch extraction. Kept at a maximum
            of 4 to protect legacy TAHMO server infrastructure while accelerating retrieval.
            Pass 1 or None for serial execution.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, received: {dataset}")

        self.credentials = CredentialsManager.get_tahmo_credentials(
            api_key=api_key, api_secret=api_secret, env_file=env_file
        )
        self.api_key = self.credentials["api_key"]
        self.api_secret = self.credentials["api_secret"]
        self.env_file = env_file
        self.auth = HTTPBasicAuth(self.credentials["api_key"], self.credentials["api_secret"])
        self.request_delay = max(0.0, float(request_delay))

        # Configurable retry delay and retry attempts
        env_retry_delay = os.getenv("TAHMO_RETRY_DELAY", "").strip()
        if env_retry_delay:
            try:
                self.retry_delay = max(0.0, float(env_retry_delay))
            except ValueError:
                self.retry_delay = max(0.0, float(retry_delay))
        else:
            self.retry_delay = max(0.0, float(retry_delay))

        env_transient_delay = os.getenv("TAHMO_TRANSIENT_RETRY_DELAY", "").strip()
        if env_transient_delay:
            try:
                self.transient_retry_delay = max(0.0, float(env_transient_delay))
            except ValueError:
                self.transient_retry_delay = max(0.0, float(transient_retry_delay))
        else:
            self.transient_retry_delay = max(0.0, float(transient_retry_delay))

        env_retries = os.getenv("TAHMO_RETRIES", "").strip()
        if env_retries:
            try:
                self.retries = max(1, int(env_retries))
            except ValueError:
                self.retries = max(1, int(retries))
        else:
            self.retries = max(1, int(retries))

        # SSL verification configuration with environment variable override
        env_verify = os.getenv("TAHMO_VERIFY_SSL", "").strip().lower()
        if env_verify in ("0", "false", "no"):
            self.verify_ssl = False
        elif env_verify in ("1", "true", "yes"):
            self.verify_ssl = True
        else:
            self.verify_ssl = bool(verify_ssl)

        if not self.verify_ssl:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL certificate verification is disabled for TAHMO DataHub.")

        self.max_workers = clamp_max_workers(max_workers, default=4, source="tahmo")
        self._local = threading.local()
        self.failed_stations: Dict[str, str] = {}

    def _get_thread_session(self) -> requests.Session:
        """Get or create thread-local requests.Session to guarantee thread safety across concurrent workers."""
        if not hasattr(self._local, "session"):
            session = requests.Session()
            session.auth = self.auth
            adapter = HTTPAdapter(
                pool_connections=8,
                pool_maxsize=8,
                max_retries=False,
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._local.session = session
        return self._local.session

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute authenticated HTTP GET request with retries, configurable retry delay,
        and respectful inter-request delay to protect legacy TAHMO server infrastructure.
        """
        url = f"{API_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
        session = self._get_thread_session()
        max_attempts = retries if retries is not None else self.retries
        for attempt in range(1, max_attempts + 1):
            try:
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                resp = session.get(url, params=params, timeout=60, verify=self.verify_ssl)
                if resp.status_code == 200:
                    if orjson is not None and isinstance(resp.content, (bytes, bytearray, str)):
                        return orjson.loads(resp.content)
                    return resp.json()
                elif resp.status_code in (401, 403):
                    raise CredentialsError("TAHMO DataHub", f"Authentication failed (HTTP {resp.status_code})")
                elif resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_time = float(retry_after) if retry_after else self.retry_delay
                    logger.warning(
                        "TAHMO rate limit (HTTP 429). Waiting %.1f s before retry (attempt %d/%d)",
                        sleep_time,
                        attempt,
                        max_attempts,
                    )
                    if attempt < max_attempts:
                        time.sleep(sleep_time)
                        continue
                    else:
                        resp.raise_for_status()
                else:
                    logger.warning(
                        "TAHMO request %s failed with HTTP %d (attempt %d/%d)",
                        url,
                        resp.status_code,
                        attempt,
                        max_attempts,
                    )
                    if attempt == max_attempts:
                        resp.raise_for_status()
            except (CredentialsError, KeyboardInterrupt):
                raise
            except Exception as e:
                if attempt == max_attempts:
                    raise
                logger.warning(
                    "Network retry on %s: %s (attempt %d/%d)",
                    url,
                    e,
                    attempt,
                    max_attempts,
                )

            if attempt < max_attempts:
                transient_sleep = min(
                    self.retry_delay,
                    self.transient_retry_delay * (3 ** (attempt - 1)),
                )
                logger.info(
                    "Waiting %.1f s before transient retry attempt %d/%d on %s",
                    transient_sleep,
                    attempt + 1,
                    max_attempts,
                    url,
                )
                time.sleep(transient_sleep)
        raise RuntimeError(f"Failed to fetch {url} after {max_attempts} attempts.")

    def get_stations(
        self,
        bbox: Optional[SpatialBoundingBox] = None,
    ) -> pd.DataFrame:
        """
        Retrieve list of available weather stations, optionally filtered by spatial bounding box.

        Mathematical docstring:
        Output table shape: (N_stations, C_metadata)
        where columns include [
            'station_code', 'name', 'lat', 'lon', 'elevation', 'country',
            'status', 'start_date'
        ].
        """
        raw = self._request("services/assets/v2/stations", params={"sort": "code"})
        stations_list = raw.get("data", raw) if isinstance(raw, dict) else raw
        rows: List[Dict[str, Any]] = []
        for s in stations_list:
            if not isinstance(s, dict):
                continue
            loc = s.get("location") or {}
            lat = loc.get("latitude")
            lon = loc.get("longitude")
            elev = loc.get("elevation") or loc.get("elevationmsl")

            if lat is None or lon is None:
                continue

            lat_f = float(lat)
            lon_f = float(lon)

            if bbox is not None and not bbox.contains(lat_f, lon_f):
                continue

            country_code = loc.get("countrycode") or s.get("countrycode") or s.get("country")

            start_raw = (
                s.get("installationdate")
                or s.get("installation_date")
                or s.get("created")
                or s.get("since_date")
                or s.get("start_date")
                or loc.get("installationdate")
            )
            name_val = loc.get("name") or s.get("name") or loc.get("city") or s.get("label") or s.get("code")

            if start_raw:
                start_dt = pd.to_datetime(start_raw)
            else:
                start_dt = pd.Timestamp("2014-01-01")

            rows.append({
                "station_code": s.get("code"),
                "name": name_val,
                "lat": lat_f,
                "lon": lon_f,
                "elevation": float(elev) if elev is not None else 0.0,
                "country": str(country_code).upper() if country_code else "KE",
                "status": s.get("status"),
                "start_date": start_dt,
            })

        df = pd.DataFrame(
            rows,
            columns=[
                "station_code",
                "name",
                "lat",
                "lon",
                "elevation",
                "country",
                "status",
                "start_date",
            ],
        )
        return df.sort_values(by="station_code").reset_index(drop=True)

    def _parse_measurements_fast(
        self,
        payload: Dict[str, Any],
        variables: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Vectorized parser converting raw TAHMO InfluxDB JSON into a non-destructive flat DataFrame.

        Mathematical docstring:
        Input:
            payload: Dict[str, Any] containing InfluxDB measurement series of shape (N_raw_obs, C_fields)
                where C_fields includes ['time', 'variable', 'sensor', 'value', 'quality'].
            variables: Optional sequence of C_req requested canonical or short parameter codes.
        Output:
            flat_df: pd.DataFrame of shape (N_valid_obs, 3) with columns ['time', 'col', 'value']
                where 'col' is formatted as '<variable>_<sensor>' or '<variable>'.
        """
        if not isinstance(payload, dict) or "results" not in payload:
            return pd.DataFrame(columns=["time", "col", "value"])

        frames: List[pd.DataFrame] = []
        for res in payload.get("results", []):
            if "error" in res:
                logger.warning("TAHMO InfluxDB reported series error: %s", res["error"])
                continue
            for s in res.get("series", []):
                cols = s.get("columns", [])
                vals = s.get("values", [])
                if not cols or not vals:
                    continue

                raw_df = pd.DataFrame(vals, columns=cols)
                if "time" not in raw_df.columns or "value" not in raw_df.columns or "variable" not in raw_df.columns:
                    continue

                if variables:
                    raw_df = raw_df[raw_df["variable"].isin(variables)]
                if raw_df.empty:
                    continue

                # Quality control filter: non-quality 1 values set to NaN
                if self.dataset == "controlled" and "quality" in raw_df.columns:
                    raw_df.loc[raw_df["quality"] != 1, "value"] = np.nan

                # Coerce values safely to numeric float32
                raw_df["value"] = pd.to_numeric(raw_df["value"], errors="coerce").astype(np.float32)

                # Defensive sensor column naming (never create 'pr_None' or 'pr_nan')
                if "sensor" in raw_df.columns:
                    has_sensor = (
                        raw_df["sensor"].notna()
                        & (raw_df["sensor"].astype(str).str.strip() != "")
                        & (raw_df["sensor"].astype(str) != "None")
                        & (raw_df["sensor"].astype(str) != "nan")
                    )
                    raw_df["col"] = np.where(
                        has_sensor,
                        raw_df["variable"].astype(str) + "_" + raw_df["sensor"].astype(str),
                        raw_df["variable"].astype(str),
                    )
                else:
                    raw_df["col"] = raw_df["variable"].astype(str)

                frames.append(raw_df[["time", "col", "value"]])

        if not frames:
            return pd.DataFrame(columns=["time", "col", "value"])

        return pd.concat(frames, ignore_index=True)

    def extract_station(
        self,
        station_code: str,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        variables: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Download measurements for a single station with non-destructive sensor preservation.

        Mathematical docstring:
        Output table shape: (T_timestamps, C_sensors)
        where column headers are named '<variable>_<sensor_id>' or '<variable>'
        to guarantee zero data loss.
        """
        s_date, e_date = parse_date_range(start_date, end_date)
        if s_date >= e_date:
            raise ValueError(f"start_date ({s_date}) must be prior to end_date ({e_date})")

        endpoint = f"services/measurements/v2/stations/{station_code}/measurements/{self.dataset}"
        flat_frames: List[pd.DataFrame] = []

        filter_vars = None if (variables is None or variables == "all" or (isinstance(variables, (list, tuple)) and "all" in variables)) else variables
        if filter_vars is not None:
            filter_vars = [tahmo_canonical_to_short(v) for v in filter_vars]
        cur = s_date
        while cur < e_date:
            next_step = min(cur + timedelta(days=MAX_REQUEST_DAYS), e_date)
            params: Dict[str, Any] = {
                "start": cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": next_step.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            # Throughput optimization: Omit "variable" param when requesting all variables to pull all 33 variables in 1 call
            if filter_vars and len(filter_vars) == 1:
                params["variable"] = filter_vars[0]

            resp_json = self._request(endpoint, params=params)
            chunk_df = self._parse_measurements_fast(resp_json, filter_vars)
            if not chunk_df.empty:
                flat_frames.append(chunk_df)
            cur = next_step + timedelta(seconds=1)

        if not flat_frames:
            return pd.DataFrame()

        combined = pd.concat(flat_frames, ignore_index=True)
        combined["time"] = pd.to_datetime(combined["time"], utc=True)
        combined = combined.drop_duplicates(subset=["time", "col"], keep="last")
        pivoted = combined.pivot(index="time", columns="col", values="value").astype(np.float32)
        pivoted.index.name = "time"
        return pivoted

    def _parse_measurements_payload(
        self,
        payload: Dict[str, Any],
        variables: Optional[Sequence[str]],
        series_holder: Dict[str, List[Any]],
    ) -> None:
        """Extract measurement rows respecting quality control flags."""
        results = payload.get("results", [])
        for res in results:
            for s in res.get("series", []):
                cols = s.get("columns", [])
                vals = s.get("values", [])
                if not cols or not vals:
                    continue

                idx_t = cols.index("time")
                idx_q = cols.index("quality")
                idx_v = cols.index("variable")
                idx_s = cols.index("sensor")
                idx_val = cols.index("value")

                for row in vals:
                    var_code = row[idx_v]
                    if variables and var_code not in variables:
                        continue
                    q_flag = row[idx_q]
                    val = row[idx_val] if q_flag == 1 or self.dataset == "raw" else np.nan
                    sensor_id = row[idx_s]
                    timestamp = row[idx_t]

                    series_holder.setdefault(var_code, []).append((timestamp, val, sensor_id, q_flag))

    def _build_non_destructive_dataframe(self, series_holder: Dict[str, List[Any]]) -> pd.DataFrame:
        """
        Assemble observations into non-destructive DataFrame.
        When multiple sensors measure the same variable, each is preserved
        in a distinct '<variable>_<sensor>' column.
        """
        series_list: List[pd.DataFrame] = []
        for var_code, obs_list in series_holder.items():
            if not obs_list:
                continue

            sensors = list({x[2] for x in obs_list if x[2]})
            if len(sensors) > 1:
                for sensor in sensors:
                    sub = [x for x in obs_list if x[2] == sensor]
                    idx = pd.to_datetime([x[0] for x in sub], utc=True)
                    s = pd.Series([x[1] for x in sub], index=idx, dtype=np.float32)
                    s = s.loc[~s.index.duplicated(keep="last")]
                    series_list.append(s.to_frame(f"{var_code}_{sensor}"))
            else:
                idx = pd.to_datetime([x[0] for x in obs_list], utc=True)
                col_name = f"{var_code}_{sensors[0]}" if sensors else var_code
                s = pd.Series([x[1] for x in obs_list], index=idx, dtype=np.float32)
                s = s.loc[~s.index.duplicated(keep="last")]
                series_list.append(s.to_frame(col_name))

        if not series_list:
            return pd.DataFrame()

        merged = pd.concat(series_list, axis=1, sort=True)
        merged.index.name = "time"
        return merged

    def generate_metadata_catalogs(
        self,
        stations_df: Optional[pd.DataFrame] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path]:
        """
        Generate standardized metadata catalogs:
        - installation.parquet: Station inventory (station_id, name, latitude, longitude, elevation, etc.)
        - parameters.parquet: Meteorological parameters inventory (parameter, units, physical type, etc.)

        Mathematical docstring:
        Output catalogs:
          installation: (N_stations, C_metadata) -> {output_dir}/metadata/installation.parquet
          parameters: (N_parameters, C_param_info) -> {output_dir}/metadata/parameters.parquet

        Parameters
        ----------
        stations_df : Optional[pd.DataFrame]
            Pre-queried station metadata DataFrame. If None, queries get_stations().
        output_dir : Optional[Union[str, Path]]
            Directory to save metadata catalogs. Defaults to self.output_dir / "metadata".

        Returns
        -------
        Tuple[Path, Path]
            Paths to (installation.parquet, parameters.parquet).
        """
        meta_dir = Path(output_dir) if output_dir is not None else (self.output_dir / "metadata")
        meta_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build installation catalog
        df_st = stations_df if stations_df is not None else self.get_stations()
        install_rows: List[Dict[str, Any]] = []
        for _, row in df_st.iterrows():
            st_id = str(row.get("station_code") or row.get("station_id") or row.get("id") or "")
            start_val = row.get("start_date")
            if start_val is None or pd.isna(start_val):
                start_dt = pd.Timestamp("2014-01-01")
            else:
                start_dt = pd.to_datetime(start_val)

            st_name = str(row.get("name") or row.get("station_name") or st_id)

            elev_val = row.get("elevation")
            if elev_val is None or pd.isna(elev_val):
                elev_val = row.get("elevationmsl")
            if elev_val is None or pd.isna(elev_val):
                elev_val = row.get("elev")
            try:
                f_elev = float(elev_val)
                elevation_clean = 0.0 if np.isnan(f_elev) else f_elev
            except (ValueError, TypeError):
                elevation_clean = 0.0

            install_rows.append({
                "station_id": st_id,
                "name": st_name,
                "latitude": float(row.get("lat") or row.get("latitude") or 0.0),
                "longitude": float(row.get("lon") or row.get("longitude") or 0.0),
                "elevation": elevation_clean,
                "network": "tahmo",
                "country": str(row.get("country")) if row.get("country") else "KE",
                "start_date": start_dt,
            })
        install_df = pd.DataFrame(install_rows).drop_duplicates(subset=["station_id"])
        install_path = meta_dir / "installation.parquet"
        install_df.to_parquet(install_path, index=False)

        # Also persist stations.parquet for compatibility
        stations_path = meta_dir / "stations.parquet"
        install_df.to_parquet(stations_path, index=False)

        # 2. Build parameters catalog
        param_df = pd.DataFrame(DEFAULT_TAHMO_PARAMETERS).drop(columns=["canonical_name"], errors="ignore")
        if "network" not in param_df.columns:
            param_df["network"] = "tahmo"
        if "source" not in param_df.columns:
            param_df["source"] = "tahmo"
        param_path = meta_dir / "parameters.parquet"
        param_df.to_parquet(param_path, index=False)

        logger.info(
            "Generated metadata catalogs at %s: installation.parquet (%d stations), parameters.parquet (%d variables)",
            meta_dir, len(install_df), len(param_df),
        )
        return install_path, param_path

    def open_zarr(
        self,
        zarr_path: Union[str, Path],
        **kwargs: Any,
    ) -> Any:
        """
        Lazily open a TAHMO Zarr dataset from disk.

        Parameters
        ----------
        zarr_path : Union[str, Path]
            Path to existing TAHMO Zarr store.

        Returns
        -------
        xr.Dataset
            Lazy xarray Dataset.
        """
        import xarray as xr
        return xr.open_zarr(str(zarr_path), **kwargs)

    @staticmethod
    def _split_date_range_by_year(
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
    ) -> List[Tuple[int, pd.Timestamp, pd.Timestamp]]:
        """
        Partition a temporal range [start_dt, end_dt] into calendar-year slices.

        Mathematical docstring:
        Input: [t_start, t_end]
        Output: List of (year, t_y_start, t_y_end) where t_y_start < t_y_end.
        """
        tz = getattr(start_dt, "tz", None) or getattr(end_dt, "tz", None)
        s_cmp = start_dt.tz_localize(None) if getattr(start_dt, "tz", None) is not None else start_dt
        e_cmp = end_dt.tz_localize(None) if getattr(end_dt, "tz", None) is not None else end_dt
        slices: List[Tuple[int, pd.Timestamp, pd.Timestamp]] = []
        for year in range(s_cmp.year, e_cmp.year + 1):
            y_start = max(s_cmp, pd.Timestamp(year=year, month=1, day=1))
            y_end = min(e_cmp, pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59))
            if tz is not None:
                y_start = y_start.tz_localize(tz)
                y_end = y_end.tz_localize(tz)
            if y_start < y_end:
                slices.append((year, y_start, y_end))
        return slices

    @staticmethod
    def _station_df_to_semi_wide(df: pd.DataFrame, station_code: str) -> pd.DataFrame:
        """
        Convert a single station DataFrame into O(1)-width semi-wide format.

        Mathematical docstring:
        Input: DataFrame indexed by time with sensor-qualified columns (e.g. 'te_S001402')
               or plain variable columns (e.g. 'te').
        Output: Semi-wide DataFrame with columns ['time', 'station_id', 'sensor_id', ...vars...]
                where plain columns without a sensor suffix receive sensor_id='primary'.
        """
        if df.empty:
            return pd.DataFrame(columns=["time", "station_id", "sensor_id"])

        work = df.copy()
        if "time" not in work.columns:
            if "index" in work.columns:
                work = work.rename(columns={"index": "time"})
            else:
                work = work.rename_axis("time").reset_index()
        work["time"] = pd.to_datetime(work["time"], utc=True)

        if "sensor_id" in work.columns:
            if "station_id" not in work.columns:
                work["station_id"] = station_code
            return work

        # Group columns by sensor_id
        sensor_groups: Dict[str, Dict[str, str]] = {}
        for col in work.columns:
            if col in ("time", "station_id", "station"):
                continue
            m = SENSOR_COL_PATTERN.match(str(col))
            if m:
                var_name = m.group("variable")
                sid = m.group("sensor")
            else:
                var_name = str(col)
                sid = "primary"
            sensor_groups.setdefault(sid, {})[col] = var_name

        if not sensor_groups:
            return pd.DataFrame(columns=["time", "station_id", "sensor_id"])

        parts: List[pd.DataFrame] = []
        for sid in sorted(sensor_groups.keys()):
            col_map = sensor_groups[sid]
            sub_cols = list(col_map.keys())
            sub = work[["time"] + sub_cols].rename(columns=col_map)
            var_cols = list(col_map.values())
            # Drop timestamps where this sensor has no measurements at all
            sub = sub.dropna(subset=var_cols, how="all")
            if sub.empty:
                continue
            sub["station_id"] = station_code
            sub["sensor_id"] = sid
            for vc in var_cols:
                sub[vc] = pd.to_numeric(sub[vc], errors="coerce").astype(np.float32)
            parts.append(sub)

        if not parts:
            return pd.DataFrame(columns=["time", "station_id", "sensor_id"])

        combined = pd.concat(parts, axis=0, ignore_index=True, sort=False)
        combined = combined.sort_values(by=["time", "sensor_id"]).reset_index(drop=True)
        return combined

    @classmethod
    def _write_yearly_parquet_streaming(
        cls,
        station_files: Sequence[Tuple[str, Path]],
        dest_path: Path,
        compression: str = "snappy",
    ) -> Optional[Path]:
        """
        Stream semi-wide per-station Parquet files into a single yearly Parquet file in O(1 station) RAM.

        Mathematical docstring:
        Input: Sequence of (station_id, Path) in semi-wide format.
        Output: Single Parquet at dest_path with O(1) schema width:
                ['time', 'station_id', 'sensor_id', ...sorted variable shortcodes...].
        """
        if not station_files:
            return None

        import pyarrow as pa
        import pyarrow.parquet as pq

        # Pass 1: inspect schemas only (zero data rows loaded) to build O(1)-width variable set
        param_cols_set: Set[str] = set()
        valid_files: List[Tuple[str, Path]] = []
        for code, pf in station_files:
            if not pf.exists():
                continue
            try:
                sc = pq.read_schema(pf)
            except Exception as err:
                logger.warning("Skipping unreadable station cache %s: %s", pf, err)
                continue
            for name in sc.names:
                if name in ("time", "index", "station_id", "station", "sensor_id"):
                    continue
                m = SENSOR_COL_PATTERN.match(name)
                param_cols_set.add(m.group("variable") if m else name)
            valid_files.append((code, pf))

        if not valid_files:
            return None

        param_cols = sorted(param_cols_set)
        time_type = pa.timestamp("us", tz="UTC")
        fields: List[pa.Field] = [
            pa.field("time", time_type),
            pa.field("station_id", pa.string()),
            pa.field("sensor_id", pa.string()),
        ] + [pa.field(col, pa.float32()) for col in param_cols]
        unified_schema = pa.schema(fields)

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(".parquet.tmp")

        # Pass 2: stream one station at a time into ParquetWriter (O(1 station) memory)
        writer: Optional[pq.ParquetWriter] = None
        try:
            writer = pq.ParquetWriter(str(tmp_path), unified_schema, compression=compression)
            for idx, (code, pf) in enumerate(valid_files, start=1):
                df = pd.read_parquet(pf)
                if df.empty:
                    del df
                    continue
                if "sensor_id" not in df.columns:
                    df = cls._station_df_to_semi_wide(df, code)
                    if df.empty:
                        del df
                        continue
                else:
                    if "time" not in df.columns:
                        df = df.reset_index()
                    df["time"] = pd.to_datetime(df["time"], utc=True)

                n_rows = len(df)
                arrays: List[pa.Array] = [
                    pa.array(df["time"], type=time_type),
                    pa.array([code] * n_rows, type=pa.string()),
                    pa.array(df["sensor_id"].astype(str).tolist(), type=pa.string()),
                ]
                for col in param_cols:
                    if col in df.columns:
                        arrays.append(pa.array(df[col].to_numpy(dtype=np.float32, copy=False), type=pa.float32()))
                    else:
                        arrays.append(pa.nulls(n_rows, type=pa.float32()))

                table = pa.Table.from_arrays(arrays, schema=unified_schema)
                writer.write_table(table)
                del df, table, arrays
                pa.default_memory_pool().release_unused()
                if idx % 20 == 0:
                    gc.collect()
        finally:
            if writer is not None:
                writer.close()
            gc.collect()

        tmp_path.replace(dest_path)
        return dest_path

    @staticmethod
    def _write_zarr_ordinal_slots(
        zarr_path: Path,
        station_files: Optional[Sequence[Tuple[str, Path]]] = None,
        dataframes: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Optional[Path]:
        """
        Write a multi-station Zarr store using ordinal sensor slots and incremental per-variable writes.

        Adapted from AI4Weather build_dataset.py:
        - Single-sensor variables without sensor IDs: (time, station)
        - Multi-sensor / sensor-qualified variables: (time, station, {var}_slot) + {var}_sensor_id(station, {var}_slot)
        """
        import xarray as xr

        # Gather per-station time indices and sensor inventories
        frames_by_station: Dict[str, pd.DataFrame] = {}
        if station_files:
            for code, pf in station_files:
                if pf.exists():
                    df_st = pd.read_parquet(pf)
                    if not df_st.empty:
                        if code in frames_by_station:
                            frames_by_station[code] = pd.concat(
                                [frames_by_station[code], df_st], ignore_index=True, sort=False
                            )
                        else:
                            frames_by_station[code] = df_st
        elif dataframes:
            for code, df_st in dataframes.items():
                if df_st is not None and not df_st.empty:
                    frames_by_station[code] = df_st

        if not frames_by_station:
            return None

        station_codes = sorted(frames_by_station.keys())
        master_time = pd.DatetimeIndex([], tz="UTC")
        sensor_table: Dict[str, Dict[str, List[str]]] = {}

        # Normalize each frame to time-indexed and build sensor_table
        norm_frames: Dict[str, pd.DataFrame] = {}
        for code in station_codes:
            df_st = frames_by_station[code].copy()
            if "time" in df_st.columns:
                df_st["time"] = pd.to_datetime(df_st["time"], utc=True)
            else:
                df_st = df_st.rename_axis("time").reset_index()
                df_st["time"] = pd.to_datetime(df_st["time"], utc=True)

            if "sensor_id" in df_st.columns:
                # Semi-wide format: pivot per variable and sensor_id
                master_time = master_time.union(pd.DatetimeIndex(df_st["time"].unique()))
                var_cols = [c for c in df_st.columns if c not in ("time", "station_id", "station", "sensor_id")]
                sids = sorted(df_st["sensor_id"].astype(str).unique().tolist())
                for v in var_cols:
                    for sid in sids:
                        if df_st.loc[df_st["sensor_id"].astype(str) == sid, v].notna().any():
                            norm_sid = "" if sid == "primary" else sid
                            sensor_table.setdefault(v, {}).setdefault(code, [])
                            if norm_sid not in sensor_table[v][code]:
                                sensor_table[v][code].append(norm_sid)
                norm_frames[code] = df_st
            else:
                df_st = df_st.set_index("time").sort_index()
                df_st = df_st.loc[~df_st.index.duplicated(keep="last")]
                master_time = master_time.union(df_st.index)
                for col in df_st.columns:
                    m = SENSOR_COL_PATTERN.match(str(col))
                    var_name = m.group("variable") if m else str(col)
                    sid = m.group("sensor") if m else ""
                    sensor_table.setdefault(var_name, {}).setdefault(code, [])
                    if sid not in sensor_table[var_name][code]:
                        sensor_table[var_name][code].append(sid)
                norm_frames[code] = df_st

        master_time = master_time.sort_values()
        master_time_naive = master_time.tz_localize(None) if master_time.tz is not None else master_time
        for v in sensor_table:
            for code in sensor_table[v]:
                sensor_table[v][code].sort()

        if not sensor_table or len(master_time) == 0:
            return None

        if zarr_path.exists():
            shutil.rmtree(zarr_path, ignore_errors=True)

        # Write one variable at a time to Zarr (O(1 variable) memory)
        first_write = True
        for var in sorted(sensor_table.keys()):
            var_table = sensor_table[var]
            max_slots = max(len(sids) for sids in var_table.values())
            only_plain = all(sids == [""] for sids in var_table.values())
            is_2d = only_plain and (max_slots == 1)

            n_t = len(master_time)
            n_s = len(station_codes)

            if is_2d:
                matrix = np.full((n_t, n_s), np.nan, dtype=np.float32)
                for s_idx, code in enumerate(station_codes):
                    f_st = norm_frames[code]
                    if "sensor_id" in f_st.columns:
                        if var in f_st.columns:
                            sub = f_st[f_st[var].notna()].drop_duplicates(subset=["time"], keep="last")
                            if not sub.empty:
                                locs = master_time.get_indexer(pd.DatetimeIndex(sub["time"]))
                                valid = locs >= 0
                                matrix[locs[valid], s_idx] = sub[var].to_numpy(dtype=np.float32)[valid]
                    elif var in f_st.columns:
                        locs = master_time.get_indexer(f_st.index)
                        valid = locs >= 0
                        matrix[locs[valid], s_idx] = f_st[var].to_numpy(dtype=np.float32)[valid]
                ds_var = xr.Dataset(
                    data_vars={var: (("time", "station"), matrix)},
                    coords={"time": master_time_naive, "station": station_codes},
                )
            else:
                slot_dim = f"{var}_slot"
                matrix = np.full((n_t, n_s, max_slots), np.nan, dtype=np.float32)
                sensor_ids = np.full((n_s, max_slots), "", dtype="<U32")
                for s_idx, code in enumerate(station_codes):
                    f_st = norm_frames[code]
                    for slot_idx, sid in enumerate(var_table.get(code, [])):
                        sensor_ids[s_idx, slot_idx] = sid if sid else "primary"
                        if "sensor_id" in f_st.columns:
                            target_sid = sid if sid else "primary"
                            sub = f_st[f_st["sensor_id"].astype(str) == target_sid].drop_duplicates(subset=["time"], keep="last")
                            if not sub.empty and var in sub.columns:
                                locs = master_time.get_indexer(pd.DatetimeIndex(sub["time"]))
                                valid = locs >= 0
                                matrix[locs[valid], s_idx, slot_idx] = sub[var].to_numpy(dtype=np.float32)[valid]
                        else:
                            col = f"{var}_{sid}" if sid else var
                            if col in f_st.columns:
                                locs = master_time.get_indexer(f_st.index)
                                valid = locs >= 0
                                matrix[locs[valid], s_idx, slot_idx] = f_st[col].to_numpy(dtype=np.float32)[valid]

                ds_var = xr.Dataset(
                    data_vars={
                        var: (("time", "station", slot_dim), matrix),
                        f"{var}_sensor_id": (("station", slot_dim), sensor_ids),
                    },
                    coords={"time": master_time_naive, "station": station_codes, slot_dim: list(range(max_slots))},
                )

            ds_var.to_zarr(str(zarr_path), mode="w" if first_write else "a", consolidated=False)
            first_write = False
            del ds_var, matrix
            gc.collect()

        try:
            import zarr
            zarr.consolidate_metadata(str(zarr_path))
        except Exception:
            pass

        return zarr_path

    def extract(
        self,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        stations: Optional[Sequence[str]] = None,
        bbox: Optional[SpatialBoundingBox] = None,
        variables: Optional[Sequence[str]] = None,
        save_parquet: bool = True,
        save_zarr: bool = False,
        zarr_name: Optional[str] = None,
        generate_metadata: bool = True,
        max_workers: Optional[int] = None,
        parquet_layout: Literal["yearly", "station", "both"] = "yearly",
        keep_station_cache: bool = False,
        show_progress: bool = True,
        resume: bool = True,
        overwrite: bool = False,
        keep_cache: Optional[bool] = None,
        dry_run: bool = False,
    ) -> Union[Dict[str, Union[pd.DataFrame, Path]], ExtractionEstimate]:
        """
        Execute batch station extraction with legacy-server-respecting concurrency.

        Mathematical docstring:
        Input:
            start_date, end_date: Temporal boundaries
            stations: Sequence of N_target station codes or None
        Output:
            results: Dict[str, Union[pd.DataFrame, Path]] mapping year (when parquet_layout='yearly')
                     or station_code (when parquet_layout='station' or save_parquet=False) -> Path or DataFrame,
                     or ExtractionEstimate when dry_run=True.

        Parameters
        ----------
        start_date : Union[str, datetime, pd.Timestamp]
            Initial observation timestamp.
        end_date : Union[str, datetime, pd.Timestamp]
            Terminal observation timestamp.
        stations : Optional[Sequence[str]]
            Specific station codes to extract.
        bbox : Optional[SpatialBoundingBox]
            Bounding box to filter stations if stations is None.
        variables : Optional[Sequence[str]]
            List of variable codes to extract.
        save_parquet : bool, default True
            Whether to write extracted observations to local Parquet files.
        save_zarr : bool, default False
            Whether to compile extracted stations into a unified multi-station Zarr store.
        zarr_name : Optional[str]
            Optional filename for compiled Zarr store.
        generate_metadata : bool, default True
            Whether to generate installation.parquet and parameters.parquet catalogs.
        max_workers : Optional[int], default None
            Maximum number of concurrent worker threads. If None, defaults to `self.max_workers`
            (default 4). Kept at a maximum of 4 to respect legacy TAHMO server infrastructure
            and avoid rate limits. Pass 1 for serial execution.
        parquet_layout : Literal["yearly", "station", "both"], default "yearly"
            Output Parquet layout when `save_parquet=True`:
            - "yearly": Stream stations into consolidated `{year}.parquet` files in O(1 station) RAM.
            - "station": Write individual `{station_code}_{start}_{end}.parquet` files.
            - "both": Write both individual station files and consolidated `{year}.parquet` files.
        keep_station_cache : bool, default False
            Deprecated alias for `keep_cache`.
        show_progress : bool, default True
            If True, displays a station-level progress bar during extraction.
        resume : bool, default True
            If True, resumes from manifest-backed cache partitions when available.
        overwrite : bool, default False
            If True, ignores cached partitions and re-fetches all stations.
        keep_cache : Optional[bool], default None
            Whether to retain cache partitions after successful extraction. If None,
            falls back to `keep_station_cache`.
        dry_run : bool, default False
            If True, returns an ExtractionEstimate without making any remote API calls.

        Returns
        -------
        Union[Dict[str, Union[pd.DataFrame, Path]], ExtractionEstimate]
            Dictionary mapping year (or station code) to DataFrame or Parquet Path,
            or ExtractionEstimate if `dry_run=True`.
        """
        t_range = self.validate_temporal_range(start_date, end_date)
        if stations is not None:
            self.validate_stations(stations, strict=True)
        if variables is not None:
            self.validate_variables(variables, strict=True)
        self.validate_temporal_bounds(t_range, strict=False)

        if parquet_layout not in ("yearly", "station", "both"):
            raise ValueError(f"parquet_layout must be 'yearly', 'station', or 'both', received: {parquet_layout}")

        if dry_run:
            return self.estimate_extraction(
                start_date=start_date,
                end_date=end_date,
                bbox=bbox or AFRICA_BBOX,
                variables=variables,
                stations=stations,
            )

        effective_keep_cache = keep_cache if keep_cache is not None else keep_station_cache

        stations_df = self.get_stations(bbox=bbox or AFRICA_BBOX) if stations is None else None
        if stations is None:
            target_codes = stations_df["station_code"].tolist() if stations_df is not None else []
        else:
            target_codes = list(stations)

        if generate_metadata:
            self.generate_metadata_catalogs(stations_df=stations_df)

        effective_workers = (
            self.max_workers
            if max_workers is None
            else clamp_max_workers(max_workers, default=self.max_workers, source="tahmo")
        )
        logger.info("Extracting TAHMO observations for %d stations (max_workers=%s)...", len(target_codes), effective_workers)
        results: Dict[str, Union[pd.DataFrame, Path]] = {}

        s_dt, e_dt = parse_date_range(start_date, end_date)
        s_str = s_dt.strftime("%Y%m%d")
        e_str = e_dt.strftime("%Y%m%d")

        rate_limiter = (
            self.create_rate_limiter(
                requests_per_second=1.0 / self.request_delay,
                burst=max(1, effective_workers or 1),
            )
            if self.request_delay > 0
            else None
        )
        cache = self.create_cache(
            self.output_dir,
            request_params={
                "start_date": s_str,
                "end_date": e_str,
                "bbox": bbox or AFRICA_BBOX,
                "variables": list(variables) if variables is not None else None,
                "parquet_layout": parquet_layout,
                "stations": sorted(target_codes) if stations is not None else None,
            },
            resume=resume,
            overwrite=overwrite,
        )

        dataframes: Dict[str, pd.DataFrame] = {}
        all_station_files: List[Tuple[str, Path]] = []

        if save_parquet and parquet_layout in ("yearly", "both"):
            year_slices = self._split_date_range_by_year(s_dt, e_dt)
            for year, y_start, y_end in year_slices:
                yearly_path = self.output_dir / f"{year}.parquet"
                if resume and not overwrite and yearly_path.exists() and parquet_layout == "yearly" and not save_zarr:
                    logger.info("Yearly TAHMO Parquet already exists: %s", yearly_path)
                    results[str(year)] = yearly_path
                    continue

                legacy_cache_dir = self.output_dir / ".station_cache" / str(year)
                ys_str = y_start.strftime("%Y%m%d")
                ye_str = y_end.strftime("%Y%m%d")

                station_files: List[Tuple[str, Path]] = []
                pending_codes: List[str] = []
                cached_count = 0
                fetched_count = 0

                for code in target_codes:
                    part_key = f"{year}/{code}"
                    if cache.is_done(part_key):
                        cache.record_hit(part_key)
                        cached_count += 1
                        cached_p = cache.get_path(part_key)
                        if cached_p is not None:
                            if parquet_layout == "both":
                                st_path = self.output_dir / f"{code}_{ys_str}_{ye_str}.parquet"
                                if not st_path.exists():
                                    shutil.copy2(cached_p, st_path)
                                station_files.append((code, st_path))
                                results[f"{code}_{year}"] = st_path
                            else:
                                station_files.append((code, cached_p))
                        continue

                    legacy_path = (
                        self.output_dir / f"{code}_{ys_str}_{ye_str}.parquet"
                        if parquet_layout == "both"
                        else legacy_cache_dir / f"{code}.parquet"
                    )
                    if resume and not overwrite and legacy_path.exists():
                        cached_count += 1
                        station_files.append((code, legacy_path))
                        if parquet_layout == "both":
                            results[f"{code}_{year}"] = legacy_path
                        continue

                    pending_codes.append(code)

                def _fetch_station_for_year(code: str) -> Tuple[str, Optional[Path]]:
                    part_key = f"{year}/{code}"
                    try:
                        df = retry_with_backoff(
                            lambda: self.extract_station(
                                station_code=code,
                                start_date=y_start,
                                end_date=y_end,
                                variables=variables,
                            ),
                            max_retries=2,
                            base_delay=0.0 if self.request_delay == 0.0 else 1.0,
                            retry_on=(requests.RequestException, TimeoutError, ConnectionError, OSError),
                            rate_limiter=rate_limiter,
                        )
                        if df.empty:
                            logger.info("No data returned for station %s (%d).", code, year)
                            cache.mark_empty(part_key)
                            return code, None
                        semi_df = self._station_df_to_semi_wide(df, code)
                        del df
                        if semi_df.empty:
                            cache.mark_empty(part_key)
                            return code, None
                        cached_p = cache.save_parquet(part_key, semi_df)
                        if parquet_layout == "both":
                            st_path = self.output_dir / f"{code}_{ys_str}_{ye_str}.parquet"
                            semi_df.to_parquet(st_path, index=False)
                            del semi_df
                            gc.collect()
                            return code, st_path
                        del semi_df
                        gc.collect()
                        return code, cached_p
                    except Exception as e:
                        logger.error("Failed to extract station %s (%d): %s", code, year, e)
                        self.failed_stations[code] = str(e)
                        cache.mark_failed(part_key, e)
                        return code, None

                with self.progress_bar(
                    total=len(target_codes),
                    initial=cached_count,
                    unit="station",
                    desc=f"TAHMO {year}",
                    show_progress=show_progress,
                ) as pbar:
                    if effective_workers and effective_workers > 1 and len(pending_codes) > 1:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
                            futures = {executor.submit(_fetch_station_for_year, c): c for c in pending_codes}
                            for fut in concurrent.futures.as_completed(futures):
                                code, p = fut.result()
                                fetched_count += 1
                                if p is not None:
                                    station_files.append((code, p))
                                    if parquet_layout == "both":
                                        results[f"{code}_{year}"] = p
                                pbar.set_postfix(cached=cached_count, fetched=fetched_count)
                                pbar.update(1)
                    else:
                        for code in pending_codes:
                            code, p = _fetch_station_for_year(code)
                            fetched_count += 1
                            if p is not None:
                                station_files.append((code, p))
                                if parquet_layout == "both":
                                    results[f"{code}_{year}"] = p
                            pbar.set_postfix(cached=cached_count, fetched=fetched_count)
                            pbar.update(1)

                station_files.sort(key=lambda item: item[0])
                all_station_files.extend(station_files)
                written_year = self._write_yearly_parquet_streaming(station_files, yearly_path)
                if written_year is not None:
                    results[str(year)] = written_year
                    logger.info("Compiled yearly TAHMO Parquet (%d stations): %s", len(station_files), written_year)

            if save_zarr and all_station_files:
                name = zarr_name or f"tahmo_stations_{s_str}_{e_str}.zarr"
                zarr_path = self.output_dir / name
                written_zarr = self._write_zarr_ordinal_slots(zarr_path, station_files=all_station_files)
                if written_zarr is not None:
                    results["_zarr"] = written_zarr
                    logger.info("Compiled multi-station TAHMO Zarr store (ordinal slots): %s", written_zarr)
        else:
            pending_codes = []
            cached_count = 0
            fetched_count = 0

            for code in target_codes:
                parquet_path = self.output_dir / f"{code}_{s_str}_{e_str}.parquet"
                if cache.is_done(code):
                    cache.record_hit(code)
                    cached_count += 1
                    cached_p = cache.get_path(code)
                    if cached_p is not None:
                        if save_parquet:
                            if not parquet_path.exists() or overwrite:
                                shutil.copy2(cached_p, parquet_path)
                            results[code] = parquet_path
                            if save_zarr:
                                dataframes[code] = cache.load_parquet(code)
                        else:
                            df_cached = cache.load_parquet(code)
                            results[code] = df_cached
                            if save_zarr:
                                dataframes[code] = df_cached
                    continue

                if save_parquet and resume and not overwrite and parquet_path.exists() and not save_zarr:
                    logger.info("Station %s already extracted: %s", code, parquet_path)
                    cached_count += 1
                    results[code] = parquet_path
                    continue

                pending_codes.append(code)

            def _fetch_single_station(code: str) -> Tuple[str, Optional[pd.DataFrame], Optional[Path]]:
                parquet_path = self.output_dir / f"{code}_{s_str}_{e_str}.parquet"
                try:
                    df = retry_with_backoff(
                        lambda: self.extract_station(
                            station_code=code,
                            start_date=start_date,
                            end_date=end_date,
                            variables=variables,
                        ),
                        max_retries=2,
                        base_delay=0.0 if self.request_delay == 0.0 else 1.0,
                        retry_on=(requests.RequestException, TimeoutError, ConnectionError, OSError),
                        rate_limiter=rate_limiter,
                    )
                    if df.empty:
                        logger.info("No data returned for station %s.", code)
                        cache.mark_empty(code)
                        return code, None, None

                    cache.save_parquet(code, df.reset_index())
                    saved_p = None
                    if save_parquet:
                        df.reset_index().to_parquet(parquet_path, index=False)
                        saved_p = parquet_path
                        if not save_zarr:
                            del df
                            gc.collect()
                            return code, None, saved_p
                    return code, df, saved_p
                except Exception as e:
                    logger.error("Failed to extract station %s: %s", code, e)
                    self.failed_stations[code] = str(e)
                    cache.mark_failed(code, e)
                    return code, None, None

            with self.progress_bar(
                total=len(target_codes),
                initial=cached_count,
                unit="station",
                show_progress=show_progress,
            ) as pbar:
                if effective_workers and effective_workers > 1 and len(pending_codes) > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
                        futures = {executor.submit(_fetch_single_station, c): c for c in pending_codes}
                        for fut in concurrent.futures.as_completed(futures):
                            code, df, p = fut.result()
                            fetched_count += 1
                            if save_zarr and df is not None:
                                dataframes[code] = df
                            if p is not None:
                                results[code] = p
                            elif df is not None and not save_parquet:
                                results[code] = df
                            pbar.set_postfix(cached=cached_count, fetched=fetched_count)
                            pbar.update(1)
                else:
                    for code in pending_codes:
                        code, df, p = _fetch_single_station(code)
                        fetched_count += 1
                        if save_zarr and df is not None:
                            dataframes[code] = df
                        if p is not None:
                            results[code] = p
                        elif df is not None and not save_parquet:
                            results[code] = df
                        pbar.set_postfix(cached=cached_count, fetched=fetched_count)
                        pbar.update(1)

        if save_zarr and dataframes:
            name = zarr_name or f"tahmo_stations_{s_str}_{e_str}.zarr"
            zarr_path = self.output_dir / name
            written_zarr = self._write_zarr_ordinal_slots(zarr_path, dataframes=dataframes)
            if written_zarr is not None:
                results["_zarr"] = written_zarr
                logger.info("Compiled multi-station TAHMO Zarr store (ordinal slots): %s", written_zarr)

        if not effective_keep_cache:
            cache.cleanup(keep_on_failure=True)

        return results
