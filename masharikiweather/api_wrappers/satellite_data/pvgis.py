"""
EU JRC PVGIS Solar Radiation & Photovoltaic Simulation API Wrapper
==================================================================
Queries the Photovoltaic Geographical Information System (PVGIS) API
developed by the European Commission Joint Research Centre (JRC).

Key Features:
-------------
- ZERO-KEY PUBLIC API: Free public REST service provided by the European Commission.
- OPEN DATA LICENSE: European Commission Reuse Notice (free for academic, commercial,
  and open-source applications with attribution).
- HIGH-RESOLUTION CM SAF SATELLITE DATA:
  * 'PVGIS-SARAH3' (default): EUMETSAT CM SAF satellite solar irradiance (2005–2023, Europe & Africa).
  * 'PVGIS-SARAH2': CM SAF satellite generation 2 (2005–2020).
  * 'PVGIS-ERA5': ECMWF ERA5 global reanalysis (2005–2023, worldwide coverage).
- MULTI-TOOL CAPABILITIES:
  * 'seriescalc': Hourly time series of radiation components (GHI, DNI, DHI), weather, and PV power.
  * 'PVcalc': Grid-connected PV performance simulation (annual Ey, monthly Em, daily Ed, system losses).
  * 'tmy': Typical Meteorological Year (TMY) synthesis (8,760 representative hourly hours).
  * 'MRcalc': Monthly radiation averages across years.
  * 'printhorizon': 360-degree topographical shading horizon profile.
- PV SIMULATION ENGINE:
  * Automatic calculation of optimal tilt ('optimalinclination') and optimal azimuth ('optimalangles').
  * Simulation of PV power generation (Watts) based on system capacity, module technology, and mounting.
- PERSISTENT PARQUET CACHING:
  * Automatic local disk caching in Parquet format to prevent redundant network queries.

References:
-----------
- Official Portal: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en
- PVGIS API Docs: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en
"""

from __future__ import annotations

import json
import logging
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

logger = logging.getLogger("MasharikiWeather.PVGIS")

# Radiation databases available in PVGIS v5.3
PVGIS_DATABASES: Dict[str, Dict[str, Any]] = {
    "PVGIS-SARAH3": {
        "description": "EUMETSAT CM SAF SARAH-3 Satellite Irradiance (Default for Europe & Africa)",
        "coverage": "Europe, Africa, parts of South America & Middle East",
        "spatial_res": "0.05° (~5 km)",
        "years": (2005, 2023),
        "source": "EUMETSAT CM SAF / Meteosat",
    },
    "PVGIS-SARAH2": {
        "description": "EUMETSAT CM SAF SARAH-2 Satellite Irradiance",
        "coverage": "Europe, Africa, parts of South America",
        "spatial_res": "0.05° (~5 km)",
        "years": (2005, 2020),
        "source": "EUMETSAT CM SAF",
    },
    "PVGIS-ERA5": {
        "description": "ECMWF ERA5 Atmospheric Reanalysis (Global Coverage)",
        "coverage": "Global (Worldwide)",
        "spatial_res": "0.28° (~31 km)",
        "years": (2005, 2023),
        "source": "ECMWF",
    },
    "PVGIS-NSRDB": {
        "description": "NREL National Solar Radiation Database (Americas)",
        "coverage": "North and Central America",
        "spatial_res": "4 km",
        "years": (2005, 2020),
        "source": "NREL",
    },
}

# Parameter definitions, units, and friendly aliases
PVGIS_CATALOG: Dict[str, Dict[str, Any]] = {
    # --- Solar Irradiance ---
    "G(i)": {
        "aliases": ["ghi", "global_horizontal", "solar_radiation", "surface_solar", "irradiance"],
        "description": "Global Irradiance on Inclined (or Horizontal) Plane",
        "units": "W/m^2",
    },
    "Gb(i)": {
        "aliases": ["dni_horizontal", "beam", "direct", "beam_irradiance"],
        "description": "Direct Beam Irradiance on Inclined (or Horizontal) Plane",
        "units": "W/m^2",
    },
    "Gd(i)": {
        "aliases": ["dhi", "diffuse", "diffuse_irradiance"],
        "description": "Diffuse Solar Irradiance on Inclined (or Horizontal) Plane",
        "units": "W/m^2",
    },
    "Gr(i)": {
        "aliases": ["refl", "reflected", "ground_reflected"],
        "description": "Ground Reflected Solar Irradiance on Plane",
        "units": "W/m^2",
    },
    "H_sun": {
        "aliases": ["sun_elevation", "solar_elevation", "sun_height", "elevation"],
        "description": "Sun Height / Solar Elevation Angle above Horizon",
        "units": "degrees",
    },

    # --- Photovoltaic Simulation ---
    "P": {
        "aliases": ["p", "power", "pv_power", "pv_output", "electric_power"],
        "description": "Simulated PV System Electric Power Generation",
        "units": "W",
    },

    # --- Surface Meteorology ---
    "T2m": {
        "aliases": ["t2m", "temp", "temperature"],
        "description": "Air Temperature at 2 Meters",
        "units": "°C",
    },
    "WS10m": {
        "aliases": ["ws10m", "wind", "wind_speed"],
        "description": "Wind Speed at 10 Meters",
        "units": "m/s",
    },
    "Int": {
        "aliases": ["int", "reconstruction_index"],
        "description": "Data Processing & Reconstruction Index (0: normal, 1: missing, 2: reconstructed)",
        "units": "integer code",
    },
}

# Reverse mapping: alias -> official PVGIS parameter code
PVGIS_LEXICON: Dict[str, str] = {}
for code, meta in PVGIS_CATALOG.items():
    PVGIS_LEXICON[code.lower()] = code
    for alias in meta["aliases"]:
        PVGIS_LEXICON[alias.lower()] = code


def list_databases() -> List[str]:
    """Return all supported radiation databases in PVGIS."""
    return list(PVGIS_DATABASES.keys())


def search_parameters(query: Optional[str] = None) -> pd.DataFrame:
    """
    Search the PVGIS parameter catalog by keyword or view all available variables.

    Parameters
    ----------
    query : str, optional
        Keyword to match across parameter codes, aliases, descriptions, or units.

    Returns
    -------
    pd.DataFrame
        Table with parameter code, primary alias, description, and units.
    """
    rows = []
    q = (query or "").lower().strip()
    for code, meta in PVGIS_CATALOG.items():
        aliases_str = ", ".join(meta["aliases"])
        desc = meta["description"]
        units = meta["units"]
        search_text = f"{code} {aliases_str} {desc} {units}".lower()

        if not q or q in search_text:
            rows.append({
                "code": code,
                "primary_alias": meta["aliases"][0] if meta["aliases"] else code.lower(),
                "all_aliases": aliases_str,
                "description": desc,
                "units": units,
            })
    return pd.DataFrame(rows)


class PVGIS(_BaseDatasetWrapper):
    """
    Comprehensive client for the European Commission Joint Research Centre (JRC)
    Photovoltaic Geographical Information System (PVGIS) API.

    Supports:
    - Hourly solar radiation and surface weather time series ('seriescalc')
    - Grid-connected PV electricity simulation with optimal angle discovery ('PVcalc')
    - Typical Meteorological Year generation ('tmy')
    - Historical monthly solar radiation tables ('MRcalc')
    - Horizon topographical shading analysis ('printhorizon')

    Parameters
    ----------
    database : str, default 'PVGIS-SARAH3'
        Radiation database: 'PVGIS-SARAH3', 'PVGIS-SARAH2', or 'PVGIS-ERA5'.
    cache_dir : str, default './mw_cache'
        Local directory for persistent Parquet caching.
    rate_limit_delay : float, default 0.3
        Pause between HTTP requests to respect JRC rate limits (max 30 req/sec).
    max_workers : int, default 4
        Number of concurrent worker threads for multi-location / multi-year queries.
    """

    API_BASE = "https://re.jrc.ec.europa.eu/api/v5_3"
    FALLBACK_API_BASE = "https://re.jrc.ec.europa.eu/api/v5_2"

    def __init__(
        self,
        database: str = "PVGIS-SARAH3",
        cache_dir: str = DEFAULT_CACHE_DIR,
        rate_limit_delay: float = 0.3,
        max_workers: int = 4,
    ):
        super().__init__(
            cache_dir=cache_dir,
            rate_limit_delay=rate_limit_delay,
            max_workers=max_workers,
        )
        self.database = database

    def _call_pvgis_endpoint(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to PVGIS API with smart v5.3 / v5.2 version routing."""
        query_str = urllib.parse.urlencode(params)
        raddb = str(params.get("raddatabase", "")).upper()

        # PVGIS-SARAH2 is hosted on v5_2; PVGIS-SARAH3 and PVGIS-ERA5 are on v5_3
        if "SARAH2" in raddb:
            endpoints = [
                f"{self.FALLBACK_API_BASE}/{tool_name}?{query_str}",
                f"{self.API_BASE}/{tool_name}?{query_str}",
            ]
        else:
            endpoints = [
                f"{self.API_BASE}/{tool_name}?{query_str}",
                f"{self.FALLBACK_API_BASE}/{tool_name}?{query_str}",
            ]

        last_error = None
        for url in endpoints:
            try:
                time.sleep(self.rate_limit_delay)
                raw_bytes = _safe_request(url, timeout=35, max_retries=2)
                data = json.loads(raw_bytes.decode("utf-8"))
                if "outputs" in data or "inputs" in data:
                    return data
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(f"PVGIS endpoint '{tool_name}' failed for params {params}: {last_error}")

    def fetch_point(
        self,
        lat: float,
        lon: float,
        start_year: int,
        end_year: int,
        variables: Optional[List[str]] = None,
        database: Optional[str] = None,
        pvcalculation: bool = False,
        peakpower: float = 1.0,
        loss: float = 14.0,
        mountingplace: str = "free",
        angle: Optional[float] = None,
        aspect: Optional[float] = None,
        optimalinclination: bool = False,
        optimalangles: bool = False,
        components: bool = True,
        clearsky: bool = False,
        location_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch hourly PVGIS observations for a specific coordinate pair.
        Stitches multi-year series and caches results per year in Parquet format.
        """
        loc_label = location_name or f"{lat:.3f}_{lon:.3f}"
        db = database or self.database
        vars_requested = variables or (
            ["ghi", "direct", "diffuse", "t2m", "ws10m", "power"]
            if pvcalculation
            else ["ghi", "direct", "diffuse", "t2m", "ws10m"]
        )

        records: List[Dict[str, Any]] = []

        for yr in range(start_year, end_year + 1):
            cache_key = {
                "source": "pvgis",
                "tool": "seriescalc",
                "db": db,
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "year": yr,
                "pvcalc": int(pvcalculation),
                "peakpower": peakpower if pvcalculation else None,
                "loss": loss if pvcalculation else None,
                "angle": angle,
                "aspect": aspect,
                "opt_inc": int(optimalinclination),
                "opt_ang": int(optimalangles),
                "clearsky": int(clearsky),
            }
            cache_file = self._get_cache_path("pvgis_series", cache_key)
            if self._is_cache_valid(cache_file):
                try:
                    cached_df = pd.read_parquet(cache_file)
                    records.extend(cached_df.to_dict(orient="records"))
                    continue
                except Exception:
                    pass

            params: Dict[str, Any] = {
                "lat": f"{lat:.4f}",
                "lon": f"{lon:.4f}",
                "startyear": yr,
                "endyear": yr,
                "pvcalculation": 1 if pvcalculation else 0,
                "components": 1 if components else 0,
                "clearsky": 1 if clearsky else 0,
                "outputformat": "json",
            }
            if db:
                params["raddatabase"] = db

            if pvcalculation:
                params["peakpower"] = peakpower
                params["loss"] = loss
                params["mountingplace"] = mountingplace
                if optimalangles:
                    params["optimalangles"] = 1
                elif optimalinclination:
                    params["optimalinclination"] = 1
                else:
                    if angle is not None:
                        params["angle"] = angle
                    if aspect is not None:
                        params["aspect"] = aspect

            yr_records: List[Dict[str, Any]] = []
            try:
                data = self._call_pvgis_endpoint("seriescalc", params)
            except Exception as e:
                logger.warning(f"PVGIS seriescalc failed for ({lat}, {lon}) in year {yr}: {e}")
                continue

            hourly_list = data.get("outputs", {}).get("hourly", [])
            for entry in hourly_list:
                t_str = entry.get("time", "")
                if len(t_str) < 12:
                    continue

                # Parse timestamp: "20200101:1205" -> "2020-01-01 12:05:00 UTC"
                try:
                    iso = f"{t_str[:4]}-{t_str[4:6]}-{t_str[6:8]} {t_str[9:11]}:{t_str[11:13]}:00"
                    ts = pd.Timestamp(iso, tz="UTC")
                except Exception:
                    continue

                beam = float(entry.get("Gb(i)", 0.0) or 0.0)
                diff = float(entry.get("Gd(i)", 0.0) or 0.0)
                refl = float(entry.get("Gr(i)", 0.0) or 0.0)
                ghi_val = entry.get("G(i)")
                ghi = float(ghi_val) if ghi_val is not None else (beam + diff + refl)

                raw_field_map: Dict[str, float] = {
                    "ghi": max(ghi, 0.0),
                    "global_horizontal": max(ghi, 0.0),
                    "dni_horizontal": max(beam, 0.0),
                    "direct": max(beam, 0.0),
                    "beam": max(beam, 0.0),
                    "dhi": max(diff, 0.0),
                    "diffuse": max(diff, 0.0),
                    "refl": max(refl, 0.0),
                    "reflected": max(refl, 0.0),
                    "sun_elevation": float(entry.get("H_sun", 0.0) or 0.0),
                    "h_sun": float(entry.get("H_sun", 0.0) or 0.0),
                    "t2m": float(entry.get("T2m", 0.0) or 0.0),
                    "temp": float(entry.get("T2m", 0.0) or 0.0),
                    "temperature": float(entry.get("T2m", 0.0) or 0.0),
                    "ws10m": float(entry.get("WS10m", 0.0) or 0.0),
                    "wind": float(entry.get("WS10m", 0.0) or 0.0),
                    "wind_speed": float(entry.get("WS10m", 0.0) or 0.0),
                }

                if "P" in entry:
                    raw_field_map["p"] = float(entry["P"] or 0.0)
                    raw_field_map["power"] = float(entry["P"] or 0.0)
                    raw_field_map["pv_power"] = float(entry["P"] or 0.0)

                for v in vars_requested:
                    canon = v.lower().strip()
                    if canon in raw_field_map:
                        yr_records.append({
                            "time": ts,
                            "location": loc_label,
                            "lat": lat,
                            "lon": lon,
                            "variable": canon,
                            "value": raw_field_map[canon],
                        })

            if yr_records:
                try:
                    pd.DataFrame(yr_records).to_parquet(cache_file, index=False)
                except Exception:
                    pass
            records.extend(yr_records)

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
        database: Optional[str] = None,
        pvcalculation: bool = False,
        peakpower: float = 1.0,
        loss: float = 14.0,
        mountingplace: str = "free",
        angle: Optional[float] = None,
        aspect: Optional[float] = None,
        optimalinclination: bool = False,
        optimalangles: bool = False,
        components: bool = True,
        clearsky: bool = False,
        format: str = "wide",
    ) -> pd.DataFrame:
        """
        Query PVGIS solar radiation and weather time series across one or more locations.
        Defaults to Kenya/Africa coverage if no coordinates are supplied.

        Parameters
        ----------
        locations : dict, tuple, list, or str, optional
            Coordinates or location descriptor.
        lat, lon : float, optional
            Direct latitude and longitude coordinates.
        region : str, optional
            Predefined region name ('kenya', 'east_africa', etc.).
        variables : list of str, optional
            Variables to retrieve (e.g. ['ghi', 'direct', 'diffuse', 'temp', 'wind', 'pv_power']).
        start_date, end_date : str or datetime, optional
            Date range (e.g. '2020-01-01' to '2020-01-31').
            Valid years for SARAH3: 2005 to 2023.
        database : str, optional
            Radiation database. Defaults to 'PVGIS-SARAH3'.
        pvcalculation : bool, default False
            If True, calculates simulated hourly PV electricity power (Watts).
        peakpower : float, default 1.0
            Installed PV peak power in kWp.
        loss : float, default 14.0
            Total PV system loss in percentage (cables, inverter, dirt).
        mountingplace : str, default 'free'
            'free' (free-standing rack) or 'building' (rooftop/integrated).
        angle, aspect : float, optional
            Tilt angle (0=horizontal) and azimuth orientation (-180 to 180).
        optimalinclination : bool, default False
            If True, calculates optimal tilt angle automatically.
        optimalangles : bool, default False
            If True, calculates both optimal tilt and azimuth angles automatically.
        format : str, default 'wide'
            'wide' (datetime index with columns) or 'long' (tidy dataframe).
        """
        locs = _normalize_locations(locations=locations, lat=lat, lon=lon, region=region)
        s_date = _parse_date(start_date, "2020-01-01")
        e_date = _parse_date(end_date, "2020-01-07")
        start_yr = int(s_date[:4])
        end_yr = int(e_date[:4])

        all_records: List[Dict[str, Any]] = []

        if len(locs) == 1:
            loc = locs[0]
            all_records = self.fetch_point(
                lat=loc["lat"],
                lon=loc["lon"],
                start_year=start_yr,
                end_year=end_yr,
                variables=variables,
                database=database,
                pvcalculation=pvcalculation,
                peakpower=peakpower,
                loss=loss,
                mountingplace=mountingplace,
                angle=angle,
                aspect=aspect,
                optimalinclination=optimalinclination,
                optimalangles=optimalangles,
                components=components,
                clearsky=clearsky,
                location_name=loc["name"],
            )
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self.fetch_point,
                        lat=loc["lat"],
                        lon=loc["lon"],
                        start_year=start_yr,
                        end_year=end_yr,
                        variables=variables,
                        database=database,
                        pvcalculation=pvcalculation,
                        peakpower=peakpower,
                        loss=loss,
                        mountingplace=mountingplace,
                        angle=angle,
                        aspect=aspect,
                        optimalinclination=optimalinclination,
                        optimalangles=optimalangles,
                        components=components,
                        clearsky=clearsky,
                        location_name=loc["name"],
                    )
                    for loc in locs
                ]
                for f in as_completed(futures):
                    try:
                        all_records.extend(f.result())
                    except Exception as e:
                        logger.warning(f"PVGIS point query failed: {e}")

        # Filter strictly within the requested temporal window
        if all_records:
            s_ts = pd.Timestamp(s_date, tz="UTC")
            e_ts = pd.Timestamp(e_date, tz="UTC") + pd.Timedelta(days=1)
            all_records = [r for r in all_records if s_ts <= r["time"] < e_ts]

        return self._reshape_output(all_records, format=format)

    def pvcalc(
        self,
        lat: float,
        lon: float,
        peakpower: float = 1.0,
        loss: float = 14.0,
        mountingplace: str = "free",
        pvtechchoice: str = "crystSi",
        angle: Optional[float] = None,
        aspect: Optional[float] = None,
        optimalinclination: bool = True,
        optimalangles: bool = True,
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Simulate the performance of a grid-connected PV system using the PVGIS 'PVcalc' tool.

        Returns
        -------
        dict with:
        - 'totals': Annual energy production Ey (kWh), Daily Ed, Irradiation Hy (kWh/m^2), Losses.
        - 'monthly': DataFrame of monthly energy yields (Em) and in-plane irradiation (Hm).
        - 'optimal_angles': Dict of optimal tilt slope and azimuth.
        - 'inputs': Elevation, system configuration, and metadata.
        """
        db = database or self.database
        params: Dict[str, Any] = {
            "lat": f"{lat:.4f}",
            "lon": f"{lon:.4f}",
            "peakpower": peakpower,
            "loss": loss,
            "mountingplace": mountingplace,
            "pvtechchoice": pvtechchoice,
            "outputformat": "json",
        }
        if db:
            params["raddatabase"] = db

        if optimalangles:
            params["optimalangles"] = 1
        elif optimalinclination:
            params["optimalinclination"] = 1
        else:
            if angle is not None:
                params["angle"] = angle
            if aspect is not None:
                params["aspect"] = aspect

        cache_key = {
            "source": "pvgis",
            "tool": "PVcalc",
            "db": db,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "params": params,
        }
        cache_file = self._get_cache_path("pvgis_pvcalc", cache_key)
        data = None
        if self._is_cache_valid(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass

        if not data:
            data = self._call_pvgis_endpoint("PVcalc", params)
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
            except Exception:
                pass

        outputs = data.get("outputs", {})
        monthly_list = outputs.get("monthly", {}).get("fixed", [])
        monthly_df = pd.DataFrame(monthly_list) if monthly_list else pd.DataFrame()
        totals = outputs.get("totals", {}).get("fixed", {})
        optimal_info = data.get("inputs", {}).get("mounting_system", {}).get("fixed", {})

        return {
            "totals": totals,
            "monthly": monthly_df,
            "optimal_angles": optimal_info,
            "inputs": data.get("inputs", {}),
            "elevation": data.get("inputs", {}).get("location", {}).get("elevation"),
        }

    def tmy(
        self,
        lat: float,
        lon: float,
        database: Optional[str] = None,
        usehorizon: bool = True,
    ) -> pd.DataFrame:
        """
        Download a Typical Meteorological Year (TMY) dataset for a location.
        A TMY consists of 8,760 hourly records selected from typical historical months.

        Returns
        -------
        pd.DataFrame
            Hourly dataframe (8,760 rows) with T2m, RH, G(h) GHI, Gb(n) DNI, Gd(h) DHI, IR(h), WS10m, WD10m, SP.
        """
        db = database or self.database
        params: Dict[str, Any] = {
            "lat": f"{lat:.4f}",
            "lon": f"{lon:.4f}",
            "usehorizon": 1 if usehorizon else 0,
            "outputformat": "json",
        }
        if db:
            params["raddatabase"] = db

        cache_key = {
            "source": "pvgis",
            "tool": "tmy",
            "db": db,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
        }
        cache_file = self._get_cache_path("pvgis_tmy", cache_key)
        if self._is_cache_valid(cache_file):
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

        data = self._call_pvgis_endpoint("tmy", params)
        tmy_hourly = data.get("outputs", {}).get("tmy_hourly", [])
        if not tmy_hourly:
            return pd.DataFrame()

        records = []
        for row in tmy_hourly:
            t_str = row.get("time(UTC)", "")
            try:
                iso = f"{t_str[:4]}-{t_str[4:6]}-{t_str[6:8]} {t_str[9:11]}:{t_str[11:13]}:00"
                dt = pd.Timestamp(iso, tz="UTC")
            except Exception:
                continue

            records.append({
                "time": dt,
                "t2m": float(row.get("T2m", 0.0) or 0.0),
                "rh": float(row.get("RH", 0.0) or 0.0),
                "ghi": float(row.get("G(h)", 0.0) or 0.0),
                "dni": float(row.get("Gb(n)", 0.0) or 0.0),
                "dhi": float(row.get("Gd(h)", 0.0) or 0.0),
                "infra_red": float(row.get("IR(h)", 0.0) or 0.0),
                "wind_speed": float(row.get("WS10m", 0.0) or 0.0),
                "wind_dir": float(row.get("WD10m", 0.0) or 0.0),
                "surface_pressure": float(row.get("SP", 0.0) or 0.0),
            })

        df = pd.DataFrame(records).set_index("time")
        if not df.empty:
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass

        return df

    def monthly_radiation(
        self,
        lat: float,
        lon: float,
        start_year: int = 2015,
        end_year: int = 2020,
        database: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query monthly solar irradiation statistics across multiple historical years ('MRcalc').

        Returns
        -------
        pd.DataFrame
            Table with columns: year, month, and H(h)_m (monthly global horizontal irradiation in kWh/m^2).
        """
        db = database or self.database
        params: Dict[str, Any] = {
            "lat": f"{lat:.4f}",
            "lon": f"{lon:.4f}",
            "startyear": start_year,
            "endyear": end_year,
            "horirrad": 1,
            "outputformat": "json",
        }
        if db:
            params["raddatabase"] = db

        data = self._call_pvgis_endpoint("MRcalc", params)
        monthly_records = data.get("outputs", {}).get("monthly", [])
        return pd.DataFrame(monthly_records)

    def horizon(self, lat: float, lon: float) -> pd.DataFrame:
        """
        Query the 360-degree topographical shading horizon height profile ('printhorizon').

        Returns
        -------
        pd.DataFrame
            Azimuth (0 to 360 degrees) and corresponding horizon height angle (degrees).
        """
        params = {
            "lat": f"{lat:.4f}",
            "lon": f"{lon:.4f}",
            "outputformat": "json",
        }
        data = self._call_pvgis_endpoint("printhorizon", params)
        horizon_list = data.get("outputs", {}).get("horizon_profile", [])
        return pd.DataFrame(horizon_list)


def query_pvgis(
    locations: Optional[Union[Dict[str, float], Tuple[float, float], List[Any], str]] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    region: Optional[str] = None,
    variables: Optional[List[str]] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    database: str = "PVGIS-SARAH3",
    pvcalculation: bool = False,
    peakpower: float = 1.0,
    loss: float = 14.0,
    optimalinclination: bool = False,
    optimalangles: bool = False,
    format: str = "wide",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Convenience function to query hourly PVGIS solar radiation and weather telemetry in 1-2 lines.

    Parameters
    ----------
    locations : dict, tuple, list, or str, optional
        Target coordinates. Defaults to Kenya/Africa centroid.
    lat, lon : float, optional
        Target latitude and longitude.
    region : str, optional
        Predefined region ('kenya', 'east_africa', etc.).
    variables : list of str, optional
        Variables to retrieve (e.g. ['ghi', 'direct', 'diffuse', 'temp', 'wind', 'pv_power']).
    start_date, end_date : str or datetime, optional
        Date range (e.g. '2020-01-01' to '2020-01-07'). Note: SARAH3 covers 2005 to 2023.
    database : str, default 'PVGIS-SARAH3'
        'PVGIS-SARAH3' (default), 'PVGIS-SARAH2', or 'PVGIS-ERA5'.
    pvcalculation : bool, default False
        If True, calculates simulated hourly PV electricity power (Watts).
    format : str, default 'wide'
        'wide' (datetime index with columns) or 'long' (tidy dataframe).
    """
    client = PVGIS(database=database, cache_dir=cache_dir)
    return client.query(
        locations=locations,
        lat=lat,
        lon=lon,
        region=region,
        variables=variables,
        start_date=start_date,
        end_date=end_date,
        database=database,
        pvcalculation=pvcalculation,
        peakpower=peakpower,
        loss=loss,
        optimalinclination=optimalinclination,
        optimalangles=optimalangles,
        format=format,
    )


def simulate_pvgis_pv(
    lat: float = -0.398,
    lon: float = 36.958,
    peakpower: float = 1.0,
    loss: float = 14.0,
    optimalangles: bool = True,
    database: str = "PVGIS-SARAH3",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Dict[str, Any]:
    """
    Simulate annual and monthly photovoltaic electricity generation for a grid-connected PV system.

    Returns
    -------
    dict with:
    - 'totals': Annual Ey (kWh), Daily Ed (kWh), In-plane irradiation Hy (kWh/m^2), Losses.
    - 'monthly': Monthly electricity production Em and irradiation Hm.
    - 'optimal_angles': Optimal tilt and azimuth orientation.
    """
    client = PVGIS(database=database, cache_dir=cache_dir)
    return client.pvcalc(
        lat=lat,
        lon=lon,
        peakpower=peakpower,
        loss=loss,
        optimalangles=optimalangles,
        database=database,
    )


def query_pvgis_tmy(
    lat: float = -0.398,
    lon: float = 36.958,
    database: str = "PVGIS-SARAH3",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Download a Typical Meteorological Year (TMY) 8,760-hour dataset."""
    client = PVGIS(database=database, cache_dir=cache_dir)
    return client.tmy(lat=lat, lon=lon, database=database)
