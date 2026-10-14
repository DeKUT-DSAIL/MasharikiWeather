# MasharikiWeather API Wrappers — Comprehensive Usage Guide
=============================================================

`MasharikiWeather` provides clean, high-performance Python wrappers for ingesting **ground weather stations**, **earth observation satellites**, and **numerical reanalyses** across East Africa.

Every wrapper features:
- **Zero API Key Requirement:** Works out of the box with open-access endpoints.
- **Smart Disk Caching:** Automated local caching ensures repeat queries run instantly without redundant network calls.
- **Clean Pandas DataFrames:** Standardized UTC datetime indices, CF-compliant units, and wide-format tables.
- **Flexible Geospatial Querying:** Query by single coordinate **point**, named **region**, or **bounding box**.

---

## Quick Reference Table

| Category | Source Wrapper | Point Query | Regional Name | Bounding Box | Key Variables |
|:---|:---|:---:|:---:|:---:|:---|
| **Station** | `GHCN` (NOAA) | | | | Precipitation, $T_{\max}$, $T_{\min}$, Snow |
| **Satellite** | `NASAPower` | | | | GHI, DNI, 2m Temperature, Humidity, Wind |
| **Satellite** | `PVGIS` (EU JRC) | | | | GHI, DNI, Temperature, PV Power ($P_{\text{calc}}$), TMY |
| **Satellite** | `SARAH3` (CM SAF) | | | | Satellite Solar Irradiance (GHI, Direct, Diffuse) |
| **Satellite** | `LANDSAF` (LSA SAF) | | | | 30-min DSSF Solar Flux, LST, Vegetation Fraction |
| **Satellite** | `TROPOMI` (Sentinel-5P) | | | | Trace gases ($\text{NO}_2, \text{CO}, \text{CH}_4, \text{SO}_2$), UV Aerosol Index |
| **Reanalysis** | `CAMS` (Copernicus) | | | | AOD (550nm), Dust, Organic Carbon, PM2.5, PM10 |
| **Reanalysis** | `MERRA2` (NASA GMAO) | | | | Speciated AOD, Surface Downward Solar (SWGDN), 2m Temp |
| **Reanalysis** | `OpenMeteo` | | | | Multi-model weather, 16-day forecast, multi-depth soil |

---

## 1. Weather Station Data API Wrappers Usage

### NOAA GHCN (Global Historical Climatology Network)
The GHCN wrapper discovers African ground weather stations, retrieves quality-controlled daily and hourly climate records via AWS S3 Open Data and NOAA NCEI, and aligns them into tabular matrices.

```python
from masharikiweather import query_ghcn_daily, query_ghcn_hourly, get_stations

# 1. Query by Coordinate Point & Radius (e.g., DeKUT Nyeri campus, within 60 km)
df_point = query_ghcn_daily(
    lat=-0.398, lon=36.958, radius_km=60.0,
    variables=["prcp", "tmax", "tmin"],
    start_date="2020-01-01", end_date="2020-12-31"
)

# 2. Query by Country Name (e.g., all active stations in Kenya)
df_kenya = query_ghcn_daily(
    country="Kenya",
    variables=["prcp", "tmax", "tmin"],
    start_date="2022-01-01", end_date="2022-01-31"
)

# 3. Query by Bounding Box [min_lat, min_lon, max_lat, max_lon]
df_bbox = query_ghcn_daily(
    bbox=[-1.5, 36.5, -0.2, 37.5],
    variables=["prcp"],
    start_date="2023-01-01", end_date="2023-03-31"
)

# 4. Query by Specific Station ID (e.g., Nairobi Dagoretti KE000063740)
df_station = query_ghcn_daily(
    stations=["KE000063740"],
    start_date="2020-01-01", end_date="2021-12-31"
)
```

---

## 2. Satellite Data API Wrappers Usage

### NASA POWER (CERES Satellite Solar & Surface Meteorology)
Global hourly and daily satellite-derived solar irradiance and agro-meteorological variables.

```python
from masharikiweather import query_nasa_power

# 1. Query by Single Point (DeKUT Nyeri)
df_point = query_nasa_power(
    lat=-0.398, lon=36.958,
    variables=["ghi", "t2m", "rh2m", "wind_speed"],
    start_date="2024-01-01", end_date="2024-01-05",
    resolution="hourly"
)

# 2. Query by Regional Name (Country Preset: "kenya", "uganda", "tanzania", etc.)
df_region = query_nasa_power(
    region="kenya",
    variables=["ghi", "t2m"],
    start_date="2024-01-01", end_date="2024-01-03",
    resolution="daily"
)

# 3. Query by Bounding Box (min_lat, min_lon, max_lat, max_lon)
df_bbox = query_nasa_power(
    bbox=(-1.5, 36.5, -0.2, 37.8),
    variables=["ghi", "t2m"],
    start_date="2024-01-01", end_date="2024-01-02",
    resolution="hourly"
)
```

---

### PVGIS (EU Joint Research Centre Solar Radiation & PV Simulation)
Access calibrated satellite databases (PVGIS-SARAH2, PVGIS-SARAH3, ERA5), simulate grid-connected PV systems, or generate Typical Meteorological Years (TMY).

```python
from masharikiweather import query_pvgis, simulate_pvgis_pv, query_pvgis_tmy

# 1. Hourly Time Series Query (DeKUT Nyeri using SARAH-2 satellite radiation)
df_solar = query_pvgis(
    lat=-0.398, lon=36.958,
    variables=["ghi", "t2m", "wind_speed"],
    start_date="2020-01-01", end_date="2020-01-07",
    database="PVGIS-SARAH2"
)

# 2. PV Power Production Simulation (5.0 kWp system, 15° tilt facing South)
df_pv = simulate_pvgis_pv(
    lat=-0.398, lon=36.958,
    peakpower=5.0, loss=14.0, angle=15.0, aspect=180.0,
    start_date="2020-01-01", end_date="2020-01-05"
)

# 3. Typical Meteorological Year (TMY) synthesis for long-term project design
df_tmy = query_pvgis_tmy(lat=-0.398, lon=36.958)
```

---

### SARAH-3 (EUMETSAT CM SAF Surface Solar Irradiance)
High-precision geostationary satellite records for Global Horizontal (GHI), Direct Normal (DNI), and Diffuse Horizontal Irradiance (DHI).

```python
from masharikiweather import query_sarah3

# Query hourly satellite solar components for DeKUT Nyeri
df_sarah = query_sarah3(
    lat=-0.398, lon=36.958,
    variables=["ghi", "direct", "diffuse"],
    start_date="2020-01-01", end_date="2020-01-05",
    format="wide"
)
```

---

### LANDSAF (EUMETSAT Land Surface Analysis SAF)
Geostationary Meteosat MSG SEVIRI 30-minute Downward Surface Shortwave Flux (MDSSFTD), Land Surface Temperature (MLST), and Fractional Vegetation Cover.

```python
from masharikiweather import query_landsaf

# 1. High-Cadence 30-Minute Surface Energy Fluxes (Zero-key open access mode)
df_flux = query_landsaf(
    lat=-0.398, lon=36.958,
    variables=["dssf_tot", "dssf_dir", "dssf_diff", "lst", "fvc"],
    start_date="2020-01-01", end_date="2020-01-03",
    frequency="30min"
)

# 2. Direct LSA SAF Data Server Mode (For users with registered credentials)
df_auth = query_landsaf(
    lat=-0.398, lon=36.958,
    variables=["dssf_tot", "lst"],
    start_date="2020-01-01", end_date="2020-01-02",
    username="user@dekut.ac.ke", password="secret_password"
)
```

---

### TROPOMI (Copernicus Sentinel-5P Atmospheric Gases & Aerosols)
Level-2 spectrometer observations of greenhouse gases, combustion tracers, and absorbing dust plumes.

```python
from masharikiweather import query_tropomi

# Query multi-variable atmospheric composition for Nairobi / Mount Kenya
df_tropomi = query_tropomi(
    lat=-0.398, lon=36.958,
    variables=["no2", "co", "ch4", "so2", "aer_ai"],
    start_date="2024-01-01", end_date="2024-01-05"
)
```

---

## 3. Reanalysis API Wrappers Usage

### CAMS (Copernicus Atmosphere Monitoring Service)
Atmospheric composition, aerosol optical depth partitioned by species, and surface particulate matter.

```python
from masharikiweather import query_cams_aerosols, query_cams_forecast

# 1. Query by Regional Name (Country Preset: "kenya", "uganda", "tanzania")
df_aero_ke = query_cams_aerosols(
    region="kenya",
    variables=["aod", "dust", "pm2p5", "pm10"],
    start_date="2024-01-01", end_date="2024-01-03"
)

# 2. Query by Coordinate Point (DeKUT Nyeri)
df_point = query_cams_aerosols(
    lat=-0.398, lon=36.958,
    variables=["aod", "dust", "organic_carbon", "pm10"],
    start_date="2024-01-01", end_date="2024-01-05"
)

# 3. Query 5-Day Near-Real-Time Air Quality Forecast
df_forecast = query_cams_forecast(
    lat=-0.398, lon=36.958,
    forecast_days=5
)
```

---

### MERRA-2 (NASA GMAO 2D Atmospheric Reanalysis)
Hourly 2D reanalysis for total and speciated aerosol optical thickness, downward solar flux, and boundary layer meteorology.

```python
from masharikiweather import query_merra2

# Query speciated aerosols and downward solar radiation for DeKUT Nyeri
df_merra = query_merra2(
    lat=-0.398, lon=36.958,
    variables=["aod_total", "aod_dust", "aod_black_carbon", "swgdn", "t2m"],
    start_date="2020-01-01", end_date="2020-01-03"
)
```

---

### Open-Meteo (High-Resolution Multi-Model NWP & Land Physics)
Blended high-resolution numerical weather prediction (ECMWF IFS, GFS, ICON) at 1–11km resolution, historical archive, and agro-meteorological soil layers.

```python
from masharikiweather import query_open_meteo, query_open_meteo_forecast, query_open_meteo_soil

# 1. Historical High-Resolution Hourly Weather
df_hist = query_open_meteo(
    lat=-0.398, lon=36.958,
    variables=["temp", "precip", "rh", "wind_speed", "solar_radiation"],
    start_date="2024-01-01", end_date="2024-01-05"
)

# 2. 16-Day Operational Forecast
df_fc = query_open_meteo_forecast(
    lat=-0.398, lon=36.958,
    forecast_days=7
)

# 3. Multi-Depth Soil Moisture & Temperature (Agro-Climatology)
df_soil = query_open_meteo_soil(
    lat=-0.398, lon=36.958,
    start_date="2024-01-01", end_date="2024-01-05"
)
```

---

## 4. Universal Unified Query Dispatcher

You can also use the top-level `query_dataset(...)` dispatcher to query any provider through a single entry point:

```python
from masharikiweather import query_dataset

# Query NASA POWER
df_power = query_dataset(
    source="nasa_power", lat=-0.398, lon=36.958,
    variables=["ghi"], start_date="2024-01-01", end_date="2024-01-02"
)

# Query PVGIS
df_pvgis = query_dataset(
    source="pvgis", lat=-0.398, lon=36.958,
    variables=["ghi"], start_date="2020-01-01", end_date="2020-01-02"
)

# Query SARAH-3
df_sarah = query_dataset(
    source="sarah3", lat=-0.398, lon=36.958,
    variables=["ghi"], start_date="2020-01-01", end_date="2020-01-02"
)

# Query Open-Meteo
df_meteo = query_dataset(
    source="open_meteo", lat=-0.398, lon=36.958,
    variables=["temp", "precip"], start_date="2024-01-01", end_date="2024-01-02"
)
```

---

## 5. Directory Layout & Organization

The wrappers are cleanly organized inside the package as follows:

```
masharikiweather/
├── __init__.py                                 # Unified package exports
├── api_wrappers.py                             # High-level API wrappers interface
├── api-wrappers/
│   ├── USAGE.md                                # This documentation file
│   ├── weather-station-data/                   # Ground weather stations
│   │   ├── __init__.py
│   │   └── ghcn_apiwrapper.py                  # NOAA GHCN-D & GHCN-H pipeline
│   ├── satellite-data/                         # Earth observation satellites
│   │   ├── __init__.py
│   │   ├── nasa_power.py                       # NASA POWER (CERES SYN1deg)
│   │   ├── pvgis.py                            # EU JRC PVGIS (SARAH-2/3, ERA5)
│   │   ├── sarah3.py                           # EUMETSAT CM SAF SARAH-3
│   │   ├── landsaf.py                          # EUMETSAT LSA SAF MSG SEVIRI
│   │   └── tropomi.py                          # Copernicus Sentinel-5P Level-2
│   └── reanalysis-data/                        # Atmospheric reanalyses & NWP
│       ├── __init__.py
│       ├── cams.py                             # Copernicus Atmosphere (CAMS)
│       ├── merra2.py                           # NASA GMAO MERRA-2
│       └── open_meteo.py                       # Multi-model NWP & Soil physics
└── wrappers/                                   # Core backward-compatible engine
    ├── base.py
    └── ...
```