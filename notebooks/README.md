# MasharikiWeather Demonstration Notebooks Catalog
======================================================

Interactive demonstration notebooks showcasing data acquisition, quality control, analysis, and visualization across **weather station observations**, **satellite missions**, and **atmospheric reanalyses** over East Africa.

---

## 1. Quick Navigation Table

| Category | Notebook | Target Data Provider | Key Parameters & Variables | Case Study Focus |
|:---|:---|:---|:---|:---|
| **Station** | [Demonstrate_GHCN_Pipeline.ipynb](Demonstrate_GHCN_Pipeline.ipynb) | NOAA NCEI (GHCN-D & GHCN-H) | Daily/Hourly rainfall, $T_{\max}, T_{\min}$, Quality Flags | East Africa station discovery & coverage |
| **Satellite** | [Demonstrate_NASA_POWER.ipynb](Demonstrate_NASA_POWER.ipynb) | NASA Langley CERES | Hourly/Daily GHI, DNI, Diffuse, Temp, Humidity, Wind | DeKUT Nyeri & Kenya climate profile |
| **Satellite** | [Demonstrate_PVGIS.ipynb](Demonstrate_PVGIS.ipynb) | EU Joint Research Centre | Solar radiation, TMY, PV power production ($P_{\text{calc}}$) | Photovoltaic generation modeling at DeKUT |
| **Satellite** | [Demonstrate_SARAH3.ipynb](Demonstrate_SARAH3.ipynb) | EUMETSAT CM SAF | High-resolution satellite solar irradiance (GHI, DNI) | Long-term solar climatology & validation |
| **Satellite** | [Demonstrate_LANDSAF.ipynb](Demonstrate_LANDSAF.ipynb) | EUMETSAT LSA SAF | 30-min geostationary MSG SEVIRI DSSF, LST, FVC | High-cadence diurnal surface fluxes |
| **Satellite** | [Demonstrate_TROPOMI.ipynb](Demonstrate_TROPOMI.ipynb) | Copernicus Sentinel-5P | Level-2 $\text{NO}_2, \text{CO}, \text{CH}_4, \text{SO}_2$, UV Aerosol Index | Regional air pollution & greenhouse gas monitoring |
| **Reanalysis** | [Demonstrate_CAMS.ipynb](Demonstrate_CAMS.ipynb) | Copernicus Atmosphere Service | Total AOD, Dust, Organic Carbon, Sulfate, PM2.5, PM10 | Dust transport & air quality over East Africa |
| **Reanalysis** | [Demonstrate_MERRA2.ipynb](Demonstrate_MERRA2.ipynb) | NASA GMAO Reanalysis | Hourly aerosol speciations, SWGDN solar flux, 2m temp | 2D atmospheric reanalysis over Kenya |
| **Reanalysis** | [Demonstrate_Open_Meteo.ipynb](Demonstrate_Open_Meteo.ipynb) | Multi-Model NWP & Land | High-res temperature, precipitation, wind, soil metrics | Diurnal weather & agro-meteorology at DeKUT |

---

## 2. Notebook Summaries

### Weather Station Data
* **[Demonstrate_GHCN_Pipeline.ipynb](Demonstrate_GHCN_Pipeline.ipynb)**
  - Automated station discovery across Kenya, Uganda, Tanzania, Ethiopia, and Rwanda.
  - Multi-tiered ingestion from AWS S3 Open Data and NOAA NCEI.
  - Rigorous QC filtering, gap analysis, and wide-format matrix transformations.

### Satellite Missions
* **[Demonstrate_NASA_POWER.ipynb](Demonstrate_NASA_POWER.ipynb)**
  - Automated API latency discovery and active parameter probing.
  - 1-line queries for single points, bounding boxes, or country boundaries (`"kenya"`).
  - Daily & hourly solar energy potential for Mount Kenya slopes (DeKUT Nyeri).
* **[Demonstrate_PVGIS.ipynb](Demonstrate_PVGIS.ipynb)**
  - Photovoltaic system power simulation with tilt/azimuth optimization.
  - Typical Meteorological Year (TMY) synthesis across PVGIS-SARAH2, PVGIS-SARAH3, and PVGIS-ERA5.
* **[Demonstrate_SARAH3.ipynb](Demonstrate_SARAH3.ipynb)**
  - Calibrated surface solar irradiance from geostationary Meteosat satellites.
  - Global Horizontal (GHI), Direct Normal (DNI), and Diffuse Horizontal Irradiance (DHI).
* **[Demonstrate_LANDSAF.ipynb](Demonstrate_LANDSAF.ipynb)**
  - 30-minute high-cadence Land Surface Analysis from Meteosat Second Generation (MSG).
  - Downward Surface Shortwave Flux (MDSSFTD), Land Surface Temperature (MLST), and Fractional Vegetation Cover.
* **[Demonstrate_TROPOMI.ipynb](Demonstrate_TROPOMI.ipynb)**
  - Sentinel-5P TROPOMI UV-VIS-NIR-SWIR spectrometer observations.
  - Methane column mixing ratio ($X\text{CH}_4$), Tropospheric $\text{NO}_2$ column, Carbon Monoxide ($X\text{CO}$), and UV Aerosol Index (UVAI).

### Atmospheric Reanalyses
* **[Demonstrate_CAMS.ipynb](Demonstrate_CAMS.ipynb)**
  - Copernicus European atmospheric monitoring.
  - Aerosol Optical Depth (AOD at 550nm) partitioned by dust, organic matter, sea salt, and sulfate.
  - Particulate matter (PM2.5, PM10) and CAQI air quality classification.
* **[Demonstrate_MERRA2.ipynb](Demonstrate_MERRA2.ipynb)**
  - NASA Goddard Earth Observing System (GEOS-5) reanalysis.
  - Hourly total and speciation-resolved aerosol optical thickness, downward solar radiation, and boundary layer meteorology.
* **[Demonstrate_Open_Meteo.ipynb](Demonstrate_Open_Meteo.ipynb)**
  - Blended high-resolution NWP (ECMWF IFS, GFS, ICON) at 1–11km resolution.
  - Historical archive and 16-day forecast, with multi-depth soil moisture and soil temperature modeling.

---

## 3. Running Locally

Ensure your Python virtual environment is active:
```bash
# In the repository root
source .venv/bin/activate  # On Linux/macOS
# or: .\.venv\Scripts\Activate.ps1  # On Windows PowerShell

# Launch Jupyter Lab / Notebook
jupyter lab notebooks/
```
