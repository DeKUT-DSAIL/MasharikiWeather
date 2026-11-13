# MasharikiWeather

## Overview

**MasharikiWeather** is an open initiative to create an **ML-ready, framework-agnostic weather dataset for Africa**.  
It draws inspiration from the [**PeakWeather**](https://huggingface.co/datasets/MeteoSwiss/PeakWeather) project — an integrated, harmonized, and machine-learning–ready climate dataset.

The goal is to study and adapt PeakWeather's design philosophy to **African data realities**, such as sparse ground station observations, multimodal data sources, and irregular spatiotemporal grids.

Ultimately, MasharikiWeather provides a **multi-variable benchmark dataset** supporting **physics-based**, **AI-based**, and **hybrid forecasting** pipelines across frameworks including **PyTorch**, **TensorFlow**, **JAX**, and **NumPy**.

---

## Authors

Developed at the **Centre for Data Science and Artificial Intelligence (DSAIL)**, Dedan Kimathi University of Technology:
- Teofilo Ligawa
- Samuel Mbugua
- James Muiru
- Hilda Muthoni
- Ciira wa Maina

---

## Documentation

Full API references, data models, architecture, and user guides are available at:
- **Read the Docs:** [https://masharikiweather.readthedocs.io/](https://masharikiweather.readthedocs.io/)

To build and view documentation locally:
```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
python -m http.server 8080 --directory docs/_build/html
```

---

## Usage

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
pip install masharikiweather
```

### Authentication (Bring Your Own Key)

MasharikiWeather adheres to a Bring-Your-Own-Key (BYOK) architecture. External API keys can be passed explicitly via arguments, environment variables (`TAHMO_API_KEY`, `CDSAPI_KEY`, `EARTHDATA_USERNAME`), standard configuration files (`~/.cdsapirc`, `~/.netrc`), or local `.env` files.

For account creation and credential setup, consult the official documentation:
- Copernicus Climate Data Store: [Copernicus CDS User Guide](https://cds.climate.copernicus.eu/user-guide)
- NASA Earthdata / IMERG: [NASA earthaccess Documentation](https://earthaccess.readthedocs.io/en/stable/)

If reading from remote Hugging Face stores, you can authenticate via the CLI (`huggingface-cli login`) or pass a token to the dataset constructor.

### Quickstart

```python
from masharikiweather import MasharikiWeatherDataset, pull_weather_data
import matplotlib.pyplot as plt

# 1. Initialize dataset with local storage and physical temporal frequency
dataset = MasharikiWeatherDataset(
    root="./mashariki_data",
    source_obs=["tahmo", "ghcn"],
    years=[2023, 2024],
    freq="h",
    aggregation_methods={"precipitation": "sum", "temperature": "mean"},
)

# 2. Retrieve separated observation strands with consistent DatetimeIndex and MultiIndex
df_tahmo, mask_tahmo = dataset.get_tahmo_observations(
    stations=["TA00001", "TA00016"],
    parameters=["temperature", "precipitation"],
    first_date="2023-01-01",
    last_date="2023-01-07",
    return_mask=True,
)

df_ghcn = dataset.get_ghcn_observations(
    parameters=["temperature"],
    first_date="2023-01-01",
    last_date="2023-01-07",
)

# 3. Synchronize strands for cross-analysis
aligned = dataset.align_observations({"tahmo": df_tahmo, "ghcn": df_ghcn})

# 4. Extract Gridded Satellite/Reanalysis context at station locations
gridded = dataset.get_gridded_for_stations(
    groups=["era5"],
    stations=["TA00001", "TA00016"],
    variables=["total_precipitation"],
    method_overrides={"total_precipitation": "linear"},
)

# 5. Generate aligned rolling window tensors for machine learning
windows = dataset.get_windows(
    window_size=24,
    horizon_size=6,
    stations=["TA00001", "TA00016"],
    gridded_url=["era5"],
)

print("Features (x):", windows.x.shape)
print("Targets (y):", windows.y.shape)
```

## Objectives

1. **Reproduce and understand the PeakWeather pipeline**
   - Explore dataset schema, preprocessing design, and data fusion principles.
2. **Develop an Africa-centered multi-source fusion framework**
   - Harmonize station, reanalysis, satellite, and static prior datasets in a unified structure.
3. **Build a benchmark-ready, multi-variable dataset**
   - Include precipitation, temperature, humidity, solar radiation, wind, and other atmospheric variables.
4. **Enable framework-agnostic ML integration**
   - Support export and loading across ML frameworks using formats like **Zarr**, **NetCDF**, and **HDF5**.
5. **Advance African climate AI infrastructure**
   - Provide standardized, transparent, and reproducible weather datasets tailored to African needs.

---

## Core Concept

Africa's meteorological landscape is characterized by:
- Sparse ground observations (TAHMO, GHCNd, CDS In-Situ).  
- Diverse gridded data products (ERA5, CHIRPS, TAMSAT, IMERG).  
- Static surface properties that influence local weather (elevation, slope, aspect, land cover).  
- Spatial and temporal inconsistencies across sources.  

MasharikiWeather addresses these challenges through:
- **Spatiotemporal Graph Learning** connecting stations, satellite, and reanalysis data.  
- **Integration of static priors** to capture topographic and land-surface context.  
- **Unified variable alignment** for consistent modeling inputs.  
- **Multi-scale representation**, enabling both local and continental evaluation.  
- **ML-ready exports**, inspired by PeakWeather's compatibility-first design.

---

## Data Sources

| Source | Type | Coverage | Variables | Role |
|--------|------|-----------|------------|------|
| **TAHMO** | In-situ (stations) | Sub-Saharan Africa | Precipitation, Temperature, Solar Radiation, Relative Humidity | Ground truth |
| **CDS In-Situ** | In-situ (stations) | Global / Africa | Surface temperature, pressure, wind, humidity | WMO ground observations |
| **GHCN (NOAA)** | In-situ (stations) | Global / Africa | Daily and hourly precipitation, max/min temperature | Climatological reference |
| **ERA5** | Reanalysis | Global | Full atmospheric suite (surface & pressure levels) | Baseline physics |
| **IMERG** | Satellite | Global | Precipitation (half-hourly, V07 Final & V08 Late) | High-resolution rainfall |
| **CHIRPS** | Satellite + Gauge | 1981–Present | Precipitation | Long-term rainfall |
| **TAMSAT** | Satellite | Africa | Precipitation | Bias-corrected rainfall |
| **CAMS** | Reanalysis | Global | Aerosols, PM2.5, PM10, reactive gases | Atmospheric composition |
| **MERRA-2** | Reanalysis | Global | Meteorology, aerosol optical depth | NASA atmospheric baseline |
| **Open-Meteo** | Reanalysis / NWP | Global | Multimodel weather parameters | Low-latency queries |
| **NASA POWER** | Satellite / Reanalysis | Global | Solar irradiance, meteorology | Surface solar and agroclimatology |
| **PVGIS** | Satellite / Solar | Europe, Africa | Solar irradiance, PV performance | Photovoltaic modeling |
| **SARAH-3** | Satellite | Europe, Africa | Surface solar irradiance, DNI | EUMETSAT solar reference |
| **LandSAF** | Satellite | Africa, Europe | Surface temperature, albedo, DSSF | Land surface processes |
| **TROPOMI** | Satellite | Global | NO2, CO, CH4, O3, SO2 | Sentinel-5P trace gas monitoring |
| **Static Priors (EE)** | Earth Engine Layers | Africa | Elevation, Slope, Aspect, Land Cover, Distance to Water | Geophysical context |

---

## Project Structure & Architecture

The core package logic is modularized inside `masharikiweather/`:

### Core Data & Preprocessing (`masharikiweather/`)
- **`dataset.py`**: User-facing dataset interface. Implements `MasharikiWeatherDataset`, separated observation accessors (`get_tahmo_observations`, `get_cds_insitu_observations`, `get_ghcn_observations`), cross-strand alignment (`align_observations`), metadata catalogs (`installation.parquet`, `parameters.parquet`), and high-level extraction entrypoints (`pull_station_data`, `pull_gridded_data`, `pull_weather_data`).
- **`extract/apis/`**: Dagster extraction layer. Software-Defined Assets with native temporal partitioning, decoupled IO managers (`ParquetStationIOManager`, `ZarrGriddedIOManager`), and Bring-Your-Own-Key resolution (`CredentialsManager`).
- **`dataloader.py`**: Constructs PyTorch Geometric `HeteroData` graphs with distance-bounded edges and terrain-steered advection.
- **`losses.py` / `metrics.py`**: Evaluation metrics and loss functions.
- **`rechunker.py` / `utils.py`**: Dask and Zarr chunking optimizations.

---

## Alignment with PeakWeather Roadmap

| PeakWeather Focus | MasharikiWeather Adaptation |
|-------------------|-----------------------------|
| Global ML-ready weather dataset | African-focused ML-ready dataset |
| Harmonized across ERA5, GFS, and observations | Fusion of TAHMO, GHCN, CDS In-Situ, ERA5, IMERG, static priors |
| Precipitation-focused benchmarking | Multi-variable (precip, temp, humidity, radiation, topography) |
| Cloud-scale Zarr exports | Cloud and local exports via Zarr, NetCDF, and Parquet |
| Open and reproducible ML access | Reproducible African weather research |

---

## Phased Roadmap

### Phase 1 — PeakWeather Exploration
- Study PeakWeather's documentation, schema, and data loaders.
- Analyze variable harmonization and metadata organization.

### Phase 2 — MasharikiWeather Schema Design
- Define temporal resolution and coordinate references (lat/lon/time).
- Standardize variable names and CF-compliant metadata.

### Phase 3 — TAHMO + ERA5 Integration
- Align station-based and gridded data through interpolation.
- Preserve collocated sensor channels without destructive averaging.
- Store as unified datasets with metadata catalogs.

### Phase 4 — Multi-source Expansion
- Integrate TAHMO, CDS In-Situ, GHCN, ERA5, IMERG, CAMS, MERRA-2, Open-Meteo, NASA POWER, PVGIS, SARAH-3, LandSAF, and TROPOMI.
- Evaluate inter-product correlations and consistency.

### Phase 5 — Integrate Static Priors
- Merge static features (elevation, slope, aspect, land cover, distance to water).
- Cyclically encode aspect angles to prevent angular boundary discontinuities.

### Phase 6 — ML-Ready Export
- Export standardized datasets to Zarr, NetCDF, and Parquet.
- Provide data loaders for PyTorch Geometric, TensorFlow, and JAX.

### Phase 7 — Benchmark & Evaluation
- Implement baseline models using PeakWeather-style workflows.
- Compare performance across variables and African regions.

---

## Guiding Principles

- **Reproducibility** — Version-controlled, scriptable data processing.  
- **Transparency** — Clear documentation for transformation steps.  
- **Scalability** — Built for high-throughput workflows (Dask, Zarr, Parquet).  
- **Inclusivity** — Designed around African data realities.  
- **Framework-agnosticism** — ML-ready for PyTorch, TensorFlow, JAX, and NumPy.

---

## Contributing
We welcome contributions from the DSAIL team and the broader scientific community. Please review our [Contribution Guidelines](CONTRIBUTING.md) before submitting pull requests.

## Credits & Affiliation
Developed at the **Centre for Data Science and Artificial Intelligence (DSAIL)**, Dedan Kimathi University of Technology, inspired by **[PeakWeather](https://arxiv.org/abs/2506.13652)** and **[WeatherBench2](https://sites.research.google/gr/weatherbench/)**.

MasharikiWeather is a contribution toward open, harmonized climate AI infrastructure for Africa.
