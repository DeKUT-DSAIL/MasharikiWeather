# MasharikiWeather

## Overview

**MasharikiWeather** is an open experimental initiative to create an **ML-ready, framework-agnostic weather dataset for East Africa**.  
It draws inspiration from the [**PeakWeather**](https://huggingface.co/datasets/MeteoSwiss/PeakWeather) project — an integrated, harmonized, and machine-learning–ready global climate dataset.

**At the moment, this is to remain an in-house tool for DSAIL.**

The goal is to **study and reproduce PeakWeather’s design philosophy**, adapting its core principles to **African data realities** such as sparse station coverage, multimodal data sources, and irregular spatiotemporal grids.

Ultimately, MasharikiWeather aims to be a **multi-variable benchmark dataset** that supports **physics-based**, **AI-based**, and **hybrid forecasting** pipelines across frameworks like **PyTorch**, **TensorFlow**, **JAX**, and **NumPy**.

---

## Usage
Create a python virtual environment and activate it.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package.

```bash
pip install masharikiweather
```

### Authentication

This pipeline streams data directly from the DeKUT-DSAIL/weather-data Hugging Face repository. You must have a specific Hugging Face Access Token.

- If running locally, you can authenticate via the CLI: `huggingface-cli login`

- If running in Colab, securely store your token in the Colab Secrets manager.


### Quickstart

```python
from masharikiweather import MasharikiWeatherDataset

# 1. Initialize the Pipeline (Handles caching and network fusion)
ds = MasharikiWeatherDataset(
    repo_id="DeKUT-DSAIL/weather-data",
    token="YOUR_HF_TOKEN", 
    source_obs=["tahmo", "ghcnd"], # Fusing hourly and daily networks
    freq="h", 
    years=[2023, 2024]
)

# 2. Extract Gridded Satellite/Reanalysis Context
gridded_data = ds.get_gridded_for_stations(
    groups=["era5"], 
    stations=['TA00001', 'TA00283'], 
    variables=['total_precipitation'],
    method="linear" # Bilinear interpolation
)

# 3. Generate ML Tensors (Aligned and Windowed)
ml_tensors = ds.get_windows(
    window_size=24,  # 24 hours of historical context
    horizon_size=6,  # 6 hours of prediction
    stations=['TA00001', 'TA00283'],
    gridded_url=["era5"],
    as_xarray=True
)

print(ml_tensors.x) # Your aligned features
print(ml_tensors.y) # Your targets
```

## Objectives

1. **Reproduce and understand the PeakWeather pipeline**
   - Explore its dataset schema, preprocessing philosophy, and data fusion principles.
2. **Develop an East Africa-centered multi-source fusion framework**
   - Harmonize **station**, **reanalysis**, **satellite**, and **static prior** datasets in a unified structure.
3. **Build a benchmark-ready, multi-variable dataset**
   - Include precipitation, temperature, humidity, solar radiation, wind, and other key atmospheric variables.
4. **Enable framework-agnostic ML integration**
   - Support easy export and loading across ML frameworks using formats like **Zarr**, **NetCDF**, and **HDF5**.
5. **Advance East African climate AI infrastructure**
   - Provide standardized, transparent, and reproducible weather datasets tailored to African needs.

---

## Core Concept

East Africa’s meteorological landscape is characterized by:
- Sparse ground observations (TAHMO, GHCNd).  
- Diverse gridded data products (ERA5, CHIRPS, TAMSAT, IMERG).  
- Static surface properties that influence local weather (elevation, slope, aspect, land cover).  
- Spatial and temporal inconsistencies across sources.  

MasharikiWeather seeks to **bridge these gaps** through:
- **Spatiotemporal Graph Learning** of station, satellite, and reanalysis data.  
- **Integration of static priors** to capture topographic and land–surface context.  
- **Unified variable alignment** for consistent modeling inputs.  
- **Multi-scale representation**, enabling both local and continental model evaluation.  
- **ML-ready exports**, inspired by PeakWeather’s compatibility-first design.

---

## Data Sources

| Source | Type | Coverage | Variables | Role |
|--------|------|-----------|------------|------|
| **TAHMO** | In-situ (stations) | Sub-Saharan Africa | Precipitation, Temperature | Ground truth |
| **ERA5** | Reanalysis | Global | Full atmospheric suite | Physics-based baseline |
| **CHIRPS** | Satellite + Gauge | 1981–Present | Precipitation | Long-term rainfall |
| **TAMSAT** | Satellite | Africa | Precipitation | Bias-corrected rainfall |
| **IMERG** | Satellite | Global | Precipitation | Half-houly rainfall |
| **Static Priors (EE)** | Earth Engine Layers | Africa | Elevation, Slope, Aspect, Land Cover, Distance to Water | Geophysical context |
| **(Future)** ECMWF ML, FuXi, GraphCast, FourCastNet | Global | Precip, Temp, Wind, Radiation | ML & hybrid forecasts |

---

## Alignment with PeakWeather Roadmap

| PeakWeather Focus | MasharikiWeather Adaptation |
|--------------------|------------------------------|
| Global ML-ready weather dataset | East African-focused ML-ready dataset |
| Harmonized across ERA5, GFS, and observations | Fusion of TAHMO, ERA5, CHIRPS, TAMSAT, static priors |
| Precipitation-focused benchmarking | Multi-variable (precip, temp, humidity, radiation, topography) |
| Cloud-scale Zarr exports | Cloud and local exports via Zarr / NetCDF |
| Open and reproducible ML access | Reproducible African weather research |

---

## Phased Roadmap

### Phase 1 — PeakWeather Exploration
- Study PeakWeather’s documentation, schema, and data loaders.
- Analyze its variable harmonization and metadata organization.
- Run sample ML-ready preprocessing on a small African region.

### Phase 2 — MasharikiWeather Schema Design
- Define temporal resolution (e.g., 6-hourly or daily).
- Define spatial structure (station points vs gridded data).
- Standardize variable names and CF-compliant metadata.
- Establish coordinate references (lat/lon/time).

### Phase 3 — TAHMO + ERA5 Integration
- Align station-based and gridded data through nearest-grid or interpolation.
- Handle irregular sampling and missing timestamps.
- Store as unified `xarray.Dataset` with metadata and attributes.

### Phase 4 — Multi-source Expansion
- Add CHIRPS, IMERG and TAMSAT for multi-sensor rainfall comparison.
- Incorporate temperature, humidity, radiation, and wind from ERA5.
- Evaluate inter-product correlations, bias, and consistency.

### Phase 5 — Integrate Static Priors
- Merge Earth Engine static features (elevation, slope, aspect, land cover, distance to water).
- Harmonize to match ERA5 and CHIRPS grids.
- Enable topography-aware model development.

### Phase 6 — ML-Ready Export
- Export standardized, chunked datasets to **Zarr** and **NetCDF**.
- Develop lightweight data loaders for **PyTorch**, **TensorFlow**, and **JAX**.
- Preserve metadata and normalization info for each variable.

### Phase 7 — Benchmark & Evaluation
- Implement baseline models using PeakWeather-style workflows.
- Compare model performance across variables and regions.
- Publish visual and quantitative evaluations.

---


## Guiding Principles

- **Reproducibility** — Version-controlled, scriptable data processing.  
- **Transparency** — Clear documentation for every transformation step.  
- **Scalability** — Built for cloud-scale workflows (DVC, Prefect, Zarr).  
- **Inclusivity** — Designed around African data sources and use cases.  
- **Framework-agnosticism** — ML-ready for PyTorch, TensorFlow, and beyond.

---

## Contributing
We welcome active experimentation and stress-testing from the DSAIL team! Whether you are testing a new spatial masking technique, adding a new satellite data source, or optimizing the data loaders, we want your contributions.

To ensure the core engine remains stable while we experiment, please review our [Contribution Guidelines](CONTRIBUTING.md) before pushing code. All new features and experiments should be developed on a separate branch and submitted via a Pull Request (PR) for peer review.

## Credits
Developed as part of an effort to **advance localized, data-driven weather prediction for East Africa**,  
inspired by **[PeakWeather](https://arxiv.org/abs/2506.13652)** and **[WeatherBench2](https://sites.research.google/gr/weatherbench/)**.  

MasharikiWeather is a **step toward open, harmonized, and equitable climate AI infrastructure for East Africa**.
