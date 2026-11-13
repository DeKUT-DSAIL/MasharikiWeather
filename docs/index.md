# MasharikiWeather Documentation

Welcome to the documentation for **MasharikiWeather**, an open-source, machine-learning-ready meteorological data engine and spatiotemporal pipeline designed for Africa. 

Developed at the **Centre for Data Science and Artificial Intelligence (DSAIL)** at Dedan Kimathi University of Technology by Teofilo Ligawa, Samuel Mbugua, James Muiru, Hilda Muthoni, and Ciira wa Maina, MasharikiWeather addresses the challenges of sparse ground observations, complex topography, and heterogeneous satellite and reanalysis feeds across the African continent.

---

## Authors

* **Teofilo Ligawa** (Centre for Data Science and Artificial Intelligence, Dedan Kimathi University of Technology)
* **Samuel Mbugua** (Centre for Data Science and Artificial Intelligence, Dedan Kimathi University of Technology)
* **James Muiru** (Centre for Data Science and Artificial Intelligence, Dedan Kimathi University of Technology)
* **Hilda Muthoni** (Centre for Data Science and Artificial Intelligence, Dedan Kimathi University of Technology)
* **Ciira wa Maina** (Centre for Data Science and Artificial Intelligence, Dedan Kimathi University of Technology)

---

## Key Capabilities

* **Declarative Orchestration with Dagster:** High-throughput extraction pipeline utilizing Software-Defined Assets, native temporal partitioning, automatic exponential retries, and defined concurrency limits for external APIs.
* **Bring-Your-Own-Key (BYOK) Architecture:** Modular, tiered credential discovery for TAHMO DataHub, NASA Earthdata (IMERG), and ECMWF Copernicus Climate Data Store (CDS).
* **PeakWeather-Inspired Dataset Interface:** Object-oriented `MasharikiWeatherDataset` interface providing offline-first local Parquet and Zarr loading, metadata catalogs (`installation.parquet` and `parameters.parquet`), multi-sensor column preservation, and physical temporal resampling.
* **Physical Sanity and Atmospheric Conservation:** Enforcement of mass, energy, and flux conservation during temporal resampling and spatial extraction.
* **Topographic and Geophysical Priors:** Integration of high-resolution Digital Elevation Models (DEM), slope, Topographic Position Index (TPI), and cyclically encoded terrain aspect angles.

---

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: Core Architecture & Data Engine

architecture
byok_credentials
dagster_extraction
dataset_user_guide
```

```{toctree}
:maxdepth: 1
:caption: Ecosystem Governance & Reference

credentials_and_byok_guide
```

---

## Citation & Affiliation

If you use MasharikiWeather in your research or operational forecasting systems, please cite:

```bibtex
@misc{masharikiweather2026,
  author = {Teofilo Ligawa and Samuel Mbugua and James Muiru and Hilda Muthoni and Ciira {wa Maina}},
  title = {MasharikiWeather: A Machine-Learning-Ready Weather and Climate Dataset for Africa},
  year = {2026},
  institution = {Centre for Data Science and Artificial Intelligence (DSAIL), Dedan Kimathi University of Technology},
  publisher = {Dedan Kimathi University of Technology},
  url = {https://github.com/DeKUT-DSAIL/MasharikiWeather}
}
```
