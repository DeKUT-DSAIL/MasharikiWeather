# MasharikiWeather Documentation

Welcome to the **MasharikiWeather** documentation. MasharikiWeather is an open-source Python library and experimental dataset interface developed at the **Centre for Data Science and Artificial Intelligence (DSAIL)**, Dedan Kimathi University of Technology (DeKUT). It provides standardized tools for downloading, processing, harmonizing, and loading meteorological observations, satellite products, reanalysis fields, and static topographic priors across Africa.

The framework is tailored to the observational, convective, and topographic realities of the African continent.

---

## Sequential Documentation Roadmap

MasharikiWeather follows an offline-first, pipeline-oriented lifecycle. New users, contributors, and researchers should follow the documentation in this chronological sequence:

```text
+-----------------------+     +-----------------------+     +-----------------------+     +-----------------------+
|  Step 1: Setup &      |     |  Step 2: Data         |     |  Step 3: Dataset      |     |  Step 4: System       |
|  Credentials          | --> |  Extraction           | --> |  Ingestion & OOP      | --> |  Architecture &       |
|  (credentials.md)     |     |  (extraction.md)      |     |  (dataset_user_guide) |     |  Physical Standards   |
+-----------------------+     +-----------------------+     +-----------------------+     +-----------------------+
```

| Step | Guide | Description & Focus |
|---|---|---|
| **Step 1** | [**Environment Setup & Credentials**](credentials.md) | Configure authentication tokens (`.env`, `~/.cdsapirc`, `HF_TOKEN`, `CDSAPI_KEY`, TAHMO, Earthdata). |
| **Step 2** | [**Data Extraction Pipelines**](extraction.md) | Download in-situ stations (TAHMO 33-variables), ERA5 (Google ARCO -> CDS cascade), satellites (IMERG), and topography Zarr into local storage (`DATA_DIR`). **Must be completed before Step 3.** |
| **Step 3** | [**Dataset Ingestion & Data Interface**](dataset_user_guide.md) | Instantiate `MasharikiWeatherDataset(root_dir=DATA_DIR)`, inject topography into station metadata, query scoped station/parameter tables, and slice 2-level MultiIndex observations `(station_id, parameter)`. |
| **Step 4** | [**System Architecture & Physical Standards**](architecture.md) | Deep-dive into physical conservation laws, CF metadata standards, and coordinate systems. |

---

## Quick Start: The End-to-End Workflow

### Step 1: Configure Credentials
Place a `.env` file in your project root directory (or export environment variables):
```bash
HF_TOKEN="hf_your_personal_access_token"
TAHMO_API_KEY="your-tahmo-api-key"
TAHMO_API_SECRET="your-tahmo-api-secret"
CDSAPI_KEY="your-cds-api-key"
EARTHDATA_USERNAME="your-earthdata-username"
EARTHDATA_PASSWORD="your-earthdata-password"
```

### Step 2: Extract Weather Data (In-Situ & Gridded)
Data extraction runs first, populating the local target directory (`./mashariki_data`):
```python
from masharikiweather.dataset import extract_station, extract_gridded

# 1. Extract TAHMO weather stations (all 33 parameters, 1-year chunks)
extract_station(
    source="tahmo",
    start_date="2023-01-01",
    end_date="2023-12-31",
    output_dir="./mashariki_data/obs/tahmo",
)

# 2. Extract gridded atmospheric reanalysis (ERA5 Google ARCO -> CDS cascade)
ds_era5 = extract_gridded(
    source="era5",
    start_date="2023-01-01",
    end_date="2023-01-07",
    bbox=[-5.0, 33.0, 5.0, 42.0],  # East Africa domain
    variables=["2m_temperature", "total_precipitation"],
    output_dir="./mashariki_data/gridded/era5",
    save_zarr=True,
)
```

### Step 3: Instantiate Dataset and Query Observations
Now initialize `MasharikiWeatherDataset` against your local data directory:
```python
from masharikiweather.dataset import MasharikiWeatherDataset

# Instantiate dataset (automatically enriches station tables with topography from .zarr)
ds = MasharikiWeatherDataset(
    root="./mashariki_data",
    years=[2023],
    source_obs=["tahmo"],
    freq="h",
)

# Inspect source-scoped parameter catalog (all 33 TAHMO parameters)
print("Parameters:", len(ds.parameters_table))
print(ds.parameters_table.head())

# Inspect station inventory (enriched with topography, non-NaN lifecycles, and sensor counts)
print("Stations:", len(ds.stations_table))
print(ds.stations_table[["station_name", "latitude", "longitude", "dem", "Slope", "Aspect"]].head())

# Load observations with two-level MultiIndex slicing: (station_id, parameter)
obs = ds.load()
print(obs["TA00001", "2m_temperature"])
print(obs["TA00001"]["total_precipitation"])
```

### Step 4: Spatiotemporal Modeling & Graph Learning
Generate sliding windows and PyTorch Geometric `HeteroData` graphs:
```python
# Construct rolling windows for spatio-temporal ML models
windows = ds.get_windows(
    input_steps=24,   # 24 hours of history
    output_steps=6,   # 6 hours ahead forecast
)
print("Input Tensor Shape:", windows.inputs.shape)    # (N_samples, T_in, N_stations, C_features)
print("Target Tensor Shape:", windows.targets.shape)  # (N_samples, T_out, N_stations, C_targets)
```

---

## Repository Structure

```text
docs/
|-- README.md             # This document (GitHub root index for docs/)
|-- index.md              # Sphinx / ReadTheDocs table of contents
|-- conf.py               # Sphinx configuration with autodoc and myst-parser
|-- credentials.md        # Step 1: Environment Setup & API Authentication
|-- extraction.md         # Step 2: Data Extraction Pipelines (TAHMO, ERA5, Satellites)
|-- dataset_user_guide.md # Step 3: Dataset Ingestion, Metadata & Data Interface
\-- architecture.md       # Step 4: System Architecture, Physical Conservation & CF Standards
```

---

## Citation

If you use MasharikiWeather in your academic research, please cite:

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
