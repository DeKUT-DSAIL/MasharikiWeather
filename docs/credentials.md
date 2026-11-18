# Step 1: Environment Setup & Credentials

Before downloading or querying data with MasharikiWeather, configure the required environment variables and API credentials. MasharikiWeather discovers credentials automatically without requiring hardcoded secrets in user scripts.

> **Sequential Workflow Position:**
> 1. **Step 1: Credentials & Environment Setup** (This guide)
> 2. **Step 2: Data Extraction Pipelines** ([extraction.md](extraction.md))
> 3. **Step 3: Dataset Ingestion & PeakWeather Interface** ([dataset_user_guide.md](dataset_user_guide.md))
> 4. **Step 4: System Architecture & Physical Standards** ([architecture.md](architecture.md))

---

## Overview

When an extractor or dataset loader requires authentication, it discovers credentials in the following hierarchical order:

1. **Explicit Function Arguments:** Parameters passed directly to extractors or datasets (e.g., `token="..."`, `cds_key="..."`, `api_key="..."`).
2. **Environment Variables:** Variables exported in the current shell session or execution environment (e.g., `HF_TOKEN`, `CDSAPI_KEY`, `TAHMO_API_KEY`).
3. **Configuration Files:** Standard configuration files in the user home directory (`~/.cdsapirc`, `~/.netrc`, `~/.cache/huggingface/token`).
4. **Local `.env` File:** Environment variables defined in a `.env` file in the current working directory, parsed automatically via `python-dotenv`.

If credentials are provided via environment variables, function arguments, or `.env` files, no home-directory configuration files are required.

---

## Service Credentials Matrix

| Service / Provider | Target Datasets | Environment Variables | Config File | Registration Link |
|---|---|---|---|---|
| **Hugging Face Hub** | Continental Topography Zarr, Preprocessed Archives | `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN` | `~/.cache/huggingface/token` | [Hugging Face User Settings](https://huggingface.co/settings/tokens) |
| **TAHMO DataHub** | In-situ ground stations (Africa) | `TAHMO_API_KEY`, `TAHMO_API_SECRET` | None | [TAHMO Portal](https://tahmo.org/) |
| **Copernicus CDS** | ERA5 NetCDF4 reanalysis, CAMS atmospheric composition | `CDSAPI_KEY` | `~/.cdsapirc` | [Copernicus CDS User Guide](https://cds.climate.copernicus.eu/user-guide) |
| **NASA Earthdata** | IMERG precipitation, MERRA-2 reanalysis | `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD` | `~/.netrc` | [NASA Earthdata Login](https://urs.earthdata.nasa.gov/) |
| **Google Cloud ARCO** | ERA5 Analysis-Ready Cloud Zarr | None (Public open-access bucket) | None | None |
| **Copernicus CDS In-Situ** | Surface station observations | None (Public open access) | None | None |
| **NOAA GHCNd** | Daily station records | None (Public open access) | None | None |
| **Open-Meteo** | Numerical weather reanalysis | None (Public open access) | None | None |

---

## Configuration Examples

### 1. Local `.env` File (Recommended)

Place a `.env` file in your project root directory (ensure `.env` is listed in `.gitignore`):

```bash
# Hugging Face Hub (Topography Zarr Streaming & Datasets)
HF_TOKEN="hf_your_personal_access_token"

# TAHMO DataHub Credentials
TAHMO_API_KEY="your-tahmo-api-key"
TAHMO_API_SECRET="your-tahmo-api-secret"

# Copernicus Climate Data Store
CDSAPI_KEY="your-cds-api-key"

# NASA Earthdata (IMERG, MERRA-2)
EARTHDATA_USERNAME="your-earthdata-username"
EARTHDATA_PASSWORD="your-earthdata-password"
```

### 2. Shell Environment Variables

```bash
export HF_TOKEN="hf_your_personal_access_token"
export TAHMO_API_KEY="your-tahmo-api-key"
export TAHMO_API_SECRET="your-tahmo-api-secret"
export CDSAPI_KEY="your-cds-api-key"
export EARTHDATA_USERNAME="your-earthdata-username"
export EARTHDATA_PASSWORD="your-earthdata-password"
```

### 3. Copernicus CDS Configuration File (`~/.cdsapirc`)

```text
url: https://cds.climate.copernicus.eu/api
key: your-cds-api-key
```

### 4. NASA Earthdata Configuration File (`~/.netrc`)

```text
machine urs.earthdata.nasa.gov login your-earthdata-username password your-earthdata-password
```

### 5. Explicit Constructor Parameters

```python
from masharikiweather.dataset import MasharikiWeatherDataset
from masharikiweather.extract.apis.era5 import ERA5Extractor

# Initialize dataset with explicit Hugging Face token
dataset = MasharikiWeatherDataset(
    root="./mashariki_data",
    source_obs=["tahmo"],
    use_hf=True,
    token="hf_your_personal_access_token",
)

# Initialize ERA5 extractor with explicit CDS key
extractor = ERA5Extractor(cds_key="your-cds-api-key")
```
