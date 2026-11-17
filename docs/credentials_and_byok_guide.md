# MasharikiWeather Credentials and BYOK Ingestion Guide

This document defines the Bring Your Own Key (BYOK) credential management architecture, temporal windowing rules (including month/year transitions and leap year handling), and ECMWF ERA5 GRIB file structures in the `masharikiweather` ecosystem.

---

## 1. Bring Your Own Key (BYOK) Architecture

MasharikiWeather enforces a strict Bring Your Own Key (BYOK) policy. No private authentication keys or tokens are hardcoded into the package or bundled in repository code. All extraction engines dynamically discover credentials via flexible precedence hierarchies.

### Credential Resolution Precedence

For any external meteorological service, credentials are discovered in the following order:
1. **Direct Function / Method Arguments:** Explicit arguments passed directly to extraction functions or constructor calls (`api_key`, `api_secret`, `cds_key`, `cds_url`, `username`, `password`, `token`).
2. **Environment Variables:** Standard environment variables defined in the user's shell session (`TAHMO_API_KEY`, `CDSAPI_KEY`, `EARTHDATA_USERNAME`, etc.).
3. **Official Service Configuration Files:** Standard user configuration files placed in the user's home directory (`$HOME/.cdsapirc`, `$HOME/.netrc`).
4. **Optional Environment Files:** If an optional `env_file` argument is supplied or a `.env` file exists in the current working directory or `$HOME`. **A `.env` file is never strictly required.** If keys are passed as arguments or environment variables, the system executes without reading any disk files.

---

## 2. Service-by-Service Authentication Reference

### 2.1 TAHMO DataHub Ground Stations
* **Service Identifier:** `tahmo`
* **Required Secrets:** `api_key` and `api_secret`
* **Discovery Options:**
  - **Direct Arguments:**
    ```python
    from masharikiweather.dataset import extract_stations

    # Direct keyword arguments
    stations = extract_stations(
        source="tahmo",
        start_date="2024-02-20",
        end_date="2024-03-05",
        api_key="your_tahmo_api_key",
        api_secret="your_tahmo_api_secret",
    )
    ```
  - **Environment Variables:**
    ```bash
    export TAHMO_API_KEY="your_tahmo_api_key"
    export TAHMO_API_SECRET="your_tahmo_api_secret"
    ```
  - **Optional `.env` File:**
    ```ini
    TAHMO_API_KEY=your_tahmo_api_key
    TAHMO_API_SECRET=your_tahmo_api_secret
    ```
    ```python
    stations = extract_stations(source="tahmo", env_file="./secrets.env")
    ```

---

### 2.2 Copernicus Climate Data Store (CDS) & ERA5 Reanalysis
* **Service Identifier:** `cds`, `era5`, `cds_insitu`
* **Required Secrets:** `key` (CDS API key) and optional `url`
* **Discovery Options:**
  - **Automatic Discovery (Recommended by Copernicus):**
    If you store your CDS API credentials in your home directory as instructed by ECMWF, `cdsapi.Client()` automatically discovers them without any code arguments:
    ```bash
    # Location: ~/.cdsapirc
    url: https://cds.climate.copernicus.eu/api
    key: <your-cds-api-key>
    ```
    When `cds_key` is not explicitly passed, MasharikiWeather defers directly to `cdsapi`'s native discovery:
    ```python
    from masharikiweather.dataset import extract_gridded

    # Uses native ~/.cdsapirc automatically
    era5_zarr = extract_gridded(
        source="era5",
        start_date="2024-02-25",
        end_date="2024-03-02",
        use_arco=False, # Force CDS API fallback
    )
    ```
  - **Direct Function Arguments:**
    ```python
    era5_zarr = extract_gridded(
        source="era5",
        start_date="2024-01-01",
        end_date="2024-01-05",
        cds_key="your_cds_api_key",
        cds_url="https://cds.climate.copernicus.eu/api",
        use_arco=False,
    )
    ```
  - **Environment Variables:**
    ```bash
    export CDSAPI_KEY="your_cds_api_key"
    export CDSAPI_URL="https://cds.climate.copernicus.eu/api"
    ```
  - **Google Cloud ARCO ERA5 (Zero Credentials):**
    By default, `extract_gridded(source="era5")` prioritizes Google Cloud ARCO-ERA5 (`prefer_arco=True`), which operates via anonymous cloud storage without any API key or registration.

---

### 2.3 NASA Earthdata / IMERG Precipitation
* **Service Identifier:** `imerg`, `earthdata`
* **Required Secrets:** `username` and `password` (or `token`)
* **Discovery Options:**
  - **Direct Arguments:**
    ```python
    from masharikiweather.dataset import extract_gridded

    imerg_zarr = extract_gridded(
        source="imerg",
        start_date="2024-02-28",
        end_date="2024-03-02",
        username="your_earthdata_username",
        password="your_earthdata_password",
    )
    ```
  - **Environment Variables:**
    ```bash
    export EARTHDATA_USERNAME="your_username"
    export EARTHDATA_PASSWORD="your_password"
    # or alternatively:
    export EARTHDATA_TOKEN="your_token"
    ```
  - **Netrc Configuration File:**
    ```bash
    # Location: ~/.netrc
    machine urs.earthdata.nasa.gov login <username> password <password>
    ```

---

### 2.4 Zero-Key Public REST APIs
The following reanalysis, satellite, and solar products operate via zero-key public endpoints and require no API keys:
* **Open-Meteo Global NWP & Archive:** High-resolution ECMWF IFS, NOAA GFS, DWD ICON, and ERA5 seamless time series.
* **NASA POWER:** Agroclimatology and surface solar energy parameters.
* **PVGIS:** Photovoltaic Geographical Information System solar radiation time series.
* **CM SAF SARAH-3:** EUMETSAT surface solar irradiance and DNI.
* **CAMS Air Quality (via Open-Meteo):** Aerosol optical depth (AOD 550nm), particulate matter ($\text{PM}_{2.5}$, $\text{PM}_{10}$), mineral dust, and reactive trace gases ($\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, $\text{O}_3$, $\text{NH}_3$).

---

## 3. Spatiotemporal & Date Boundary Handling

MasharikiWeather provides validated calendar parsing to handle multi-day queries across edge boundaries.

### 3.1 Month-Crossing Queries
When queries span across month boundaries (e.g. `2024-01-28` to `2024-02-04`), temporal partitioners (`generate_n_days_dict` and `TemporalRange.split_by_month`) enforce split points at exact month boundaries. This is mandatory for the CDS API, where request payloads require single-month index arrays (`"month": ["01"]` vs `"month": ["02"]`).

### 3.2 Year-Crossing Queries
When querying across calendar years (e.g. `2023-12-28` to `2024-01-05`), chunks are segmented under their respective year keys:
```python
from masharikiweather.extract.apis.era5 import generate_n_days_dict

chunks = generate_n_days_dict("2023-12-28", "2024-01-05", n_days=5)
# chunks:
# {
#   2023: {('28_12', '31_12'): ['28', '29', '30', '31']},
#   2024: {('01_01', '05_01'): ['01', '02', '03', '04', '05']}
# }
```

### 3.3 Leap Year Management
Leap year cycles are verified via `calendar.monthrange(year, month)`:
* **Leap Years (e.g., 2024, 2020):** February contains 29 days (`2024-02-29` is included).
* **Common Years (e.g., 2023, 2025):** February contains 28 days.
* A query spanning `2024-02-27` to `2024-03-02` transitions seamlessly from `2024-02-29` to `2024-03-01`.

---

## 4. ECMWF ERA5 GRIB Architecture & Step Types

ECMWF Climate Data Store delivers raw ERA5 model levels and single levels in WMO GRIB edition 1 or 2 format.

### 4.1 Instantaneous vs. Accumulated Variables
ERA5 single-level parameters belong to two fundamentally different meteorological categories:

| Parameter Category | Example Variables | GRIB `stepType` | Physical Representation |
| :--- | :--- | :--- | :--- |
| **Instantaneous** | `2t` (2m temperature), `10u`/`10v` (10m wind), `sp` (surface pressure), `msl` (mean sea level pressure), `tcc` (cloud cover), `tcwv` (water vapour), `sst`, `swvl1` | `instant` | State of the atmosphere at valid timestamp $t$. Step is $0\text{ h}$. |
| **Accumulated** | `tp` (total precipitation), `cp` (convective precip), `ssrd` (solar radiation downwards), `strd` (thermal radiation downwards), `sshf` (sensible heat flux), `slhf` (latent heat flux), `e` (evaporation) | `accum` | Integrated total accumulated over the preceding 1-hour forecast window ($t-1$ to $t$). |

### 4.2 GRIB Reading via `open_era5_grib`
Opening mixed GRIB files with `cfgrib.open_dataset()` without filtering causes `eccodes.DatasetBuildError: multiple values for unique key, try filter_by_keys`.

MasharikiWeather implements `open_era5_grib()` in [`masharikiweather/extract/apis/era5.py`](../masharikiweather/extract/apis/era5.py):
* It filters `stepType: "instant"` and `stepType: "accum"` in separate passes.
* It aligns their spatiotemporal coordinates and merges them cleanly into a single unified `xarray.Dataset`.
* **Physical Units Policy:** Per project specifications, **no physical unit conversions are applied** at extraction time. Raw units ($J/\text{m}^2$ for radiation/energy fluxes, $\text{m}$ for precipitation, $\text{K}$ for temperature) are strictly preserved.

---
