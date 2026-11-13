# MasharikiWeather Credentials and BYOK Ingestion Guide

This document defines the Bring Your Own Key (BYOK) credential management architecture, temporal windowing rules (including month/year transitions and leap year handling), extraction interfaces for all 13 supported data strands, and ECMWF ERA5 GRIB file structures in the `masharikiweather` ecosystem.

---

## 1. Bring Your Own Key (BYOK) Architecture

MasharikiWeather enforces a Bring Your Own Key (BYOK) policy. No private authentication keys or tokens are hardcoded into the package or bundled in repository code. All extraction engines dynamically discover credentials via flexible precedence hierarchies.

### Credential Resolution Precedence

For any external meteorological service, credentials are discovered in the following order:
1. **Direct Function / Method Arguments:** Explicit arguments passed directly to extraction functions or constructor calls (`api_key`, `api_secret`, `cds_key`, `cds_url`, `username`, `password`, `token`).
2. **Environment Variables:** Standard environment variables defined in the user shell session (`TAHMO_API_KEY`, `CDSAPI_KEY`, `EARTHDATA_USERNAME`, etc.).
3. **Official Service Configuration Files:** Standard user configuration files placed in the home directory (`$HOME/.cdsapirc`, `$HOME/.netrc`).
4. **Optional Environment Files:** If an optional `env_file` argument is supplied or a `.env` file exists in the current working directory or `$HOME`. **A `.env` file is not required.** If keys are passed as arguments or environment variables, the system executes without reading any disk files.

---

## 2. Ingestion Coverage Across All 13 Data Strands

MasharikiWeather supports unified extraction across 13 meteorological data strands, spanning in-situ station networks, reanalysis models, and satellite instruments.

### 2.1 In-Situ Station Networks

#### 2.1.1 TAHMO DataHub (`tahmo`)
* **Authentication:** Requires `api_key` and `api_secret`.
* **Discovery:** Direct arguments, environment variables (`TAHMO_API_KEY`, `TAHMO_API_SECRET`), or optional `.env`.
```python
from masharikiweather.dataset import extract_stations

stations = extract_stations(
    source="tahmo",
    start_date="2023-01-01",
    end_date="2023-01-05",
    api_key="your_tahmo_api_key",
    api_secret="your_tahmo_api_secret",
    output_dir="./mashariki_data/obs/tahmo",
)
```

#### 2.1.2 Copernicus CDS In-Situ (`cds_insitu`)
* **Authentication:** Requires Copernicus CDS API key (via `~/.cdsapirc` or environment variable `CDSAPI_KEY`).
```python
from masharikiweather.dataset import extract_stations

stations = extract_stations(
    source="cds_insitu",
    start_date="2023-01-01",
    end_date="2023-01-05",
    output_dir="./mashariki_data/obs/cds_insitu",
)
```

#### 2.1.3 NOAA GHCN-Daily (`ghcn`)
* **Authentication:** Zero-key public service.
```python
from masharikiweather.dataset import extract_stations

stations = extract_stations(
    source="ghcn",
    start_date="2023-01-01",
    end_date="2023-01-05",
    output_dir="./mashariki_data/obs/ghcn",
)
```

---

### 2.2 Atmospheric Reanalysis Strands

#### 2.2.1 ECMWF ERA5 Reanalysis (`era5`)
* **Authentication:** Anonymous for Google Cloud ARCO-ERA5; requires CDS key (`CDSAPI_KEY` or `~/.cdsapirc`) for Copernicus CDS and CDS ARCO access. Refer to the [Copernicus CDS User Guide](https://cds.climate.copernicus.eu/user-guide).
* **Backends:**
  - `use_legacy=True` (default): Standard CDS API extraction. By default downloads in NetCDF-4 (`data_format="netcdf4"`), bypassing `cfgrib` and its C-dependencies. Features adaptive temporal chunking starting at 5 days that automatically adapts and downsizes if CDS request limits or timeouts occur.
  - `use_cds_arco=True`: Lightweight access to Copernicus CDS analysis-ready cloud Zarr stores via `obstore` HTTPStore and ObjectStore.
    *Query-Aware Routing Matrix:* Routes automatically to geo-chunked or time-chunked stores depending on query shape (or via manual `chunking="geo"` / `"time"` override):
    - Surface (sfc): Geo (`https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr`), Time (`https://arco.datastores.ecmwf.int/cadl-arco-time-002/arco/reanalysis_era5_single_levels/sfc/timeChunked.zarr`)
    - Waves (wav): Geo (`https://arco.datastores.ecmwf.int/cadl-arco-geo-003/arco/reanalysis_era5_single_levels/wav/geoChunked.zarr`), Time (`https://arco.datastores.ecmwf.int/cadl-arco-time-003/arco/reanalysis_era5_single_levels/wav/timeChunked.zarr`)
    - Pressure levels (pl): Geo (`https://arco.datastores.ecmwf.int/cadl-arco-geo-048/arco/reanalysis_era5_pressure_levels/pl/geoChunked.zarr`), Time (`https://arco.datastores.ecmwf.int/cadl-arco-time-048/arco/reanalysis_era5_pressure_levels/pl/timeChunked.zarr`)
  - `use_google_arco=True`: Google Cloud Public Dataset ERA5 ARCO Zarr store.

```python
from masharikiweather.dataset import extract_gridded

# 1. Legacy CDS API with NetCDF-4 and adaptive chunking
era5_legacy = extract_gridded(
    source="era5",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["2m_temperature", "total_precipitation"],
    use_legacy=True,
    data_format="netcdf4",
    chunk_days=5,
    output_dir="./mashariki_data/gridded/era5",
)

# 2. Copernicus CDS ARCO via obstore with query-aware routing
era5_cds_arco = extract_gridded(
    source="era5",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["2m_temperature", "total_precipitation"],
    use_cds_arco=True,
    chunking="geo",  # 'geo' for station time series, 'time' for snapshot maps
    output_dir="./mashariki_data/gridded/era5_cds_arco",
)

# 3. Google Cloud ARCO-ERA5 (zero credentials required)
era5_google = extract_gridded(
    source="era5",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["2m_temperature", "total_precipitation"],
    use_google_arco=True,
    output_dir="./mashariki_data/gridded/era5_google",
)
```

#### 2.2.2 Copernicus Atmosphere Monitoring Service (`cams`)
* **Authentication:** Public REST endpoints via Open-Meteo or CDS API.
```python
from masharikiweather.dataset import extract_gridded

cams_zarr = extract_gridded(
    source="cams",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["aod550", "pm2p5", "pm10"],
    output_dir="./mashariki_data/gridded/cams",
)
```

#### 2.2.3 NASA MERRA-2 Reanalysis (`merra2`)
* **Authentication:** NASA Earthdata Login credentials (`EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD` or `~/.netrc`).
```python
from masharikiweather.dataset import extract_gridded

merra2_zarr = extract_gridded(
    source="merra2",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["T2M", "PRECTOT"],
    output_dir="./mashariki_data/gridded/merra2",
)
```

#### 2.2.4 Open-Meteo Historical Archive (`open_meteo`)
* **Authentication:** Zero-key public REST endpoint.
```python
from masharikiweather.dataset import extract_gridded

om_zarr = extract_gridded(
    source="open_meteo",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["temperature_2m", "precipitation"],
    output_dir="./mashariki_data/gridded/open_meteo",
)
```

---

### 2.3 Satellite Earth Observation Strands

#### 2.3.1 NASA GPM IMERG Precipitation (`imerg`)
* **Authentication:** NASA Earthdata Login credentials (`username`, `password`, or `~/.netrc`). Refer to the [NASA earthaccess Documentation](https://earthaccess.readthedocs.io/en/stable/).
```python
from masharikiweather.dataset import extract_gridded

imerg_zarr = extract_gridded(
    source="imerg",
    start_date="2023-01-01",
    end_date="2023-01-02",
    username="your_earthdata_username",
    password="your_earthdata_password",
    output_dir="./mashariki_data/gridded/imerg",
)
```

#### 2.3.2 NASA POWER Agroclimatology (`nasa_power`)
* **Authentication:** Zero-key public API.
```python
from masharikiweather.dataset import extract_gridded

power_zarr = extract_gridded(
    source="nasa_power",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["ALLSKY_SFC_SW_DWN", "T2M"],
    output_dir="./mashariki_data/gridded/nasa_power",
)
```

#### 2.3.3 PVGIS Solar Radiation (`pvgis`)
* **Authentication:** Zero-key public service.
```python
from masharikiweather.dataset import extract_gridded

pvgis_zarr = extract_gridded(
    source="pvgis",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["G(h)", "T2m"],
    output_dir="./mashariki_data/gridded/pvgis",
)
```

#### 2.3.4 CM SAF SARAH-3 Surface Radiation (`sarah3`)
* **Authentication:** Zero-key public service.
```python
from masharikiweather.dataset import extract_gridded

sarah3_zarr = extract_gridded(
    source="sarah3",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["SIS", "SID"],
    output_dir="./mashariki_data/gridded/sarah3",
)
```

#### 2.3.5 EUMETSAT LandSAF (`landsaf`)
* **Authentication:** Zero-key public archive or Open-Meteo satellite feed.
```python
from masharikiweather.dataset import extract_gridded

landsaf_zarr = extract_gridded(
    source="landsaf",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["LST", "DSSF"],
    output_dir="./mashariki_data/gridded/landsaf",
)
```

#### 2.3.6 Sentinel-5P TROPOMI Atmospheric Composition (`tropomi`)
* **Authentication:** Zero-key public archive or Copernicus Data Space Ecosystem.
```python
from masharikiweather.dataset import extract_gridded

tropomi_zarr = extract_gridded(
    source="tropomi",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["tropospheric_NO2_column_number_density"],
    output_dir="./mashariki_data/gridded/tropomi",
)
```

---

## 3. Spatiotemporal & Date Boundary Handling

MasharikiWeather provides validated calendar parsing to handle multi-day queries across edge boundaries.

### 3.1 Month-Crossing Queries
When queries span across month boundaries (e.g. `2024-01-28` to `2024-02-04`), temporal partitioners (`generate_n_days_dict` and `TemporalRange.split_by_month`) enforce split points at exact month boundaries. This is required for the CDS API, where request payloads specify single-month index arrays (`"month": ["01"]` vs `"month": ["02"]`).

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
* A query spanning `2024-02-27` to `2024-03-02` transitions from `2024-02-29` to `2024-03-01`.

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

MasharikiWeather implements `open_era5_grib()` in `masharikiweather/extract/apis/era5.py`:
* It filters `stepType: "instant"` and `stepType: "accum"` in separate passes.
* It aligns their spatiotemporal coordinates and merges them into a single unified `xarray.Dataset`.
* **Physical Units Policy:** Per project specifications, no physical unit conversions are applied at extraction time. Raw units ($J/\text{m}^2$ for radiation/energy fluxes, $\text{m}$ for precipitation, $\text{K}$ for temperature) are preserved without modification.

### 4.3 NetCDF Reading via `open_era5_netcdf`
To bypass `cfgrib` and `eccodes` C-dependencies, ERA5 extractions can request NetCDF-4 format (`data_format="netcdf4"`).
MasharikiWeather provides `open_era5_netcdf()` in `masharikiweather.extract.apis.era5`:
* Opens downloaded NetCDF files via `netcdf4` or `h5netcdf` engines without requiring `cfgrib`.
* Standardizes coordinate names to CF conventions (`lat`, `lon`, `time`).
* Merges datasets along time and variable dimensions into a unified `xarray.Dataset`.
* Harmonizes variable names to canonical descriptive names when `harmonize=True`.
