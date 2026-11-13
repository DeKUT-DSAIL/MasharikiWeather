# Dataset User Guide

The core user interface of MasharikiWeather is the `MasharikiWeatherDataset` class in `masharikiweather.dataset`. It offers an offline-first interface for reading, slicing, resampling, and visualizing weather observations across Africa.

---

## 1. Getting Started

### 1.1 Local Disk Loading
By default, `MasharikiWeatherDataset` loads local files in `root` (or `root_dir`). It does not require internet connectivity or remote tokens if files are present:

```python
from masharikiweather.dataset import MasharikiWeatherDataset

dataset = MasharikiWeatherDataset(
    root="./mashariki_data",
    years=[2022, 2023, 2024],
    freq="h",
    aggregation_methods={"precipitation": "sum", "temperature": "mean"},
)
```

### 1.2 Automated Data Fetching
If requested years or data files are not found on local disk, `MasharikiWeatherDataset` automatically fetches missing files, displaying progress bars:

```text
Fetching tahmo_station_observations: 100%|██████████| 12/12 [01:14<00:00, 2023-12-01.parquet: 61.2MB [00:06, 4.12MB/s]]
```

Remote Hugging Face streaming is opt-in via `use_hf=True` or by providing a Hugging Face `token`.

### 1.3 Key Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `root` (or `root_dir`) | `str` or `Path` | `"./mashariki_data"` | Local directory storing `obs/`, `gridded/`, and `metadata/` |
| `years` | `Sequence[int]` | `[2018..2024]` | Calendar years included in the dataset |
| `source_obs` | `str` or `List[str]` | `"tahmo"` | Observational source (`'tahmo'`, `'cds_insitu'`, `'ghcn'`) |
| `freq` | `Optional[str]` | `None` | Target temporal frequency (e.g., `'h'`, `'3h'`, `'D'`, or `None` for native resolution) |
| `aggregation_methods` | `Optional[Dict[str, str]]` | `None` | Variable-specific temporal aggregation overrides |
| `parameters` | `Optional[Sequence[str]]` | `None` | Restrict loaded variables to specific parameters |
| `prefer_zarr` (or `use_zarr`) | `bool` | `False` | Prioritize Zarr store reading for TAHMO observations |
| `use_hf` | `bool` | `False` | Opt-in to remote Hugging Face streaming |

---

## 2. Dataset Properties

An instantiated `MasharikiWeatherDataset` object exposes attributes and metadata catalogs:

```python
# Path and frequency configuration
print("Data Root Directory:", dataset.root)
print("Configured Frequency:", dataset.freq)

# Available vs active parameters
print("Available Parameters:", dataset.available_parameters)
print("Active Parameters:", dataset.parameters)

# Available vs active years
print("Available Years:", dataset.available_years)

# Active stations
print("Active Stations (Count: {}):".format(len(dataset.stations)))
print(dataset.stations[:5])
```

### Metadata Catalogs (`installation.parquet` & `parameters.parquet`)

Metadata catalogs are maintained under `{root}/metadata/`:
* **Station Catalog (`dataset.installation`):** DataFrame containing station identifiers, geographic coordinates (latitude, longitude), elevation, station names, network provider, and operational date ranges.
* **Parameter Catalog (`dataset.parameters_table`):** DataFrame containing canonical variable names, CF standard names, physical units, measurement types, and default temporal resampling methods.

```python
# Inspect station installation catalog
stations_df = dataset.installation
print(stations_df[["station_name", "latitude", "longitude", "elevation"]].head())

# Inspect parameter catalog
params_df = dataset.parameters_table
print(params_df[["standard_name", "units", "resampling_method"]])
```

---

## 3. Observation Retrieval by Strand

Rather than bundling all tabular data into a single method, MasharikiWeather provides dedicated methods for each observational network strand. Each method returns a standardized `pd.DataFrame` with a tz-naive `pd.DatetimeIndex` named `'time'` and a 2-level column `MultiIndex` formatted as `(station_id, parameter)`.

### 3.1 TAHMO Observations (`get_tahmo_observations`)

Retrieves in-situ observations from the Trans-African Hydro-Meteorological Observatory (TAHMO) network:

```python
# Retrieve temperature and precipitation for realistic TAHMO stations
df_tahmo, mask_tahmo = dataset.get_tahmo_observations(
    stations=["TA00001", "TA00016"],
    parameters=["temperature", "precipitation"],
    first_date="2023-01-01 00:00",
    last_date="2023-01-07 23:00",
    return_mask=True,
)

print(df_tahmo.head())
```

Output DataFrame structure:

```text
                                  TA00001                                  TA00016             
                      temperature precipitation                temperature precipitation  
time                                                                                       
2023-01-01 00:00:00          21.4           0.0                       20.1           0.0
2023-01-01 01:00:00          20.8           0.2                       19.7           0.0
```

### 3.2 Copernicus In-Situ Observations (`get_cds_insitu_observations`)

Retrieves global WMO surface station measurements from the Copernicus Climate Data Store:

```python
df_cds = dataset.get_cds_insitu_observations(
    stations=["CDS_NBO"],
    parameters=["temperature", "surface_pressure"],
    first_date="2023-01-01",
    last_date="2023-01-07",
)
print(df_cds.head())
```

### 3.3 NOAA GHCN Observations (`get_ghcn_observations`)

Retrieves daily or hourly measurements from NOAA's Global Historical Climatology Network:

```python
df_ghcn = dataset.get_ghcn_observations(
    stations=["KE000063612"],
    parameters=["temperature", "precipitation"],
    first_date="2023-01-01",
    last_date="2023-01-07",
)
print(df_ghcn.head())
```

### 3.4 Unified Observation Retrieval (`get_observations`)

The legacy entrypoint `get_observations()` remains available and returns all active observation sources combined:

```python
df_all = dataset.get_observations(
    parameters=["temperature"],
    first_date="2023-01-01",
    last_date="2023-01-07",
)
```

### 3.5 Handling Missing Station Permissions
If a user requests stations outside their licensed API key coverage or absent from local storage, observation methods issue an advisory warning rather than aborting:

```python
df = dataset.get_tahmo_observations(stations=["TA00001", "NONEXISTENT_STATION"])
# UserWarning: Requested station(s) not in dataset: ['NONEXISTENT_STATION']. Continuing with available stations.
```

### 3.6 Quality Masks and NumPy Export
When `return_mask=True`, a parallel boolean mask DataFrame is returned where `True` indicates a valid measured value and `False` indicates missing or corrupted data.

For tensor-oriented workflows, setting `as_numpy=True` exports 3D arrays:

```python
arr, mask_arr = dataset.get_tahmo_observations(
    stations=["TA00001", "TA00016"],
    parameters=["temperature", "precipitation"],
    as_numpy=True,
    return_mask=True,
)

# Output tensor shape: (T_time_steps, N_stations, C_parameters)
print("Data Tensor Shape:", arr.shape)
print("Mask Tensor Shape:", mask_arr.shape)
```

---

## 4. Cross-Analysis and Multi-Strand Alignment

Because all separated methods share a consistent index (tz-naive `pd.DatetimeIndex` named `'time'`) and column schema (`pd.MultiIndex` with levels `['station_id', 'parameter']`), cross-analysis between strands is direct.

### 4.1 Temporal Alignment via `align_observations`

The static method `MasharikiWeatherDataset.align_observations()` joins multiple observation strands onto a unified temporal grid with a 3-level MultiIndex `(source, station_id, parameter)`:

```python
strands = {
    "tahmo": dataset.get_tahmo_observations(parameters=["temperature"]),
    "ghcn": dataset.get_ghcn_observations(parameters=["temperature"]),
    "cds_insitu": dataset.get_cds_insitu_observations(parameters=["temperature"]),
}

# Align strands across their temporal union
aligned_df = MasharikiWeatherDataset.align_observations(strands, how="outer")
print("Aligned Columns (3 levels):", aligned_df.columns.names)
# ['source', 'station_id', 'parameter']
print(aligned_df.head())
```

### 4.2 Standard Pandas Joins and Concat

Standard pandas operations can also be used directly:

```python
# Join two strands side by side
joined_df = df_tahmo.join(df_ghcn, how="outer")

# Concatenate across column axis
merged_df = pd.concat({"tahmo": df_tahmo, "ghcn": df_ghcn}, axis=1)
```

### 4.3 Multi-Strand Plotting on a Single Axis

```python
import matplotlib.pyplot as plt

df_tahmo = dataset.get_tahmo_observations(stations=["TA00001"], parameters=["temperature"])
df_ghcn = dataset.get_ghcn_observations(stations=["KE000063612"], parameters=["temperature"])

fig, ax = plt.subplots(figsize=(12, 4))
if not df_tahmo.empty:
    ax.plot(df_tahmo.index, df_tahmo["TA00001", "2m_temperature"], label="TAHMO TA00001 Temp", color="#1f77b4")
if not df_ghcn.empty:
    ax.plot(df_ghcn.index, df_ghcn["KE000063612", "2m_temperature"], label="GHCN KE000063612 Temp", color="#ff7f0e", linestyle="--")

ax.set_title("Cross-Strand Temperature Comparison")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Temperature (°C)")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
```

---

## 5. TAHMO Zarr Store Loading

While station observations default to partitioned Apache Parquet on local disk, MasharikiWeather maintains the capability to load TAHMO data from a Zarr store when explicitly requested.

```python
# Load TAHMO observations prioritizing local or remote Zarr store
dataset_zarr = MasharikiWeatherDataset(
    root="./mashariki_data",
    source_obs=["tahmo"],
    prefer_zarr=True,  # or use_zarr=True
    years=[2023],
    freq="h",
)

df_tahmo = dataset_zarr.get_tahmo_observations(stations=["TA00001"])
print("Loaded from Zarr:", df_tahmo.shape)
```

Candidate locations searched when `prefer_zarr=True`:
1. Explicit path passed via `zarr_path` or `tahmo_zarr_path` argument
2. `{root}/obs/tahmo.zarr`
3. `{root}/obs/tahmo.zarr.zip`
4. Remote Hugging Face store (when `use_hf=True`)

---

## 6. Physical Temporal Resampling

Atmospheric data transformations must respect thermodynamic sanity and fluid dynamic conservation laws. Simple uniform averaging distorts precipitation fluxes and radiative energy budgets.

The `resample()` method partitions variables into physical classes:

```python
# Resample 5-minute raw observations to 3-hourly intervals
dataset_3h = dataset.resample(freq="3h")
```

### 6.1 Conservation Formulation

* **Flux Variables (Precipitation, Solar Radiation):**
  Accumulated quantities are conserved via summation over each interval $\Delta t$:

  $$Y_{\text{flux}}[t] = \sum_{k \in \Delta t} X_{\text{flux}}[k]$$

  with `min_count=1` to prevent false zero totals when measurements are missing.

* **Thermodynamic State Variables (Temperature, Dewpoint, Pressure, Wind Speed):**
  Intensive continuous state variables are conserved via the arithmetic time mean:

  $$Y_{\text{state}}[t] = \frac{1}{|\Delta t|} \sum_{k \in \Delta t} X_{\text{state}}[k]$$

* **Quality Mask Propagation:**
  Quality validity is propagated using logical OR reduction:

  $$M_{\text{out}}[t] = \bigvee_{k \in \Delta t} M_{\text{in}}[k]$$

---

## 7. Non-Destructive Collocated Sensor Preservation

TAHMO weather stations operate redundant physical sensors for high-reliability monitoring (for example, duplicate relative humidity probes `rh_S001402` and `rh_S001403`).

* Rather than performing destructive averaging across sensors, MasharikiWeather preserves distinct sensor streams in dedicated columns named `<variable>_<sensor_id>`.
* Quality-flagged data (where quality flag $\neq 1$) are converted to `NaN` without removing the sensor stream.
* When requesting canonical parameters (such as `relative_humidity`), the dataset matches both primary and qualified sensor channels.

---

## 8. Static Topographic Priors (`load_topography`)

Complex terrain in Africa (such as the Great Rift Valley, Mount Kenya, Mount Kilimanjaro, and the Ethiopian Highlands) forces localized convection and precipitation patterns.

The `load_topography()` method loads static digital elevation models, slope, aspect, and Topographic Position Index (TPI):

```python
# Load 2D gridded topography dataset
topo_ds = dataset.load_topography(filename="africa_static_priors")
print("Topographic Fields:", list(topo_ds.data_vars))

# Merge station metadata with interpolated topographic priors
stations_with_topo = dataset.load_topography(filename="africa_static_priors", merge_stations=True)
print(stations_with_topo[["elevation", "slope", "aspect_sin", "aspect_cos", "TPI_2000M"]].head())
```

### Cyclical Aspect Encoding
Terrain aspect represents an angle in degrees $[0^\circ, 360^\circ)$. Standard linear representation creates an artificial mathematical discontinuity between $359^\circ$ and $1^\circ$ (which are physically adjacent).

`load_topography()` decomposes aspect into continuous sine and cosine components:

$$\text{aspect\_sin} = \sin\left(\theta \cdot \frac{\pi}{180}\right), \quad \text{aspect\_cos} = \cos\left(\theta \cdot \frac{\pi}{180}\right)$$

This cyclical encoding eliminates boundary jump artifacts.

---

## 9. Comprehensive Extraction Coverage for All Data Strands

MasharikiWeather interfaces with thirteen distinct ground station, reanalysis, and satellite data strands across Africa. Below are functional extraction examples for each strand.

### 9.1 TAHMO In-Situ Weather Stations
```python
from masharikiweather.dataset import extract_stations

# Extract TAHMO stations to local Parquet files
tahmo_files = extract_stations(
    source="tahmo",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["temperature", "precipitation"],
    output_dir="./mashariki_data/obs/tahmo",
)
```

### 9.2 Copernicus CDS In-Situ Global Observations
```python
# Extract CDS In-Situ ground observations
cds_files = extract_stations(
    source="cds_insitu",
    start_date="2023-01-01",
    end_date="2023-01-05",
    output_dir="./mashariki_data/obs/cds_insitu",
)
```

### 9.3 NOAA GHCN (Daily and Hourly)
```python
# Extract daily NOAA GHCN observations for Africa
ghcn_files = extract_stations(
    source="ghcn",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["t2m_max", "t2m_min", "precipitation"],
    output_dir="./mashariki_data/obs/ghcn",
)
```

### 9.4 ECMWF ERA5 Reanalysis (Legacy CDS API, CDS ARCO, and Google ARCO)
Extraction and integration of gridded ERA5 atmospheric fields across three supported backends:

* **Legacy CDS API Backend (`use_legacy=True`, default):**
  Queries the Copernicus Climate Data Store API. To bypass `cfgrib` and its C-dependencies, data can be requested in `netcdf4` format (`data_format="netcdf4"`).
  *Adaptive Chunking:* NetCDF conversions require server-side processing on CDS. Chunking starts at a maximum 5-day interval. If the CDS API returns an error indicating the request size is too large or times out, MasharikiWeather catches the error, logs a notification (e.g., `Request size too large, reducing chunk size to 4 days...`), and automatically retries with smaller chunks (4 days, then 3 days, down to 1 day).
  Official CDS setup guide: [Copernicus CDS User Guide](https://cds.climate.copernicus.eu/user-guide).

* **Copernicus CDS ARCO Backend (`use_cds_arco=True`):**
  Provides lightweight access to ECMWF analysis-ready cloud Zarr stores using `obstore` for network resilience.
  Authentication is resolved automatically from the `CDSAPI_KEY` environment variable or parsed directly from `~/.cdsapirc`.

  **Query-Aware Routing Matrix:**
  The ARCO stores vary depending on whether data is accessed by a single location over time, or a single time stamp over a spatial area. MasharikiWeather implements an automated routing mechanism that defaults to the correct URL based on the shape of the user's request (e.g., if requesting a single point or small bounding box, it routes to `geo`; if requesting a single time stamp over a wide bounding box, it routes to `time`). A manual override is available via the `chunking="geo"` or `chunking="time"` parameter.

  Exact target URLs:
  1. Single Levels (Surface - sfc):
     - Geo (Optimized for time-series at a single location): `https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr`
     - Time (Optimized for global map at a single time step): `https://arco.datastores.ecmwf.int/cadl-arco-time-002/arco/reanalysis_era5_single_levels/sfc/timeChunked.zarr`
  2. Single Levels (Waves - wav):
     - Geo (Optimized for time-series at a single location): `https://arco.datastores.ecmwf.int/cadl-arco-geo-003/arco/reanalysis_era5_single_levels/wav/geoChunked.zarr`
     - Time (Optimized for global map at a single time step): `https://arco.datastores.ecmwf.int/cadl-arco-time-003/arco/reanalysis_era5_single_levels/wav/timeChunked.zarr`
  3. Pressure Levels (pl):
     - Geo (Optimized for time-series at a single location): `https://arco.datastores.ecmwf.int/cadl-arco-geo-048/arco/reanalysis_era5_pressure_levels/pl/geoChunked.zarr`
     - Time (Optimized for global map at a single time step): `https://arco.datastores.ecmwf.int/cadl-arco-time-048/arco/reanalysis_era5_pressure_levels/pl/timeChunked.zarr`

  Robust loading is performed via `obstore` using `HTTPStore` and `ObjectStore`:
  ```python
  from obstore.store import HTTPStore
  from zarr.storage import ObjectStore
  import xarray as xr

  http_store = HTTPStore(target_url, client_options={"default_headers": {"Authorization": f"Bearer {cdsapi_key}"}})
  store = ObjectStore(http_store, read_only=True)
  ds = xr.open_zarr(store)
  ```

* **Google Cloud ARCO Backend (`use_google_arco=True`):**
  Connects to the public Google Cloud Public Dataset ERA5 ARCO Zarr store for sub-second slicing without requiring CDS credentials.

```python
from masharikiweather.dataset import MasharikiWeatherDataset, extract_gridded
from masharikiweather.extract.apis.era5 import ERA5Extractor

# 1a. Legacy CDS API extraction with NetCDF4 format and adaptive chunking
era5_legacy_path = extract_gridded(
    source="era5",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["2m_temperature", "total_precipitation", "surface_pressure"],
    output_dir="./mashariki_data/gridded/era5",
    save_zarr=True,
    use_legacy=True,
    data_format="netcdf4",  # Bypasses cfgrib C-dependencies
    chunk_days=5,           # Starts at 5 days, adaptively downsizes on limit/timeout
)

# 1b. Copernicus CDS ARCO extraction via obstore with query-aware routing
era5_cds_arco_path = extract_gridded(
    source="era5",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["2m_temperature", "total_precipitation"],
    output_dir="./mashariki_data/gridded/era5_cds_arco",
    save_zarr=True,
    use_cds_arco=True,
    chunking="geo",  # Defaults to geo for localized time-series; override with 'time' for snapshot maps
)

# 1c. Copernicus CDS ARCO wave store extraction
era5_waves_path = extract_gridded(
    source="era5",
    start_date="2023-01-01",
    end_date="2023-01-05",
    variables=["significant_height_of_combined_wind_waves_and_swell"],
    output_dir="./mashariki_data/gridded/era5_waves",
    save_zarr=True,
    use_cds_arco=True,
    stream="wav",
    chunking="geo",
)

# 1d. Google Cloud ARCO cloud extraction
era5_google_path = extract_gridded(
    source="era5",
    start_date="2023-01-01",
    end_date="2023-01-03",
    variables=["2m_temperature", "total_precipitation"],
    output_dir="./mashariki_data/gridded/era5_google",
    save_zarr=True,
    use_google_arco=True,
)

# 2. Loading gridded datasets via MasharikiWeatherDataset
dataset = MasharikiWeatherDataset(root="./mashariki_data", years=[2023])
gridded_ds_dict = dataset.load_gridded_datasets(groups=["era5"], years=[2023])
era5_ds = gridded_ds_dict["era5"]
print("Loaded ERA5 Dataset:", era5_ds)

# 3. Interpolating gridded context to station coordinates
gridded_station_df, mask_df = dataset.get_gridded_for_stations(
    groups=["era5"],
    stations=["TA00001", "TA00016", "TA00168"],
    variables=["total_precipitation", "2m_temperature"],
    method_overrides={"total_precipitation": "linear", "2m_temperature": "nearest"},
    first_date="2023-01-01",
    last_date="2023-01-07",
)
print("Station-aligned ERA5 features:")
print(gridded_station_df["era5"].head())
```

### 9.5 NASA IMERG Satellite Precipitation (GPM V07 / V08)
NASA GPM IMERG satellite precipitation estimates retrieved via non-interactive NASA Earthdata authentication.
Official authentication guide: [NASA earthaccess Documentation](https://earthaccess.readthedocs.io/en/stable/).

```python
# Extract IMERG half-hourly precipitation granules
imerg_path = extract_gridded(
    source="imerg",
    start_date="2023-01-01",
    end_date="2023-01-02",
    output_dir="./mashariki_data/gridded/imerg",
    save_zarr=True,
)
```

### 9.6 Copernicus Atmosphere Monitoring Service (CAMS)
```python
# Extract atmospheric composition (PM2.5, PM10, dust, ozone)
cams_path = extract_gridded(
    source="cams",
    start_date="2023-01-01",
    end_date="2023-01-02",
    variables=["pm2_5", "pm10", "dust", "ozone"],
    output_dir="./mashariki_data/gridded/cams",
    save_zarr=True,
)
```

### 9.7 NASA MERRA-2 Atmospheric Reanalysis
```python
# Extract MERRA-2 meteorological and aerosol reanalysis fields
merra2_path = extract_gridded(
    source="merra2",
    start_date="2023-01-01",
    end_date="2023-01-02",
    output_dir="./mashariki_data/gridded/merra2",
    save_zarr=True,
)
```

### 9.8 Open-Meteo Historical Reanalysis & NWP
```python
# Extract Open-Meteo multi-model reanalysis parameters
open_meteo_path = extract_gridded(
    source="open_meteo",
    start_date="2023-01-01",
    end_date="2023-01-02",
    variables=["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
    output_dir="./mashariki_data/gridded/open_meteo",
    save_zarr=True,
)
```

### 9.9 NASA POWER Agroclimatology & Solar
```python
# Extract NASA POWER surface solar irradiance and meteorology
power_path = extract_gridded(
    source="nasa_power",
    start_date="2023-01-01",
    end_date="2023-01-02",
    variables=["T2M", "ALLSKY_SFC_SW_DWN", "PS"],
    output_dir="./mashariki_data/gridded/nasa_power",
    save_zarr=True,
)
```

### 9.10 PVGIS Photovoltaic Solar Radiation
```python
# Extract PVGIS solar radiation time series
pvgis_path = extract_gridded(
    source="pvgis",
    start_date="2023-01-01",
    end_date="2023-01-02",
    output_dir="./mashariki_data/gridded/pvgis",
    save_zarr=True,
)
```

### 9.11 CM SAF SARAH-3 Surface Solar Radiation
```python
# Extract EUMETSAT CM SAF SARAH-3 surface solar irradiance
sarah3_path = extract_gridded(
    source="sarah3",
    start_date="2023-01-01",
    end_date="2023-01-02",
    output_dir="./mashariki_data/gridded/sarah3",
    save_zarr=True,
)
```

### 9.12 EUMETSAT LandSAF Land Surface Products
```python
# Extract LandSAF land surface temperature and radiation products
landsaf_path = extract_gridded(
    source="landsaf",
    start_date="2023-01-01",
    end_date="2023-01-02",
    output_dir="./mashariki_data/gridded/landsaf",
    save_zarr=True,
)
```

### 9.13 Sentinel-5P TROPOMI Atmospheric Chemistry
```python
# Extract TROPOMI trace gas concentrations (NO2, CO, O3)
tropomi_path = extract_gridded(
    source="tropomi",
    start_date="2023-01-01",
    end_date="2023-01-02",
    output_dir="./mashariki_data/gridded/tropomi",
    save_zarr=True,
)
```

### 9.14 Unified Master Entrypoint (`pull_weather_data`)
Extract in-situ stations and gridded reanalysis together in a single call across the continental Africa bounding box:

```python
from masharikiweather.dataset import pull_weather_data
from masharikiweather.extract.apis.base import SpatialBoundingBox

africa_box = SpatialBoundingBox(lat_min=-35.0, lat_max=38.0, lon_min=-20.0, lon_max=55.0)

result = pull_weather_data(
    start_date="2023-01-01",
    end_date="2023-01-03",
    bbox=africa_box,
    station_source="tahmo",
    gridded_source="open_meteo",
    station_variables=["temperature", "precipitation"],
    gridded_variables=["temperature_2m", "surface_pressure"],
    output_dir="./mashariki_data",
)

print("Station outputs:", result["stations"])
print("Gridded outputs:", result["gridded"])
```
