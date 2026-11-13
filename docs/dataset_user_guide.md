# Dataset User Guide

The core user interface of MasharikiWeather is the `MasharikiWeatherDataset` class in `masharikiweather.dataset`. It offers an offline-first interface for reading, slicing, resampling, and visualizing weather observations across Africa.

---

## 1. Getting Started

### 1.1 Local Files
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

### 1.2 Remote Fetching
If requested years or data files are not found on local disk, `MasharikiWeatherDataset` automatically fetches missing files:

```text
Fetching tahmo_station_observations: 100%|██████████| 12/12 [01:14<00:00, 2023-12-01.parquet: 61.2MB [00:06, 4.12MB/s]]
```

Remote Hugging Face streaming is opt-in via `use_hf=True` or by providing a Hugging Face `token`.

### 1.3 Parameters

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

### Metadata Tables

Metadata tables are stored in `{root}/metadata/`:
* **Station Catalog (`dataset.installation`):** Station IDs, coordinates (lat, lon), elevation, and names.
* **Parameter Catalog (`dataset.parameters_table`):** Variable names, units, and default aggregation methods.

```python
# Inspect station installation catalog
stations_df = dataset.installation
print(stations_df[["station_name", "latitude", "longitude", "elevation"]].head())

# Inspect parameter catalog
params_df = dataset.parameters_table
print(params_df[["standard_name", "units", "resampling_method"]])
```

---

## 3. Station Observations

MasharikiWeather provides methods for each observation network. Each returns a pandas DataFrame indexed by time with columns formatted as `(station_id, parameter)`.

### 3.1 TAHMO

Retrieves observations from the TAHMO network:

```python
# Retrieve temperature and precipitation for TAHMO stations
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

### 3.2 Copernicus In-Situ

Retrieves global surface station measurements from the Copernicus Climate Data Store:

```python
df_cds = dataset.get_cds_insitu_observations(
    stations=["CDS_NBO"],
    parameters=["temperature", "surface_pressure"],
    first_date="2023-01-01",
    last_date="2023-01-07",
)
print(df_cds.head())
```

### 3.3 NOAA GHCN

Retrieves measurements from NOAA's Global Historical Climatology Network:

```python
df_ghcn = dataset.get_ghcn_observations(
    stations=["KE000063612"],
    parameters=["temperature", "precipitation"],
    first_date="2023-01-01",
    last_date="2023-01-07",
)
print(df_ghcn.head())
```

### 3.4 Combined Observations

`get_observations()` returns observations from all active sources combined:

```python
df_all = dataset.get_observations(
    parameters=["temperature"],
    first_date="2023-01-01",
    last_date="2023-01-07",
)
```

### 3.5 Missing Stations
If a requested station is not present in local storage, observation methods issue a warning and continue with available stations:

```python
df = dataset.get_tahmo_observations(stations=["TA00001", "NONEXISTENT_STATION"])
# UserWarning: Requested station(s) not in dataset: ['NONEXISTENT_STATION']. Continuing with available stations.
```

### 3.6 Quality Masks and NumPy
When `return_mask=True`, a parallel boolean mask DataFrame is returned where `True` indicates a valid measured value and `False` indicates missing data.

Setting `as_numpy=True` exports 3D arrays:

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

## 4. Aligning Data Sources

All station methods share the same index (`time`) and columns formatted as `(station_id, parameter)`.

### 4.1 Aligning by Time

`MasharikiWeatherDataset.align_observations()` joins multiple observation sources onto a unified time index:

```python
strands = {
    "tahmo": dataset.get_tahmo_observations(parameters=["temperature"]),
    "ghcn": dataset.get_ghcn_observations(parameters=["temperature"]),
    "cds_insitu": dataset.get_cds_insitu_observations(parameters=["temperature"]),
}

# Align sources across their shared time range
aligned_df = MasharikiWeatherDataset.align_observations(strands, how="outer")
print("Aligned Columns (3 levels):", aligned_df.columns.names)
# ['source', 'station_id', 'parameter']
print(aligned_df.head())
```

### 4.2 Joining DataFrames

Standard pandas operations can also be used directly:

```python
# Join two DataFrames side by side
joined_df = df_tahmo.join(df_ghcn, how="outer")

# Concatenate across columns
merged_df = pd.concat({"tahmo": df_tahmo, "ghcn": df_ghcn}, axis=1)
```

### 4.3 Plotting Multiple Sources

```python
import matplotlib.pyplot as plt

df_tahmo = dataset.get_tahmo_observations(stations=["TA00001"], parameters=["temperature"])
df_ghcn = dataset.get_ghcn_observations(stations=["KE000063612"], parameters=["temperature"])

fig, ax = plt.subplots(figsize=(12, 4))
if not df_tahmo.empty:
    ax.plot(df_tahmo.index, df_tahmo["TA00001", "2m_temperature"], label="TAHMO TA00001 Temp", color="#1f77b4")
if not df_ghcn.empty:
    ax.plot(df_ghcn.index, df_ghcn["KE000063612", "2m_temperature"], label="GHCN KE000063612 Temp", color="#ff7f0e", linestyle="--")

ax.set_title("Temperature Comparison")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Temperature (°C)")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
```

---

## 5. Zarr Storage

Station observations default to Parquet files on disk. TAHMO data can also be loaded from a Zarr store when requested with `prefer_zarr=True`:

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

Search order when `prefer_zarr=True`:
1. Path passed via `zarr_path` or `tahmo_zarr_path` argument
2. `{root}/obs/tahmo.zarr`
3. `{root}/obs/tahmo.zarr.zip`
4. Remote Hugging Face store (when `use_hf=True`)

---

## 6. Temporal Resampling

When resampling time series, precipitation totals are summed and state variables (like temperature) are averaged.

The `resample()` method handles this automatically:

```python
# Resample observations to 3-hourly intervals
dataset_3h = dataset.resample(freq="3h")
```

### 6.1 Resampling Rules

* **Flux Variables (Precipitation, Solar Radiation):**
  Accumulated quantities are summed over each interval $\Delta t$:

  $$Y_{\text{flux}}[t] = \sum_{k \in \Delta t} X_{\text{flux}}[k]$$

  with `min_count=1` to prevent false zero totals when measurements are missing.

* **State Variables (Temperature, Dewpoint, Pressure, Wind Speed):**
  State variables are averaged using the arithmetic mean:

  $$Y_{\text{state}}[t] = \frac{1}{|\Delta t|} \sum_{k \in \Delta t} X_{\text{state}}[k]$$

* **Quality Masks:**
  Masks are combined using logical OR:

  $$M_{\text{out}}[t] = \bigvee_{k \in \Delta t} M_{\text{in}}[k]$$

---

## 7. Collocated Sensors

When a station has multiple sensors measuring the same variable (such as redundant relative humidity probes `rh_S001402` and `rh_S001403`):

* Each sensor stream is preserved in a dedicated column named `<variable>_<sensor_id>`.
* Flagged values (where quality flag $\neq 1$) are set to `NaN`.
* When requesting a canonical parameter (such as `relative_humidity`), both channels are matched.

---

## 8. Topography

The `load_topography()` method loads elevation, slope, aspect, and Topographic Position Index (TPI):

```python
# Load gridded topography dataset
topo_ds = dataset.load_topography(filename="africa_static_priors")
print("Topographic Fields:", list(topo_ds.data_vars))

# Merge station metadata with interpolated topographic priors
stations_with_topo = dataset.load_topography(filename="africa_static_priors", merge_stations=True)
print(stations_with_topo[["elevation", "slope", "aspect_sin", "aspect_cos", "TPI_2000M"]].head())
```

### Aspect Encoding
Terrain aspect represents an angle in degrees $[0^\circ, 360^\circ)$. Linear values create an artificial discontinuity between $359^\circ$ and $1^\circ$.

`load_topography()` decomposes aspect into sine and cosine components:

$$\text{aspect\_sin} = \sin\left(\theta \cdot \frac{\pi}{180}\right), \quad \text{aspect\_cos} = \cos\left(\theta \cdot \frac{\pi}{180}\right)$$

---

## 9. Data Extraction

Extraction examples for each supported data source:

### 9.1 TAHMO
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

### 9.2 Copernicus In-Situ
```python
# Extract CDS In-Situ ground observations
cds_files = extract_stations(
    source="cds_insitu",
    start_date="2023-01-01",
    end_date="2023-01-05",
    output_dir="./mashariki_data/obs/cds_insitu",
)
```

### 9.3 NOAA GHCN
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

### 9.4 ECMWF ERA5
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

  Loading is performed with `obstore` using `HTTPStore` and `ObjectStore`:
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

### 9.5 NASA IMERG
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

### 9.6 CAMS
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

### 9.7 NASA MERRA-2
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

### 9.8 Open-Meteo
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

### 9.9 NASA POWER
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

### 9.10 PVGIS
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

### 9.11 SARAH-3
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

### 9.12 LandSAF
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

### 9.13 TROPOMI
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

### 9.14 Combined Extraction
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
