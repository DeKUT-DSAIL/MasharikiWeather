# Step 3: Dataset Ingestion, Metadata & Data Interface

The core user interface of MasharikiWeather is the `MasharikiWeatherDataset` class in `masharikiweather.dataset`. It provides an offline-first, high-performance interface for reading, slicing, resampling, interpolating, and structuring meteorological observations, satellite products, reanalysis fields, and static topographic priors across Africa.

> **Sequential Workflow Position:**
> 1. **Step 1: Credentials & Environment Setup** ([credentials.md](credentials.md))
> 2. **Step 2: Data Extraction Pipelines** ([extraction.md](extraction.md))
> 3. **Step 3: Dataset Ingestion & Data Interface** (This guide)
> 4. **Step 4: System Architecture & Physical Standards** ([architecture.md](architecture.md))
>
> **Prerequisites:** This guide assumes that you have already configured credentials (Step 1) and extracted weather station observations and/or gridded reanalyses into your local root directory (`root="./mashariki_data"`) via Step 2 (or enabled Hugging Face streaming via `use_hf=True`).

The interface provides framework-agnostic exports for PyTorch, NumPy, and JAX.

---

## 1. Getting Started

### 1.1 Local File Initialization
By default, `MasharikiWeatherDataset` operates in an offline-first mode. When data files are present in the root directory (`root` or `root_dir`), it loads observations and metadata without network requests:

```python
from masharikiweather.dataset import MasharikiWeatherDataset

dataset = MasharikiWeatherDataset(
    root="./mashariki_data",
    years=[2022, 2023],
    freq="h",
    source_obs=["tahmo"],
    aggregation_methods={"total_precipitation": "sum", "2m_temperature": "mean"},
)
```

### 1.2 Online Streaming via Hugging Face Hub
If observations or continental topography are not present on local disk, `MasharikiWeatherDataset` can lazily stream chunked Zarr stores and Parquet files directly from the `DeKUT-DSAIL/weather-data` repository on Hugging Face Hub using `fsspec`:

```python
# Enable online streaming by passing use_hf=True or setting HF_TOKEN in the environment
dataset = MasharikiWeatherDataset(
    root="./mashariki_data",
    years=[2023],
    source_obs=["tahmo"],
    use_hf=True,
)
```

Authentication is resolved automatically from the `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` environment variables, local `.env` files, or direct constructor tokens.

### 1.3 Dataset Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `root` (or `root_dir`) | `str` or `Path` | `"./mashariki_data"` | Local directory containing `obs/`, `gridded/`, `metadata/`, and `topography/` |
| `years` | `Sequence[int]` | `[2018..2024]` | Calendar years loaded into the dataset |
| `source_obs` | `str` or `List[str]` | `["tahmo"]` | Observational source networks (`'tahmo'`, `'cds_insitu'`, `'ghcn'`) |
| `freq` | `Optional[str]` | `None` | Target temporal frequency (e.g., `'h'`, `'3h'`, `'D'`, or `None` for native sampling) |
| `parameters` | `Optional[Sequence[str]]` | `None` | Subset of physical parameters to load |
| `extended_topo_vars` | `Union[str, Sequence[str], None]` | `None` | Topographic variables to load (`'all'`, sequence of names, or `None`) |
| `aggregation_methods` | `Optional[Dict[str, str]]` | `None` | Variable-specific temporal aggregation overrides (`'sum'`, `'mean'`, `'max'`) |
| `prefer_zarr` (or `use_zarr`) | `bool` | `False` | Prioritize local or online Zarr store reading for station observations |
| `use_hf` | `bool` | `False` | Opt-in to online Hugging Face Hub streaming |
| `token` | `Optional[str]` | `None` | Optional Hugging Face access token |

---

## 2. Dataset Properties & Scoped Metadata Registries

An instantiated `MasharikiWeatherDataset` exposes properties for querying active variables, temporal ranges, and station catalogs:

```python
# Active vs available parameters
print("Available Parameters:", dataset.available_parameters)
print("Active Parameters:", dataset.parameters)

# Active vs available years
print("Available Years:", dataset.available_years)

# Active stations
print(f"Active Station Count: {len(dataset.stations)}")
print("Sample Stations:", dataset.stations[:5])
```

### Source-Scoped Metadata Registries

Station and parameter tables are scoped to the observational sources configured in `source_obs`:

```python
# Inspect station installation catalog (scoped to requested source_obs)
stations_df = dataset.installation
print(stations_df[["station_name", "latitude", "longitude", "elevation", "start_date", "end_date", "sensor_count"]].head())

# Inspect parameter catalog (scoped to requested source_obs)
params_df = dataset.parameters_table
print(params_df[["canonical_name", "standard_name", "units", "resampling_method"]].head())
```

### Dynamic Station Lifecycles & Sensor Counts

To ensure high data quality and avoid invalid `NaN` bounds in metadata:
* **Non-NaN Operational Bounds:** The system inspects valid non-null observation timestamps for each station to determine the actual operational window ($\min(t)$ and $\max(t)$). If observations are not yet loaded, it falls back to station installation records (`installationdate`, `created`).
* **Dynamic Sensor Counts:** The `sensor_count` column reflects the true count of distinct active sensor channels and measurement parameters associated with each station, rather than assuming a static value of 1.
* **Descriptive Station Naming:** Station name resolution prioritizes human-readable location attributes (`location.name`, `location.city`, `label`, `name`) over raw numeric identifiers, ensuring meaningful names across all metadata tables.

---

## 3. Complete 33-Variable TAHMO Parameter Catalog

MasharikiWeather supports the complete 33-parameter catalog for the TAHMO network. All variables are mapped to CF-compliant standard names with physical units and default aggregation methods:

| Parameter Name | Canonical Name | Code | CF Standard Name | Units | Physical Type | Aggregation |
|---|---|---|---|---|---|---|
| Atmospheric Pressure | `surface_pressure` | `ap` | `surface_air_pressure` | `hPa` | PHYSICAL | `mean` |
| Convective Precipitation | `convective_precipitation` | `cp` | `convective_precipitation_amount` | `mm` | PRECIPITATION | `sum` |
| Water Depth | `water_depth` | `dw` | `depth_to_water_table` | `mm` | HYDROLOGICAL | `mean` |
| Electrical Conductivity | `electrical_conductivity` | `ec` | `electrical_conductivity` | `mS cm-1` | ELECTRICAL | `mean` |
| Pore Water EC | `pore_water_electrical_conductivity` | `ew` | `electrical_conductivity_of_water` | `mS cm-1` | ELECTRICAL | `mean` |
| Humidity Sensor Temp | `humidity_sensor_temperature` | `ht` | `temperature` | `degC` | DIAGNOSTIC | `mean` |
| Logger Battery Voltage | `logger_battery_voltage` | `lb` | `battery_voltage` | `V` | DIAGNOSTIC | `mean` |
| Lightning Distance | `lightning_distance` | `ld` | `lightning_strike_distance` | `km` | PHYSICAL | `mean` |
| Lightning Strike Count | `lightning_strike_count` | `le` | `number_of_lightning_flashes` | `counts` | DIAGNOSTIC | `sum` |
| Logger Pressure | `logger_pressure` | `lp` | `air_pressure` | `hPa` | DIAGNOSTIC | `mean` |
| Logger Temperature | `logger_temperature` | `lt` | `temperature` | `degC` | DIAGNOSTIC | `mean` |
| Matric Potential | `matric_potential` | `mp` | `soil_matric_potential` | `kPa` | HYDROLOGICAL | `mean` |
| Precipitation Drop Count | `precipitation_drop_count` | `pd` | `number_of_hydrometeors` | `counts` | PRECIPITATION | `sum` |
| Total Precipitation | `total_precipitation` | `pr` | `precipitation_amount` | `mm` | PRECIPITATION | `sum` |
| Tipping Bucket Tips | `precipitation_tip_count` | `pt` | `number_of_tipping_bucket_tips` | `counts` | PRECIPITATION | `sum` |
| Pore Water Permittivity | `permittivity_pore_water` | `pw` | `relative_permittivity` | `1` | ELECTRICAL | `mean` |
| Relative Humidity | `relative_humidity` | `rh` | `relative_humidity` | `%` | PHYSICAL | `mean` |
| Rain Rate | `rain_rate` | `rn` | `rainfall_rate` | `mm h-1` | PRECIPITATION | `sum` |
| Sunshine Duration | `sunshine_duration` | `sd` | `duration_of_sunshine` | `s` | RADIATION | `sum` |
| Soil Moisture | `soil_moisture` | `sm` | `volume_fraction_of_water_in_soil` | `m3 m-3` | HYDROLOGICAL | `mean` |
| Solar Radiation Downwards | `surface_solar_radiation_downwards` | `sr` | `surface_downwelling_shortwave_flux_in_air` | `W m-2` | RADIATION | `mean` |
| Canopy Temperature | `canopy_temperature` | `tc` | `canopy_temperature` | `degC` | PHYSICAL | `mean` |
| 2m Dewpoint Temperature | `2m_dewpoint_temperature` | `td` | `dew_point_temperature` | `degC` | PHYSICAL | `mean` |
| 2m Temperature | `2m_temperature` | `te` | `air_temperature` | `degC` | PHYSICAL | `mean` |
| Hydrometeor Temp | `precipitation_temperature` | `tp` | `temperature` | `degC` | PHYSICAL | `mean` |
| Ultraviolet Index | `uv_index` | `uv` | `ultraviolet_index` | `1` | RADIATION | `mean` |
| Vapor Pressure | `vapor_pressure` | `vp` | `water_vapor_pressure` | `hPa` | PHYSICAL | `mean` |
| Wind Direction | `wind_from_direction` | `wd` | `wind_from_direction` | `degrees` | PHYSICAL | `mean` |
| Wind Gust Speed | `wind_speed_of_gust` | `wg` | `wind_speed_of_gust` | `m s-1` | PHYSICAL | `max` |
| Water Level | `water_level` | `wl` | `water_surface_height_above_reference_datum` | `m` | HYDROLOGICAL | `mean` |
| Water Discharge | `water_discharge` | `wq` | `water_volume_transport_in_river_channel` | `m3 s-1` | HYDROLOGICAL | `mean` |
| Wind Speed | `wind_speed` | `ws` | `wind_speed` | `m s-1` | PHYSICAL | `mean` |
| Water Velocity | `water_velocity` | `wv` | `water_velocity` | `m s-1` | HYDROLOGICAL | `mean` |

---

## 4. Multi-Scale Topography Streaming & Persistence

Topography strongly modulates convective precipitation, diurnal thermal cycles, and wind flow regimes across East Africa. MasharikiWeather integrates a 17-channel continental African topography dataset:

### 4.1 Topography Channels

| Channel Identifier | Physical Description | Units / Scale |
|---|---|---|
| `dem` | Surface digital elevation model | Metres above sea level |
| `Slope` | Terrain slope steepness | Degrees |
| `Aspect` | Terrain slope orientation | Degrees azimuth $[0^\circ, 360^\circ)$ |
| `LandCover` | Land cover classification category | Categorical integer code |
| `DistanceToWater` | Proximity to open surface water bodies | Metres |
| `ASPECT_2000M_SIGRATIO1` | Terrain aspect at 2 km spatial filter radius | Degrees azimuth |
| `ASPECT_10000M_SIGRATIO1` | Terrain aspect at 10 km spatial filter radius | Degrees azimuth |
| `SLOPE_2000M_SIGRATIO1` | Terrain slope at 2 km spatial filter radius | Degrees |
| `SLOPE_10000M_SIGRATIO1` | Terrain slope at 10 km spatial filter radius | Degrees |
| `SN_DERIVATIVE_2000M_SIGRATIO1` | South-North spatial terrain gradient at 2 km filter | Dimensionless |
| `SN_DERIVATIVE_10000M_SIGRATIO1` | South-North spatial terrain gradient at 10 km filter | Dimensionless |
| `WE_DERIVATIVE_2000M_SIGRATIO1` | West-East spatial terrain gradient at 2 km filter | Dimensionless |
| `WE_DERIVATIVE_10000M_SIGRATIO1` | West-East spatial terrain gradient at 10 km filter | Dimensionless |
| `STD_2000M` | Elevation standard deviation (roughness) at 2 km radius | Metres |
| `STD_10000M` | Elevation standard deviation (roughness) at 10 km radius | Metres |
| `TPI_2000M` | Topographic Position Index (ridge/valley) at 2 km radius | Metres |
| `TPI_10000M` | Topographic Position Index (ridge/valley) at 10 km radius | Metres |

### 4.2 Object-Oriented Interface

```python
# Query available static topographic variables
print("Available Topography Variables:", dataset.available_topography)

# Access the gridded 2D xarray.Dataset
topo_ds = dataset.topography
print("Gridded Topography Dataset:", topo_ds)

# Retrieve a cropped bounding box slice
east_africa_topo = dataset.get_topography_data(
    variables=["dem", "Slope", "TPI_2000M", "TPI_10000M"],
    bbox=[33.0, -5.0, 42.0, 5.0],  # [min_lon, min_lat, max_lon, max_lat]
)
print("Cropped Topography Slice:", east_africa_topo)
```

### 4.3 Instantiation-Time Topography Injection onto Stations

During dataset initialization, `interpolate_topography()` samples the continental terrain fields directly onto station coordinates:
* Continuous fields (`dem`, `Slope`, `TPI`, `STD`, directional derivatives) are interpolated via bilinear interpolation.
* Categorical fields (`LandCover`) are sampled using nearest-neighbor interpolation.
* Aspect angles are decomposed into sine and cosine pairs:
  $$\text{aspect\_sin} = \sin\left(\theta \cdot \frac{\pi}{180}\right), \quad \text{aspect\_cos} = \cos\left(\theta \cdot \frac{\pi}{180}\right)$$
* These attributes are stored directly in `stations_table` and `installation`, and saved to `{root_dir}/metadata/stations.parquet` and `{root_dir}/metadata/installation.parquet`.

```python
# Inspect station installation catalog with injected topographic priors
stations_df = dataset.installation
print(stations_df[["dem", "Slope", "aspect_sin", "aspect_cos", "TPI_2000M", "TPI_10000M"]].head())
```

---

## 5. Station Observations

Observation methods return pandas DataFrames indexed by `time` with columns formatted as a two-level MultiIndex `(station_id, parameter)`.

### 5.1 TAHMO Network

```python
# Retrieve temperature and precipitation for specific TAHMO stations
df_tahmo, mask_tahmo = dataset.get_tahmo_observations(
    stations=["TA00001", "TA00016"],
    parameters=["2m_temperature", "total_precipitation"],
    first_date="2023-01-01 00:00",
    last_date="2023-01-07 23:00",
    return_mask=True,
)

print(df_tahmo.head())
```

Output DataFrame structure:

```text
                                  TA00001                                  TA00016             
                       2m_temperature total_precipitation       2m_temperature total_precipitation  
time                                                                                       
2023-01-01 00:00:00              21.4                 0.0                 20.1                 0.0
2023-01-01 01:00:00              20.8                 0.2                 19.7                 0.0
```

### 5.2 Copernicus In-Situ Network

```python
df_cds = dataset.get_cds_insitu_observations(
    stations=["CDS_NBO"],
    parameters=["2m_temperature", "surface_pressure"],
    first_date="2023-01-01",
    last_date="2023-01-07",
)
print(df_cds.head())
```

### 5.3 NOAA GHCNd Network

```python
df_ghcn = dataset.get_ghcn_observations(
    stations=["KE000063612"],
    parameters=["2m_temperature", "total_precipitation"],
    first_date="2023-01-01",
    last_date="2023-01-07",
)
print(df_ghcn.head())
```

### 5.4 Combined Observations & Multi-Source Alignment

`get_observations()` extracts data across all active networks:

```python
df_all = dataset.get_observations(
    parameters=["2m_temperature", "total_precipitation"],
    first_date="2023-01-01",
    last_date="2023-01-07",
)
```

`MasharikiWeatherDataset.align_observations()` combines disparate source dataframes onto a unified three-level MultiIndex `(source, station_id, parameter)`:

```python
strands = {
    "tahmo": dataset.get_tahmo_observations(parameters=["2m_temperature"]),
    "ghcn": dataset.get_ghcn_observations(parameters=["2m_temperature"]),
}

aligned_df = MasharikiWeatherDataset.align_observations(strands, how="outer")
print("Aligned MultiIndex Columns:", aligned_df.columns.names)
```

### 5.5 Collocated Sensor Preservation

When multiple sensors measure the same physical parameter at a single station (e.g. redundant relative humidity probes `rh_S001402` and `rh_S001403`):
* Each sensor stream is preserved in an independent column named `<variable>_<sensor_id>`.
* Quality-flagged values (flag $\neq 1$) are masked to `NaN`.
* Requesting canonical `relative_humidity` extracts and aligns both channels without destructive averaging.

### 5.6 NumPy Tensor Export

Setting `as_numpy=True` exports observations as 3D NumPy arrays:

```python
data_tensor, mask_tensor = dataset.get_tahmo_observations(
    stations=["TA00001", "TA00016"],
    parameters=["2m_temperature", "total_precipitation"],
    as_numpy=True,
    return_mask=True,
)

# Output tensor shape: (T_time_steps, N_stations, C_parameters)
print("Data Tensor Shape:", data_tensor.shape)
print("Mask Tensor Shape:", mask_tensor.shape)
```

---

## 6. Flux-Conservative Temporal Resampling

When converting observation time series between temporal scales (e.g., native 5-minute sampling to 1-hour or 3-hour cadences), MasharikiWeather only respects physical conservation laws:

```python
# Resample entire dataset to 3-hourly intervals
dataset_3h = dataset.resample(freq="3h")
```

### Resampling Rules

* **Flux Quantities (Precipitation, Solar Radiation):**
  Accumulated fluxes are summed over each interval $\Delta t$:
  $$Y_{\text{flux}}[t] = \sum_{k \in \Delta t} X_{\text{flux}}[k]$$
  with `min_count=1` to prevent false zero totals when measurements are missing.
* **State Variables (Temperature, Pressure, Relative Humidity, Wind Speed):**
  State variables are averaged using arithmetic means:
  $$Y_{\text{state}}[t] = \frac{1}{|\Delta t|} \sum_{k \in \Delta t} X_{\text{state}}[k]$$
* **Extreme Gust Variables (Wind Speed of Gust):**
  Peak values take the maximum over the interval:
  $$Y_{\text{gust}}[t] = \max_{k \in \Delta t} X_{\text{gust}}[k]$$
* **Quality Masks:**
  Quality masks are combined using logical OR:
  $$M_{\text{out}}[t] = \bigvee_{k \in \Delta t} M_{\text{in}}[k]$$

---

## 7. Gridded Reanalysis Fusion

### 7.1 Loading Gridded Fields

Gridded reanalysis fields (such as ERA5) can be loaded into memory or lazily accessed as `xarray.Dataset` objects:

```python
# Load gridded datasets for specified years
gridded_dict = dataset.load_gridded_datasets(groups=["era5"], years=[2023])
era5_ds = gridded_dict["era5"]
print("Loaded ERA5 Fields:", list(era5_ds.data_vars))
```

### 7.2 Station-Level Gridded Extraction

`get_gridded_for_stations()` interpolates gridded reanalysis fields directly onto station coordinates:

```python
gridded_station_df, mask_df = dataset.get_gridded_for_stations(
    groups=["era5"],
    stations=["TA00001", "TA00016"],
    variables=["total_precipitation", "2m_temperature"],
    method_overrides={"total_precipitation": "linear", "2m_temperature": "nearest"},
    first_date="2023-01-01",
    last_date="2023-01-07",
)

print("Station-Aligned ERA5 Features:")
print(gridded_station_df["era5"].head())
```
