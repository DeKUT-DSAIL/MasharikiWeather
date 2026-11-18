# Step 4: System Architecture, Physical Conservation & CF Standards

MasharikiWeather is designed as a modular framework separating data ingestion and extraction from data loading, alignment, and downstream machine learning preparation.

> **Sequential Workflow Position:**
> 1. **Step 1: Credentials & Environment Setup** ([credentials.md](credentials.md))
> 2. **Step 2: Data Extraction Pipelines** ([extraction.md](extraction.md))
> 3. **Step 3: Dataset Ingestion & Data Interface** ([dataset_user_guide.md](dataset_user_guide.md))
> 4. **Step 4: System Architecture & Physical Standards** (This reference)

The architecture is tailored to the observational, meteorological, and topographic characteristics of the African continent.

---

## System Overview

```text
                                 External Data Providers
       [TAHMO DataHub]   [Copernicus CDS / CADL]   [Google Cloud ARCO]   [NASA Earthdata]   [NOAA NCEI]
              |                     |                       |                     |              |
              +---------------------+-----------------------+---------------------+--------------+
                                                |
                                                v
                                Extraction Layer (masharikiweather/extract/apis/)
       +-----------------------------------------------------------------------------------------+
       | - Automated BYOK credential discovery (.env, ~/.cdsapirc, ~/.netrc, environment vars)    |
       | - ERA5 Ingestion Priority Cascade (Google Cloud ARCO -> CDS NetCDF4 -> CDS ARCO obstore)|
       | - Bidirectional CADL ARCO / CDS GRIB variable name resolution (t2m / 2t -> 2m_temp)     |
       | - Source parameter validation and UserWarning capability checking                       |
       | - Server-respecting concurrency (TAHMO pool cap: 4 workers, connection pooling)          |
       | - Adaptive temporal chunking (5 days down to 1 day on request limits)                   |
       | - Storage serializers: Parquet (stations) and WGS 84 Zarr (gridded fields)               |
       +-----------------------------------------------------------------------------------------+
                                                |
                                                v
                                 Local Storage ({root_dir}/)
       +-----------------------------------------------------------------------------------------+
       | - obs/{source}/{year}.parquet           (Two-level MultiIndex: station_id, parameter)   |
       | - gridded/{source}/{year-month}.zarr    (WGS 84 EPSG:4326 consolidated stores)          |
       | - metadata/stations.parquet & installation.parquet (Enriched with topographic priors)   |
       | - metadata/parameters.parquet           (Source-scoped parameter inventories)           |
       | - topography/africa_topography.zarr     (17 continental multi-scale terrain channels)   |
       +-----------------------------------------------------------------------------------------+
                                                |
                                                v
                              Dataset Interface (masharikiweather/dataset.py)
       +-----------------------------------------------------------------------------------------+
       | - MasharikiWeatherDataset(root, years, freq, source_obs, extended_topo_vars, ...)        |
       | - Lazy topography streaming from Hugging Face Hub (fsspec) and local Zarr cache         |
       | - Instantiation-time topography injection onto station coordinates                      |
       | - Continuous cyclical aspect encoding (aspect_sin, aspect_cos)                          |
       | - Non-NaN station lifecycle resolution (start_date, end_date) and dynamic sensor_count  |
       | - Descriptive station name resolution from nested metadata                              |
       | - Flux-conservative temporal resampling (sum for rainfall, mean for states, max gusts)  |
       | - Spatial interpolation: get_gridded_for_stations()                                     |
       +-----------------------------------------------------------------------------------------+
                                                |
                                                v
                                 Downstream ML & Graph Learning
       +-----------------------------------------------------------------------------------------+
       | - PyTorch Geometric HeteroData graphs (station nodes, bipartite grid edges)             |
       | - Dynamic terrain-steered advection vectors from ERA5 (u, v) and elevation gradients    |
       | - Framework-agnostic exports: PyTorch, NumPy, JAX, TensorFlow                           |
       +-----------------------------------------------------------------------------------------+
```

---

## Core Architectural Pillars

### 1. Ingestion and Extraction Layer (`masharikiweather/extract/apis/`)

The extraction layer provides modular extractors for remote weather stations, satellites, and numerical reanalysis models:
* **Automated Credential Discovery:** Implements a Bring Your Own Key (BYOK) model. Credentials are automatically resolved from function arguments, shell environment variables (`CDSAPI_KEY`, `TAHMO_API_KEY`, `EARTHDATA_USERNAME`, `HF_TOKEN`), standard configuration files (`~/.cdsapirc`, `~/.netrc`), or local `.env` files without hardcoded secrets.
* **Concurrency Controls:** Ingestion rates are throttled to respect provider server limits. TAHMO ground station extraction uses configurable thread concurrency capped at a maximum of 4 workers with connection pooling (`HTTPAdapter(pool_connections=8, pool_maxsize=8)`), inter-request pacing, and exponential backoff on HTTP 429 status codes.
* **Adaptive Temporal Chunking:** For legacy Copernicus CDS API requests, queries start at 5-day intervals. If CDS returns a request-size-exceeded error, the chunk window automatically decreases (4 days, 3 days, down to 1 day) to prevent download failures.
* **Standardized Formats:** Observations are written to Parquet partitioned by calendar year, and gridded fields are written to consolidated Zarr stores in WGS 84 (EPSG:4326).

---

### 2. ERA5 Priority Cascade & CADL Variable Resolution

Reanalysis extraction from ECMWF ERA5 features an automated three-tiered backend cascade that operates dynamically when users do not provide explicit backend preferences:

```text
   extract_gridded(source="era5", ...)
                  |
        Explicit backend flag?
        /                    \
     Yes                      No (Default Priority Cascade)
      |                                    |
   Respect                     1. Check Google Cloud ARCO Availability
   User Choice                 (gs://gcp-public-data-arco-era5/...)
                               - Verify requested end_date is in range
                               - Verify requested variables exist
                                           |
                               +-----------+-----------+
                               |                       |
                            Available             Unavailable / Lagging
                               |                       |
                           Extract from           2. Fallback to CDS Legacy
                           Google ARCO            (NetCDF4 format, adaptive chunking)
                                                       |
                                           +-----------+-----------+
                                           |                       |
                                        Success                 Failure / Quota
                                           |                       |
                                        Return            3. Fallback to CDS ARCO
                                        Dataset           (obstore, CADL endpoints)
```

#### CADL ARCO vs CDS GRIB Variable Resolution
The Copernicus Analysis-Ready Data Layer (CADL) ARCO stores use internal variable identifiers that differ from legacy CDS GRIB short codes. For example, CADL stores designate 2-metre temperature as `t2m` rather than the GRIB code `2t`, and 10-metre wind components as `u10`/`v10` rather than `10u`/`10v`.

MasharikiWeather implements a bidirectional multi-alias candidate resolver:
$$\text{Candidates}(V) = \{V, \text{canonical}(V)\} \cup \{k \mid \text{Registry}[k] = \text{canonical}(V)\} \cup \{\text{short}(V)\}$$

Any candidate variable matching a dataset's active data variables is extracted and mapped to its canonical standard name, preventing variable omission across CADL Zarr stores and NetCDF4 files.

#### Source Parameter Validation
Each extractor registers its physical parameter capabilities. If a user requests a variable unsupported by a given source (e.g. requesting `temperature` from `source="imerg"`), the system emits an actionable `UserWarning` documenting the valid available variables and filters the request safely.

---

### 3. African Multi-Scale Topography Zarr Streaming

Topography plays a critical role in East African weather dynamics due to the complex terrain of the Great Rift Valley, Mount Kenya, Mount Kilimanjaro, and the Ethiopian Highlands. MasharikiWeather incorporates a 17-channel continental African topography dataset:

* **Spatial Domain:** Extends across the African continent ($\text{lat} \in [-46.97^\circ, 37.34^\circ]$, $\text{lon} \in [-25.36^\circ, 63.50^\circ]$) at high resolution.
* **17 Static Channels:**
  1. `dem`: Digital elevation model in metres above sea level.
  2. `Slope`: Local terrain slope steepness in degrees.
  3. `Aspect`: Local terrain slope aspect direction in degrees azimuth $[0^\circ, 360^\circ)$.
  4. `LandCover`: Categorical land cover classification.
  5. `DistanceToWater`: Distance to open surface water bodies in metres.
  6. `ASPECT_2000M_SIGRATIO1`, `ASPECT_10000M_SIGRATIO1`: Multi-scale terrain aspect at 2 km and 10 km filter radii.
  7. `SLOPE_2000M_SIGRATIO1`, `SLOPE_10000M_SIGRATIO1`: Multi-scale slope steepness at 2 km and 10 km radii.
  8. `SN_DERIVATIVE_2000M_SIGRATIO1`, `SN_DERIVATIVE_10000M_SIGRATIO1`: South-to-North spatial terrain gradients.
  9. `WE_DERIVATIVE_2000M_SIGRATIO1`, `WE_DERIVATIVE_10000M_SIGRATIO1`: West-to-East spatial terrain gradients.
  10. `STD_2000M`, `STD_10000M`: Terrain roughness (elevation standard deviation) at 2 km and 10 km radii.
  11. `TPI_2000M`, `TPI_10000M`: Topographic Position Index (ridge vs valley classification) at 2 km and 10 km radii.

#### Lazy Streaming & Cache Resolution
Topography is streamed lazily from the Hugging Face Hub (`DeKUT-DSAIL/weather-data` repository at `data/topography/africa_topography.zarr`) using `fsspec.filesystem("hf", token=token)` and `get_mapper()`, or read from a local Zarr/NetCDF cache in `{root_dir}/topography/`. This eliminates the need for downloading multi-gigabyte monolithic files prior to initialization.

#### Instantiation-Time Topography Injection
When `MasharikiWeatherDataset` is instantiated, `interpolate_topography()` samples the active topographic fields directly onto station coordinates:
* Continuous fields (`dem`, `Slope`, `TPI`, `STD`, directional derivatives) are interpolated via bilinear interpolation.
* Categorical fields (`LandCover`) are sampled using nearest-neighbor interpolation.
* The enriched attributes are stored directly in `stations_table` and `installation`, and persisted to `{root_dir}/metadata/stations.parquet` and `{root_dir}/metadata/installation.parquet`.

#### Continuous Cyclical Aspect Encoding
Raw aspect angles ($[0^\circ, 360^\circ)$) create an artificial discontinuity between $359^\circ$ and $1^\circ$. MasharikiWeather computes continuous sine and cosine pairs for all aspect variables:

$$\text{aspect\_sin} = \sin\left(\theta \cdot \frac{\pi}{180}\right), \quad \text{aspect\_cos} = \cos\left(\theta \cdot \frac{\pi}{180}\right)$$

Both the original angles and their $(\sin, \cos)$ representations are retained, providing physically continuous inputs for neural network models while preserving degrees for human inspection.

---

### 4. Source-Scoped Registries & Dynamic Station Lifecycles

To avoid cross-contamination of metadata across diverse networks, `MasharikiWeatherDataset` scopes its internal registries:
* **Source Scoping:** When `source_obs` is specified (e.g. `source_obs=["tahmo"]`), `stations_table` and `parameters_table` are dynamically resolved from the requested source's metadata directories (`obs/{source}/metadata/` or filtered by `network == source`).
* **Non-NaN Operational Bounds:** In-situ station lifecycles are resolved dynamically. The system inspects non-NaN observation time stamps to compute the operational start and end dates per station ($\min(t)$ and $\max(t)$), falling back to station installation records (`installationdate`, `created`) rather than emitting `NaN`.
* **Dynamic Sensor Counts:** The `sensor_count` metadata attribute is computed dynamically from the count of distinct active sensor channels and parameters associated with each station, rather than assuming a static value of 1.
* **Descriptive Station Naming:** Station name resolution prioritizes human-readable location attributes (`location.name`, `location.city`, `label`, `name`) over raw numeric identifiers, guaranteeing descriptive station labels across all exported tables.

---

## Directory Layout

Data is organized on local disk under the configured root directory (`./mashariki_data` by default):

```text
mashariki_data/
|-- obs/                                  # In-situ observations (Parquet)
|   |-- tahmo/
|   |   |-- 2021.parquet                  # Two-level MultiIndex (station_id, parameter)
|   |   |-- 2022.parquet
|   |   |-- 2023.parquet
|   |   \-- metadata/
|   |       |-- installation.parquet      # Station coordinates and names
|   |       \-- parameters.parquet        # 33-parameter catalog
|   |-- cds_insitu/
|   |   \-- 2023.parquet
|   \-- ghcn/
|       \-- 2023.parquet
|-- gridded/                              # Gridded fields (Zarr, EPSG:4326)
|   |-- era5/
|   |   |-- 2023-01.zarr/
|   |   \-- 2023-02.zarr/
|   |-- imerg/
|   |   |-- 2023-01.zarr/
|   |   \-- 2023-02.zarr/
|   \-- cams/
|       \-- 2023-01.zarr/
|-- metadata/                             # Global metadata registries
|   |-- stations.parquet                  # Station metadata with injected topography
|   |-- installation.parquet              # Canonical installation table
|   \-- parameters.parquet                # Standardized parameter definitions
\-- topography/                           # Continental topography priors
    \-- africa_topography.zarr            # 17-channel multi-scale terrain store
```

### Observation Matrices (`obs/`)
Station observations are stored in Parquet files partitioned by year. Columns are structured as a two-level pandas MultiIndex:
* Level 0: Station identifier (`station_id`, e.g., `TA00001`)
* Level 1: Parameter name (`parameter`, e.g., `2m_temperature`)

When a station has multiple collocated sensors measuring the same variable, sensor channels are preserved using explicit column suffixes (such as `relative_humidity_S001402`), preventing destructive spatial averaging of sensor readings.

### Gridded Reanalysis & Satellites (`gridded/`)
Gridded fields are stored as consolidated Zarr stores cropped to the continental African bounding box:
$$\text{Latitude} \in [-35.0^\circ, 38.0^\circ], \quad \text{Longitude} \in [-20.0^\circ, 55.0^\circ]$$

All coordinate axes are standardized to monotonically ascending orders with EPSG:4326 (WGS 84) coordinate reference system compliance.

### Enriched Metadata (`metadata/`)
Contains `stations.parquet` and `installation.parquet` enriched with interpolated multi-scale topographic priors (`dem`, `Slope`, `aspect_sin`, `aspect_cos`, `TPI_2000M`, `TPI_10000M`, `LandCover`), alongside `parameters.parquet` detailing variable units and flux-conservation rules.

### Topography Stores (`topography/`)
Houses local caches of `africa_topography.zarr` containing the 17 continental terrain descriptor variables.

---

## Downstream Machine Learning Integration

MasharikiWeather provides direct adapters for machine learning workflows:
* **PyTorch Geometric `HeteroData`:** Constructs graph datasets where weather stations and gridded reanalysis points form distinct node types. Distance-bounded directed edges ($30\text{--}50\text{ km}$) connect nearest stations and link gridded cells to observation points.
* **Topological Wind Advection:** Edge attributes include physical advection scalars computed from ERA5 horizontal wind vectors $(u, v)$ and static terrain gradients $(\nabla z)$:
  $$a_{ij} = (\mathbf{u} \cdot \mathbf{e}_{ij}) \exp\left(-\alpha |\Delta z_{ij}|\right)$$
  where $\mathbf{e}_{ij}$ is the unit directional vector from node $i$ to node $j$, and $\Delta z_{ij}$ is the elevation difference.
* **Framework-Agnostic Tensors:** Extracted features can be exported to NumPy arrays, PyTorch tensors, JAX arrays, or TensorFlow tensors for time series forecasting, spatial downscaling, or neural weather prediction models.
