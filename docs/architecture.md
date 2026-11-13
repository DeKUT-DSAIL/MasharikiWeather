# MasharikiWeather System Architecture

The MasharikiWeather software framework provides a unified, machine-learning-ready meteorological data engine for Africa. Developed at the DeKUT Centre for Data Science and Artificial Intelligence (DSAIL), the architecture maintains modular separation between raw data extraction, orchestration, and user-facing dataset consumption.

---

## 1. Architectural Principles and Layer Separation

The MasharikiWeather codebase follows a two-tier architecture:

```text
       External Data Providers (BYOK)
  [TAHMO DataHub] [Copernicus CDS / ERA5] [NASA Earthdata] [CDS In-Situ]
                         |
                         v
          masharikiweather/extract/apis/ (Dagster Ingestion)
      +-------------------------------------------------------+
      | - CredentialsManager (BYOK Resolution)               |
      | - Software-Defined Assets (tahmo, era5, imerg, etc.)  |
      | - Dagster Temporal Partitions & Concurrency Control   |
      | - ParquetStationIOManager & ZarrGriddedIOManager      |
      +-------------------------------------------------------+
                         |
                         v
             Local Storage ({root_dir}/)
      +-------------------------------------------------------+
      | - obs/{source}/{partition}.parquet                    |
      | - gridded/{source}/{partition}.zarr                   |
      | - metadata/installation.parquet & parameters.parquet  |
      | - static/africa_static_priors.nc                 |
      +-------------------------------------------------------+
                         |
                         v
           masharikiweather/dataset.py (PeakWeather OOP Surface)
      +-------------------------------------------------------+
      | - MasharikiWeatherDataset(root, years, freq, ...)     |
      | - .installation & .parameters_table catalogs          |
      | - .get_observations() -> MultiIndex DataFrame & Mask  |
      | - .resample() -> Physical Conservation of Fluxes/State|
      | - .load_topography() -> DEM, Slope, Cyclical Aspect   |
      +-------------------------------------------------------+
                         |
                         v
              Visualization & Analysis
            [Matplotlib]  [Plotly]  [NumPy]
```

### 1.1 Ingestion and Orchestration Layer (`masharikiweather/extract/apis/`)
The ingestion and orchestration layer is isolated in `masharikiweather/extract/apis/` (aliased via root symlink `extract/apis/`). This subsystem is solely responsible for:
* **Credential Discovery:** Dynamic Bring-Your-Own-Key (BYOK) resolution without hardcoded tokens.
* **Network Communication:** Resilient HTTP transport, exponential backoff retries, and API-specific rate limit tagging.
* **Temporal Partitioning:** Automated windowing via Dagster partition definitions (`MonthlyPartitionsDefinition`, `DailyPartitionsDefinition`).
* **Format Serialization:** Persisting heterogeneous observation streams to partitioned Apache Parquet (`ParquetStationIOManager`) and gridded atmospheric fields to consolidated Zarr (`ZarrGriddedIOManager`).

No user-facing analytical logic, quality filtering, or spatial masking is located within this layer.

### 1.2 User-Facing Interface Layer (`masharikiweather/dataset.py`)
The user-facing Object-Oriented Programming (OOP) interface is centralized in `masharikiweather/dataset.py` around the `MasharikiWeatherDataset` class. This layer is inspired by MeteoSwiss's **PeakWeather** design and is responsible for:
* **Offline-First Loading:** Loading local partitioned Parquet files and Zarr stores without requiring network access or cloud tokens.
* **Automated Materialization:** Programmatically invoking Dagster execution when local partition files are missing on disk.
* **Metadata Catalogs:** Exposing station registries (`installation.parquet`) and variable definitions (`parameters.parquet`).
* **Multi-Station Observation Slicing:** Extracting multi-sensor time series as tabular DataFrames indexed by timestamp with a two-level column `MultiIndex` (`station`, `parameter`).
* **Physical Resampling:** Resampling observation series to target temporal resolutions while conserving fluxes (summation) and state variables (arithmetic mean).
* **Topographic Fusion:** Slicing and aligning static digital elevation models, slope, and cyclical aspect angles.

The user interface layer does not construct raw HTTP requests or maintain long-running pipeline states.

---

## 2. Directory Layout and Storage Hierarchy

MasharikiWeather stores all ingested and intermediate data on local disk under the configured root directory (defaulting to `./mashariki_data` or user-defined `root_dir`):

```text
mashariki_data/
|-- obs/                                  # Partitioned station observations
|   |-- tahmo/
|   |   |-- 2021.parquet
|   |   |-- 2022.parquet
|   |   \-- 2023.parquet
|   |-- cds_insitu/
|   |   \-- 2023.parquet
|   \-- ghcn/
|       \-- 2023.parquet
|-- gridded/                              # Partitioned multidimensional gridded datasets
|   |-- era5/
|   |   |-- 2023-01.zarr/
|   |   \-- 2023-02.zarr/
|   \-- imerg/
|       |-- 2023-01.zarr/
|       \-- 2023-02.zarr/
|-- metadata/                             # PeakWeather metadata catalogs
|   |-- installation.parquet              # Station inventory, coordinates, elevations
|   \-- parameters.parquet                # Variable definitions, units, aggregation methods
\-- static/                               # Static geophysical priors
    \-- africa_static_priors.nc      # Topography, slope, aspect, distance to water
```

### 2.1 Tabular Station Data (`obs/`)
Ground station measurements (TAHMO, CDS In-Situ, GHCNd) are stored as Apache Parquet files partitioned by year.
* **Compression:** Snappy compression is enabled by default to balance I/O throughput and disk footprint.
* **Column Multi-Sensor Preservation:** Collocated sensors at the same station (such as redundant relative humidity or temperature probes) are preserved as distinct columns (`<variable>_<sensor_id>`). Destructive averaging is avoided.

### 2.2 Gridded Spatial Fields (`gridded/`)
Continuous atmospheric reanalysis (ERA5) and satellite precipitation products (NASA IMERG) are stored in partitioned Zarr directories.
* **Consolidated Metadata:** Written with `consolidated=True` to minimize random I/O metadata lookups over network file systems and local disks.
* **Spatial Cropping:** All gridded fields are cropped to the African continental bounding box (`lat: [-35.0, 38.0]`, `lon: [-20.0, 55.0]`) prior to disk persistence.

---

## 3. Separation from Downstream Modeling

MasharikiWeather decouples data extraction, harmonization, and dataset inspection from downstream predictive modeling. 

The `MasharikiWeatherDataset` interface provides clean, standard Python structures (`pandas.DataFrame`, `xarray.Dataset`, and `numpy.ndarray`) that can be consumed directly by external machine learning frameworks (such as PyTorch Geometric, PyTorch Lightning, TensorFlow, or JAX) without embedding model training logic into the data pipeline.
