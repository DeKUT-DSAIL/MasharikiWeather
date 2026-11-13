# Architecture

MasharikiWeather organizes data processing into two main parts: data extraction from external sources into local files, and an interface for loading and preparing those files for analysis.

---

## Architecture

```text
       External Data Providers
  [TAHMO] [Copernicus CDS / ERA5] [NASA Earthdata] [CDS In-Situ]
                         |
                         v
          masharikiweather/extract/apis/ (Extraction Layer)
      +-------------------------------------------------------+
      | - Credentials resolution                              |
      | - API extractors (tahmo, era5, imerg, cams, etc.)     |
      | - Rate limiting and retry handling                    |
      | - Storage writers for Parquet and Zarr                |
      +-------------------------------------------------------+
                         |
                         v
             Local Storage ({root_dir}/)
      +-------------------------------------------------------+
      | - obs/{source}/{partition}.parquet                    |
      | - gridded/{source}/{partition}.zarr                   |
      | - metadata/installation.parquet & parameters.parquet  |
      | - static/africa_static_priors.nc                      |
      +-------------------------------------------------------+
                         |
                         v
           masharikiweather/dataset.py (Dataset Interface)
      +-------------------------------------------------------+
      | - MasharikiWeatherDataset(root, years, freq, ...)     |
      | - Station and parameter metadata tables               |
      | - get_observations() -> pandas DataFrame              |
      | - get_gridded_for_stations() -> Station interpolation |
      | - Temporal resampling with flux conservation          |
      +-------------------------------------------------------+
                         |
                         v
               Downstream Applications
             [PyTorch]  [NumPy]  [Plotly]
```

### Extraction Layer

Located in `masharikiweather/extract/apis/`. This layer handles:
* **Credentials:** Reads API keys from arguments, environment variables, or config files.
* **Network Requests:** Handles HTTP requests, retries on failures, and adheres to provider rate limits.
* **File Storage:** Saves station observations as Apache Parquet files and gridded fields as Zarr stores.

### Dataset Interface

Located in `masharikiweather/dataset.py`. The `MasharikiWeatherDataset` class handles:
* **Local Loading:** Loads Parquet and Zarr files from the local directory without requiring network access.
* **Metadata:** Provides station lists (`installation.parquet`) and parameter lists (`parameters.parquet`).
* **Observation Slicing:** Returns time series as pandas DataFrames with timestamps and station columns.
* **Resampling:** Aggregates time series to different frequencies, using sums for rainfall and means for temperatures and other state variables.
* **Topography:** Loads elevation, slope, and aspect angles for station coordinates.

---

## Directory Layout

MasharikiWeather stores downloaded and processed data on disk under the configured root directory (defaulting to `./mashariki_data`):

```text
mashariki_data/
|-- obs/                                  # Station observations (Parquet)
|   |-- tahmo/
|   |   |-- 2021.parquet
|   |   |-- 2022.parquet
|   |   \-- 2023.parquet
|   |-- cds_insitu/
|   |   \-- 2023.parquet
|   \-- ghcn/
|       \-- 2023.parquet
|-- gridded/                              # Gridded data (Zarr)
|   |-- era5/
|   |   |-- 2023-01.zarr/
|   |   \-- 2023-02.zarr/
|   \-- imerg/
|       |-- 2023-01.zarr/
|       \-- 2023-02.zarr/
|-- metadata/                             # Metadata tables
|   |-- installation.parquet              # Station coordinates and elevations
|   \-- parameters.parquet                # Variable names, units, and aggregation rules
\-- static/                               # Static spatial data
    \-- africa_static_priors.nc           # Elevation, slope, aspect, distance to water
```

### Station Data (`obs/`)
Station observations (TAHMO, CDS In-Situ, GHCNd) are saved as Parquet files partitioned by year. When multiple sensors measure the same variable at a single station, each sensor channel is kept as a separate column.

### Gridded Data (`gridded/`)
Reanalysis and satellite datasets (ERA5, IMERG, CAMS, MERRA-2) are saved in Zarr format, sliced to the African bounding box:
$$\text{Latitude} \in [-35.0^\circ, 38.0^\circ], \quad \text{Longitude} \in [-20.0^\circ, 55.0^\circ]$$

### Metadata (`metadata/`)
Contains `installation.parquet` (station names, IDs, coordinates, elevation) and `parameters.parquet` (measurement variables, units, and default aggregation methods).

### Static Priors (`static/`)
Contains digital elevation model (DEM) variables, slope, aspect angles, and distance to water bodies across the African continent.
