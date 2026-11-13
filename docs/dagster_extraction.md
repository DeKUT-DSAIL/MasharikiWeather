# Dagster Software-Defined Extraction Layer

The data extraction and orchestration subsystem of MasharikiWeather is located in `masharikiweather.extract.apis`. It utilizes [Dagster](https://dagster.io/) to implement declarative Software-Defined Assets (SDAs), native temporal partitioning, automated retry backoff, concurrency rate-limiting, and decoupled I/O management.

---

## 1. Overview of Software-Defined Extraction Assets

Rather than running unstructured extraction scripts, MasharikiWeather models each external meteorological data stream as a discrete Software-Defined Asset:

```text
      Software-Defined Assets (dagster_assets.py)
   +--------------------------------------------------------------+
   | - tahmo_station_observations    (Monthly Partitions)        |
   | - era5_gridded_reanalysis       (Monthly Partitions)        |
   | - imerg_satellite_precipitation (Monthly Partitions)        |
   | - cds_insitu_observations       (Monthly Partitions)        |
   +--------------------------------------------------------------+
                                |
             Governed by Concurrency & Retry Policies
   +--------------------------------------------------------------+
   | - RetryPolicy(max_retries=3, delay=5, backoff=EXPONENTIAL)   |
   | - Op Tags: Copernicus CDS=1, TAHMO=2, NASA Earthdata=4       |
   +--------------------------------------------------------------+
                                |
                Decoupled Dagster IO Managers (dagster_io.py)
        +-------------------------------+------------------------------+
        |                                                              |
        v                                                              v
   ParquetStationIOManager                                   ZarrGriddedIOManager
   (obs/{asset}/{partition}.parquet)                         (gridded/{asset}/{partition}.zarr)
        |                                                              |
        v                                                              v
   - Tabular Station Observations                            - Gridded Atmospheric/Satellite Fields
   - installation.parquet & parameters.parquet catalogs      - Consolidated Zarr metadata
```

### Asset Specifications

| Asset Name | Output Type | Storage Engine | Default Partitioning | Upstream Provider |
|---|---|---|---|---|
| `tahmo_station_observations` | `pd.DataFrame` | `ParquetStationIOManager` | Monthly (`2014-01-01` to current) | TAHMO DataHub API |
| `era5_gridded_reanalysis` | `xr.Dataset` | `ZarrGriddedIOManager` | Monthly (`1979-01-01` to current) | Google Cloud ARCO-ERA5 / ECMWF CDS |
| `imerg_satellite_precipitation` | `xr.Dataset` | `ZarrGriddedIOManager` | Monthly (`2000-06-01` to current) | NASA Earthdata (GPM IMERG) |
| `cds_insitu_observations` | `pd.DataFrame` | `ParquetStationIOManager` | Monthly (`1979-01-01` to current) | Copernicus In-Situ Global Service |

---

## 2. Native Temporal Partitioning

Manual `for` loops across date ranges in API scripts are fragile; network dropouts, API rate-limiting, and memory exhaustion can corrupt partial runs. MasharikiWeather eliminates manual loops by defining native Dagster temporal partitions:

```python
import dagster as dg

tahmo_partitions = dg.MonthlyPartitionsDefinition(start_date="2014-01-01")
era5_partitions = dg.MonthlyPartitionsDefinition(start_date="1979-01-01")
imerg_partitions = dg.MonthlyPartitionsDefinition(start_date="2000-06-01")
cds_insitu_partitions = dg.MonthlyPartitionsDefinition(start_date="1979-01-01")
```

Inside each asset function, execution dates are retrieved directly from the context:

```python
@dg.asset(
    partitions_def=tahmo_partitions,
    retry_policy=STANDARD_RETRY_POLICY,
    op_tags={"dagster/concurrency_key": "tahmo_datahub_api", "max_concurrency": "2"},
    io_manager_key="parquet_io",
)
def tahmo_station_observations(context) -> pd.DataFrame:
    time_window = context.partition_time_window
    start_dt = time_window.start
    end_dt = time_window.end
    # Extraction proceeds for [start_dt, end_dt] without nested date loops
```

This ensures idempotency: each partition executes independently and can be retried or backfilled without affecting adjacent months.

---

## 3. Resilience and Concurrency Control

### 3.1 Exponential Retry Backoff Policy
Meteorological APIs frequently return transient `429 Too Many Requests` or `503 Service Unavailable` errors during heavy network traffic. All MasharikiWeather assets share a standardized retry policy:

```python
STANDARD_RETRY_POLICY = dg.RetryPolicy(
    max_retries=3,
    delay=5,
    backoff=dg.Backoff.EXPONENTIAL,
    jitter=dg.Jitter.PLUS_MINUS,
)
```

* **Maximum Retries:** 3 attempts per partition.
* **Initial Delay:** 5 seconds.
* **Backoff Strategy:** Exponential growth with random jitter to prevent thundering herd problems against upstream servers.

### 3.2 Operation Concurrency Limits
To prevent IP bans and respect provider rate limits, operation tags enforce concurrency limits across asset execution:

```python
# Copernicus CDS (strict single-connection queuing)
op_tags={"dagster/concurrency_key": "copernicus_cds_api", "max_concurrency": "1"}

# TAHMO DataHub (dual parallel requests)
op_tags={"dagster/concurrency_key": "tahmo_datahub_api", "max_concurrency": "2"}

# NASA Earthdata / Earth Access (four concurrent downloads)
op_tags={"dagster/concurrency_key": "nasa_earthaccess_api", "max_concurrency": "4"}
```

---

## 4. Decoupled Dagster IO Managers

Asset logic is completely decoupled from disk persistence through `ConfigurableIOManager` subclasses defined in `masharikiweather.extract.apis.dagster_io`:

### 4.1 `ParquetStationIOManager`
Dedicated to tabular ground station observations (`pd.DataFrame`).
* **Path Pattern:** `{root_dir}/obs/{asset_name}/{partition}.parquet`
* **Format:** Apache Parquet with PyArrow engine and Snappy compression.
* **Metadata Emitted:** Persisted path, row count, column list, total size in bytes, and size in megabytes.

```python
from masharikiweather.extract.apis.dagster_io import ParquetStationIOManager

io_manager = ParquetStationIOManager(root_dir="./mashariki_data")
```

### 4.2 `ZarrGriddedIOManager`
Dedicated to multidimensional continuous atmospheric and satellite fields (`xr.Dataset`).
* **Path Pattern:** `{root_dir}/gridded/{asset_name}/{partition}.zarr`
* **Format:** Zarr directory store with consolidated metadata (`consolidated=True`). Variable encodings are sanitized prior to write to avoid compression conflicts.
* **Metadata Emitted:** Persisted path, dimension sizes dictionary, variable catalog, and count.

```python
from masharikiweather.extract.apis.dagster_io import ZarrGriddedIOManager

io_manager = ZarrGriddedIOManager(root_dir="./mashariki_data")
```

---

## 5. Programmatic Pipeline Execution with Live Progress

MasharikiWeather enables automated materialization directly from Python without launching the Dagster webserver UI (`dagit`). The runner helper `materialize_extraction` coordinates partitioned extraction and outputs live transfer metrics:

```python
from masharikiweather.extract.apis.dagster_assets import tahmo_station_observations
from masharikiweather.extract.apis.dagster_pipeline import materialize_extraction

results = materialize_extraction(
    assets=[tahmo_station_observations],
    partition_keys=["2023-01-01", "2023-02-01", "2023-03-01"],
    root_dir="./mashariki_data",
    show_progress=True,
)
```

Terminal output displays progress bars with file sizes, elapsed durations, and transfer throughput:

```text
Materializing tahmo_station_observations: 100%|██████████| 3/3 [00:42<00:00, 2023-03-01.parquet: 56.9MB [00:14, 3.93MB/s]]
```

---

## 6. C-Dependency Handling and NetCDF-4 / ARCO Alternatives

Ingesting GRIB-formatted files from ECMWF CDS requires low-level C-bindings (`eccodes` and `cfgrib`). In standard Python environments without OS-level shared libraries, importing `cfgrib` fails with an `ImportError`.

### Bypassing cfgrib with NetCDF-4 Format
MasharikiWeather supports retrieval in NetCDF-4 format (`data_format="netcdf4"`), decoded via standard `netcdf4` and `h5netcdf` Python libraries without requiring `cfgrib` or `eccodes`. Adaptive temporal chunking automatically handles CDS server-side processing limits.

### Cloud-Native ARCO Backends
Alternatively, users can extract ERA5 fields directly from analysis-ready cloud stores:
* **Copernicus CDS ARCO (`use_cds_arco=True`):** Analysis-ready Zarr stores accessed via `obstore`. Refer to the [Copernicus CDS User Guide](https://cds.climate.copernicus.eu/user-guide).
* **Google Cloud ARCO (`use_google_arco=True`):** Public Google Cloud ERA5 Zarr store.

### Docker Guidance for GRIB Decoding
If GRIB files are requested, build and run your environment using the MasharikiWeather Dockerfile:

```bash
docker build -t masharikiweather .
```

which includes precompiled ecCodes C-libraries. MasharikiWeather avoids directing users to install Conda or `conda-forge`. All system-level C-dependencies are verified through standard containerization via Docker.

---

## 7. NASA IMERG V07 to V08 Operational Transition

NASA's Global Precipitation Measurement (GPM) program updated its operational processing pipeline in late 2025:
* **Final Run (`GPM_3IMERGHH`):** Concluded operation around October 2025.
* **Late Run (`GPM_3IMERGHHL`):** Continues operations from November 2025 forward.

The `imerg_satellite_precipitation` asset monitors partition dates automatically:
* Partitions prior to `2025-11-01` query `GPM_3IMERGHH`.
* Partitions on or after `2025-11-01` query `GPM_3IMERGHHL`.
* When the transition boundary is crossed, the asset logs an advisory notification and continues processing:

```text
INFO: NASA IMERG V07 Final Run concluded around 2025-10.
Transitioning automatically to Late Run (GPM_3IMERGHHL) for partition 2025-11-01.
```

### Spatial Cropping Mandate
Unprocessed global IMERG granules contain $1800 \times 3600$ half-hourly grid cells. To prevent storage bloat and unnecessary I/O, the extractor applies bounding box spatial slicing immediately upon granule decode:

$$\text{Latitude} \in [-35.0^\circ, 38.0^\circ], \quad \text{Longitude} \in [-20.0^\circ, 55.0^\circ]$$

Cropped arrays are persisted directly to partition Zarr stores.
