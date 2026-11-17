# MasharikiWeather

MasharikiWeather is an open-source Python library and research dataset interface developed at the Centre for Data Science and Artificial Intelligence (DSAIL), Dedan Kimathi University of Technology (DeKUT). It provides standardized tools for downloading, processing, harmonizing, and loading meteorological observations, satellite products, reanalysis fields, and static topographic priors across Africa.

The framework adheres to the architectural and metadata design principles established by MeteoSwiss's PeakWeather project, adapting them to the observational and topographic realities of the African continent.

---

## Key Features

* **Multi-Source In-Situ Station Ingestion:** Connects to observational networks across Africa, including the Trans-African Hydro-Meteorological Observatory (TAHMO), Copernicus Climate Data Store (CDS) In-Situ, and NOAA Global Historical Climatology Network daily (GHCNd).
* **Complete 33-Variable TAHMO Catalog:** Standardizes the complete catalog of 33 physical, hydrological, radiation, and diagnostic variables with Climate and Forecast (CF) compliant standard names, canonical units, and aggregation rules.
* **ERA5 Ingestion Priority Cascade:** Implements an automated three-tiered backend cascade (Google Cloud ARCO -> CDS Legacy NetCDF4 -> CDS ARCO via obstore) with dynamic availability verification and explicit user backend overrides.
* **Bidirectional CADL ARCO Variable Resolution:** Maps seamlessly between Copernicus Analysis-Ready Data Layer (CADL) ARCO identifiers (such as `t2m`, `u10`, `v10`, `d2m`), legacy CDS GRIB codes (`2t`, `10u`, `10v`, `2d`), and canonical names (`2m_temperature`, `10m_u_component_of_wind`).
* **Source Parameter Validation:** Verifies requested variables against provider capability registries and issues clear `UserWarning` advisories when an unsupported variable is requested.
* **Continental Multi-Scale Topography Streaming:** Lazily streams 17 African topographic channels (`dem`, `Slope`, `Aspect`, `LandCover`, `DistanceToWater`, multi-scale `TPI`, `STD`, and directional derivatives) from Hugging Face Hub or local Zarr caches.
* **Instantiation-Time Topography Injection:** Interpolates topographic attributes onto station coordinates (bilinear for continuous fields, nearest-neighbor for categorical land cover) and persists enriched metadata to `stations.parquet` and `installation.parquet`.
* **Cyclical Aspect Angle Encoding:** Decomposes terrain aspect angles into continuous $(\sin\theta, \cos\theta)$ pairs to eliminate boundary discontinuities at $0^\circ / 360^\circ$ for machine learning models.
* **Dynamic Station Lifecycles & Scoped Registries:** Computes non-NaN operational date bounds ($\min(t)$ to $\max(t)$) and dynamic sensor counts per station, filtering metadata tables strictly to requested observation sources.
* **Flux-Conservative Temporal Resampling:** Resamples time series to arbitrary temporal frequencies while strictly conserving physical flux quantities (summing precipitation and solar energy, averaging state variables, taking maximum for wind gusts).
* **Spatiotemporal Graph Modeling:** Exports observation and reanalysis data as PyTorch Geometric `HeteroData` graphs with distance-bounded bipartite edges and terrain-steered advection vectors.

---

## Installation Requirements

### Core Dependencies

MasharikiWeather requires Python 3.9 or higher and relies on standard scientific Python libraries:

```text
numpy>=1.24.0
pandas>=2.0.0
xarray>=2023.0.0
dask>=2023.10.0
distributed>=2023.10.0
fsspec>=2024.2.0
huggingface_hub>=0.19.0
zarr>=3.0.0
netCDF4>=1.6.4
h5netcdf>=1.2.0
pyarrow>=14.0.0
fastparquet>=2023.10.0
scipy>=1.11.0
scikit-learn>=1.4.2
```

### Extraction and Cloud Access Dependencies

To retrieve data from remote APIs, cloud storage buckets, and analysis-ready cloud stores, the following packages are required:

| Package | Purpose | Target Services |
|---|---|---|
| `gcsfs` | Cloud object storage filesystem for Google Cloud Storage | Google Cloud ARCO-ERA5 Zarr store |
| `cdsapi` | Official client for ECMWF Climate Data Store API | Copernicus CDS ERA5 NetCDF4 extraction |
| `earthaccess` | Authentication and search client for NASA Earthdata | NASA IMERG and MERRA-2 extraction |
| `rioxarray` | Raster geospatial operations and CRS projections | Topography and satellite raster handling |
| `python-dotenv` | Automatic loading of environment variables from `.env` files | Credentials discovery (`HF_TOKEN`, `CDSAPI_KEY`, etc.) |
| `requests` | Vectorized HTTP requests with connection pooling | TAHMO DataHub and REST API access |
| `obstore` | Analysis-ready cloud data streaming via ObjectStore | ECMWF CADL ARCO Zarr stores |

Install core and extraction requirements via pip:

```bash
pip install numpy pandas xarray dask zarr fsspec huggingface_hub pyarrow netCDF4 h5netcdf scipy
pip install gcsfs cdsapi earthaccess rioxarray python-dotenv requests obstore
```

Alternatively, install the development dependencies directly from the repository root:

```bash
git clone https://github.com/DeKUT-DSAIL/MasharikiWeather.git
cd MasharikiWeather
pip install -e .
```

---

## Contents

```{toctree}
:maxdepth: 2

architecture
extraction
credentials
dataset_user_guide
```

---

## Citation

If you use MasharikiWeather in your academic research, please cite:

```bibtex
@misc{masharikiweather2026,
  author = {Teofilo Ligawa and Samuel Mbugua and James Muiru and Hilda Muthoni and Ciira {wa Maina}},
  title = {MasharikiWeather: A Machine-Learning-Ready Weather and Climate Dataset for Africa},
  year = {2026},
  institution = {Centre for Data Science and Artificial Intelligence (DSAIL), Dedan Kimathi University of Technology},
  publisher = {Dedan Kimathi University of Technology},
  url = {https://github.com/DeKUT-DSAIL/MasharikiWeather}
}
```
