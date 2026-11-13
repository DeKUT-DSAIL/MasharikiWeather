# Extraction Layer

The extraction layer in `masharikiweather/extract/apis/` downloads and formats data from weather stations, satellites, and reanalysis models.

---

## Overview

### Weather Stations

| Source Name | Source Key | Output Format | Provider |
|---|---|---|---|
| TAHMO | `tahmo` | Parquet (`obs/tahmo/`) | TAHMO DataHub |
| Copernicus In-Situ | `cds_insitu` | Parquet (`obs/cds_insitu/`) | Copernicus Climate Data Store |
| NOAA GHCN | `ghcn` | Parquet (`obs/ghcn/`) | NOAA NCEI |

### Reanalysis

| Source Name | Source Key | Output Format | Provider |
|---|---|---|---|
| ERA5 | `era5` | Zarr (`gridded/era5/`) | ECMWF CDS / Cloud ARCO |
| CAMS | `cams` | Zarr (`gridded/cams/`) | Copernicus Atmosphere Monitoring Service |
| MERRA-2 | `merra2` | Zarr (`gridded/merra2/`) | NASA Earthdata |
| Open-Meteo | `open_meteo` | Zarr (`gridded/open_meteo/`) | Open-Meteo API |

### Satellites

| Source Name | Source Key | Output Format | Provider |
|---|---|---|---|
| NASA IMERG | `imerg` | Zarr (`gridded/imerg/`) | NASA Earthdata (GPM) |
| LandSAF | `landsaf` | Zarr (`gridded/landsaf/`) | EUMETSAT LandSAF |
| NASA POWER | `nasa_power` | Zarr (`gridded/nasa_power/`) | NASA POWER Project |
| PVGIS | `pvgis` | Zarr (`gridded/pvgis/`) | EU PVGIS |
| CM SAF SARAH-3 | `sarah3` | Zarr (`gridded/sarah3/`) | EUMETSAT CM SAF |
| Sentinel-5P TROPOMI | `tropomi` | Zarr (`gridded/tropomi/`) | ESA / Copernicus Data Space |

---

## Retries and Rate Limits

External weather APIs can return rate limit codes (`429`) or temporary errors (`503`).

The extraction layer handles this using:
* **Retries:** Retries failed requests up to 3 times with exponential backoff delays.
* **Concurrency Limits:** Limits parallel connections per provider (1 for CDS, 2 for TAHMO, 4 for NASA Earthdata).

---

## Storage Formats

* **Station Data:** Saved as Parquet files in `{root_dir}/obs/{source}/{year}.parquet`. Sensor channels are kept as separate columns.
* **Gridded Data:** Saved as Zarr stores in `{root_dir}/gridded/{source}/{year-month}.zarr`, cropped to the African bounding box ($\text{lat} \in [-35^\circ, 38^\circ]$, $\text{lon} \in [-20^\circ, 55^\circ]$).

---

## ERA5 Options

ERA5 data can be retrieved using three backends:

### 1. Copernicus CDS ARCO (`use_cds_arco=True`)
Streams Zarr stores hosted on ECMWF CADL using `obstore`:
* Routes to `geoChunked` stores for station time series to minimize data transfer.
* Routes to `timeChunked` stores for wide map queries at single timestamps.

```python
from masharikiweather.extract.apis.era5 import ERA5Extractor

extractor = ERA5Extractor(use_cds_arco=True)
ds = extractor.extract(
    start_date="2023-01-01",
    end_date="2023-01-07",
    variables=["2m_temperature", "total_precipitation"],
)
```

### 2. Google Cloud ARCO (`use_google_arco=True`)
Reads from the Google Cloud ARCO-ERA5 Zarr store.

### 3. NetCDF-4 API Extraction
Downloads files from the Copernicus CDS API using `data_format="netcdf4"`, decoding via `netcdf4` and `h5netcdf` without requiring C-library decoders like `cfgrib`.

---

## NASA IMERG

NASA updated the GPM IMERG data stream:
* **Final Run (`GPM_3IMERGHH`):** Used for dates up to October 2025.
* **Late Run (`GPM_3IMERGHHL`):** Used for dates from November 2025 onward.

The extractor switches between these streams based on the requested date.

---

## Examples

```python
from masharikiweather.dataset import extract_gridded, pull_station_data

# Download station observations for TAHMO
pull_station_data(
    source="tahmo",
    years=[2023],
    output_dir="./mashariki_data",
)

# Extract gridded reanalysis for a region
ds_era5 = extract_gridded(
    source="era5",
    start_date="2023-01-01",
    end_date="2023-01-07",
    bbox=[-5.0, 33.0, 5.0, 42.0],  # [lat_min, lon_min, lat_max, lon_max]
    variables=["2m_temperature", "total_precipitation"],
    use_cds_arco=True,
    output_dir="./mashariki_data/gridded/era5",
)
```
