# Extraction Layer

The extraction layer is located in `masharikiweather/extract/apis/`. It downloads, processes, and formats data from external weather station networks, satellite instruments, and reanalysis models.

---

## Overview

MasharikiWeather supports extraction across three data categories:

### Weather Stations (Tabular Data)

| Source Name | Source Key | Output Format | Provider |
|---|---|---|---|
| TAHMO | `tahmo` | Parquet (`obs/tahmo/`) | TAHMO DataHub |
| Copernicus In-Situ | `cds_insitu` | Parquet (`obs/cds_insitu/`) | Copernicus Climate Data Store |
| NOAA GHCN | `ghcn` | Parquet (`obs/ghcn/`) | NOAA National Centers for Environmental Information |

### Atmospheric Reanalysis (Gridded Data)

| Source Name | Source Key | Output Format | Provider |
|---|---|---|---|
| ERA5 | `era5` | Zarr (`gridded/era5/`) | ECMWF CDS / Cloud ARCO |
| CAMS | `cams` | Zarr (`gridded/cams/`) | Copernicus Atmosphere Monitoring Service |
| MERRA-2 | `merra2` | Zarr (`gridded/merra2/`) | NASA Earthdata |
| Open-Meteo | `open_meteo` | Zarr (`gridded/open_meteo/`) | Open-Meteo API |

### Satellite Products (Gridded Data)

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

External weather APIs frequently return rate limit codes (`429 Too Many Requests`) or temporary errors (`503 Service Unavailable`). 

The extraction layer applies:
* **Exponential Retries:** Automatically retries failed requests up to 3 times with exponential backoff delays.
* **Concurrency Limits:** Restricts the number of simultaneous connections per provider (e.g., 1 connection for CDS, 2 for TAHMO, 4 for NASA Earthdata) to stay within provider limits.

---

## Storage Formats

Extracted data is stored in standard, open formats:

* **Station Data:** Written as Apache Parquet files in `{root_dir}/obs/{source}/{year}.parquet`. Collocated sensor channels are preserved as distinct columns.
* **Gridded Data:** Written as Zarr directories in `{root_dir}/gridded/{source}/{year-month}.zarr`. Arrays are cropped to the African bounding box ($\text{lat} \in [-35^\circ, 38^\circ]$, $\text{lon} \in [-20^\circ, 55^\circ]$).

---

## ERA5 Extraction Options

ERA5 data can be retrieved using three methods:

### 1. Copernicus CDS ARCO (`use_cds_arco=True`)
Directly streams Analysis-Ready Cloud-Optimized (ARCO) Zarr stores hosted on ECMWF CADL using `obstore`:
* Automatically selects `geoChunked` Zarr stores for regional time series queries to minimize network overhead.
* Automatically selects `timeChunked` Zarr stores for snapshot queries across wide spatial areas.

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
Reads from the public Google Cloud ARCO-ERA5 Zarr store.

### 3. NetCDF-4 API Extraction
Downloads files directly from the Copernicus CDS API using `data_format="netcdf4"`. This decodes via `netcdf4` and `h5netcdf` without requiring `eccodes` or `cfgrib` C-libraries.

---

## NASA IMERG Satellite Precipitation

NASA's Global Precipitation Measurement (GPM) program updated its operational stream:
* **Final Run (`GPM_3IMERGHH`):** Used for dates up to October 2025.
* **Late Run (`GPM_3IMERGHHL`):** Used for dates from November 2025 onwards.

The IMERG extractor handles this transition automatically based on the requested date range.

---

## Python Usage Example

```python
from masharikiweather.dataset import extract_gridded, pull_station_data

# 1. Download station observations for TAHMO
pull_station_data(
    source="tahmo",
    years=[2023],
    output_dir="./mashariki_data",
)

# 2. Extract gridded reanalysis for a region
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
