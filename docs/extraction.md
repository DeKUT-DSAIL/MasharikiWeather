# Extraction Layer

The extraction layer in `masharikiweather/extract/apis/` provides modular, robust data pipelines for downloading, validating, and formatting meteorological data from ground weather stations, satellite remote sensing instruments, and numerical reanalysis models across Africa.

---

## Data Sources Overview

### In-Situ Weather Stations

| Source Name | Source Key | Output Format | Default Storage Location | Provider / Service |
|---|---|---|---|---|
| TAHMO | `tahmo` | Parquet | `{root_dir}/obs/tahmo/` | Trans-African Hydro-Meteorological Observatory |
| Copernicus In-Situ | `cds_insitu` | Parquet | `{root_dir}/obs/cds_insitu/` | Copernicus Climate Data Store |
| NOAA GHCNd | `ghcn` | Parquet | `{root_dir}/obs/ghcn/` | NOAA National Centers for Environmental Information |

### Atmospheric Reanalysis

| Source Name | Source Key | Output Format | Default Storage Location | Provider / Service |
|---|---|---|---|---|
| ECMWF ERA5 | `era5` | Zarr / NetCDF4 | `{root_dir}/gridded/era5/` | Google Cloud ARCO / ECMWF CDS / CADL |
| CAMS Reanalysis | `cams` | Zarr | `{root_dir}/gridded/cams/` | Copernicus Atmosphere Monitoring Service |
| NASA MERRA-2 | `merra2` | Zarr | `{root_dir}/gridded/merra2/` | NASA Goddard Earth Sciences (GES DISC) |
| Open-Meteo | `open_meteo` | Zarr | `{root_dir}/gridded/open_meteo/` | Open-Meteo Historical Reanalysis API |

### Satellite Remote Sensing

| Source Name | Source Key | Output Format | Default Storage Location | Provider / Instrument |
|---|---|---|---|---|
| NASA IMERG | `imerg` | Zarr | `{root_dir}/gridded/imerg/` | NASA Global Precipitation Measurement (GPM) |
| EUMETSAT LandSAF | `landsaf` | Zarr | `{root_dir}/gridded/landsaf/` | EUMETSAT Satellite Application Facility |
| NASA POWER | `nasa_power` | Zarr | `{root_dir}/gridded/nasa_power/` | NASA Prediction of Worldwide Energy Resources |
| EU PVGIS | `pvgis` | Zarr | `{root_dir}/gridded/pvgis/` | European Commission Joint Research Centre |
| CM SAF SARAH-3 | `sarah3` | Zarr | `{root_dir}/gridded/sarah3/` | EUMETSAT Climate Monitoring SAF |
| Sentinel-5P TROPOMI | `tropomi` | Zarr | `{root_dir}/gridded/tropomi/` | ESA / Copernicus Sentinel-5 Precursor |

---

## Server Concurrency and Rate Limiting

External meteorological APIs impose strict rate limits and connection caps. The extraction layer handles these transparently:

* **Configurable TAHMO Concurrency (DEC-020):**
  TAHMO DataHub legacy endpoints are protected by bounded thread concurrency. The default extraction concurrency is `max_workers=4`, backed by an HTTP connection pool (`HTTPAdapter(pool_connections=8, pool_maxsize=8)`), thread-local sessions, inter-request pacing delays, and exponential backoff on HTTP 429 status codes. Concurrency can be tuned via `max_workers` in `TAHMOExtractor`, `pull_station_data()`, or `pull_weather_data()`, but is capped at 4 to prevent server saturation.
* **Copernicus CDS Concurrency & Adaptive Chunking:**
  ECMWF CDS limits simultaneous user requests. In NetCDF4 download mode (`data_format="netcdf4"`), queries start at a maximum 5-day interval. If the server rejects the query due to size limits or request timeouts, MasharikiWeather catches the error, logs a notice, and automatically downsizes the chunk size (4 days, 3 days, down to 1 day).
* **NASA Earthdata Non-Interactive Authentication:**
  IMERG and MERRA-2 extractors utilize `earthaccess` to authenticate automatically from `~/.netrc` or environment variables (`EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`), with automated token refresh.

---

## ERA5 Ingestion Priority Cascade (DEC-023)

ECMWF ERA5 provides hourly global atmospheric reanalysis. Because data providers vary significantly in latency, server quotas, and data format, MasharikiWeather implements an automated three-tiered priority cascade:

```text
               extract_gridded(source="era5", ...)
                              |
               User Specified Explicit Backend?
               /                              \
            Yes                                No (Default Priority Cascade)
             |                                              |
     Respect Explicit Flag                     1. Check Google Cloud ARCO
    (use_google_arco=True,                     (gs://gcp-public-data-arco-era5/...)
     use_legacy=True,                          - Verify end_date <= data horizon
     use_cds_arco=True)                        - Verify requested variables exist
                                                            |
                                              +-------------+-------------+
                                              |                           |
                                           Available                 Unavailable / Lagging
                                              |                           |
                                      Extract from               2. Fallback to CDS Legacy
                                      Google ARCO                (NetCDF4, adaptive chunking)
                                                                          |
                                                          +---------------+---------------+
                                                          |                               |
                                                       Success                         Failure / Quota
                                                          |                               |
                                                       Return                     3. Fallback to CDS ARCO
                                                       Dataset                    (obstore, CADL endpoints)
```

### Cascade Backend Tiers

1. **Tier 1: Google Cloud ARCO (`use_google_arco=True`):**
   Reads directly from the public Google Cloud ARCO-ERA5 Zarr store (`gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`) via `gcsfs` and `xarray`.
   * **Advantages:** High throughput, sub-second temporal slicing, and zero user quota constraints.
   * **Dynamic Availability Check:** The system inspects the store's temporal coordinates to ensure the requested `end_date` is available (Google Cloud ARCO has a typical 15- to 90-day operational lag) and confirms that all requested variables exist in the store. If the requested dates exceed the catalog or a network error occurs, the cascade automatically advances to Tier 2.

2. **Tier 2: Copernicus CDS Legacy API (`use_legacy=True`):**
   Requests data from the official Copernicus Climate Data Store using `cdsapi` and `data_format="netcdf4"`.
   * **Advantages:** Real-time availability up to 5 days before the present date; bypasses external `cfgrib` C-library dependencies by decoding NetCDF4 via `netcdf4` and `h5netcdf`.
   * **Adaptive Chunking:** Automatically segments queries into 1- to 5-day windows to prevent server timeouts. If CDS quotas are exceeded or credentials are absent, the cascade advances to Tier 3.

3. **Tier 3: Copernicus CDS ARCO via `obstore` (`use_cds_arco=True`):**
   Streams analysis-ready cloud Zarr stores from ECMWF CADL endpoints using `obstore` with query-aware routing (`geoChunked` for localized station extraction, `timeChunked` for broad spatial maps).

### Explicit User Overrides

Users can bypass the automated cascade at any time by supplying explicit boolean flags:
* `use_google_arco=True`: Forces Google Cloud ARCO extraction.
* `use_legacy=True` (or `force_cds=True`): Forces Copernicus CDS NetCDF4 extraction.
* `use_cds_arco=True`: Forces Copernicus CDS CADL ARCO extraction via `obstore`.

---

## CADL ARCO vs CDS GRIB Variable Name Resolution (DEC-023)

In ECMWF CADL ARCO Zarr stores, variable identifiers differ from legacy CDS GRIB codes. For example, CADL stores designate 2-metre temperature as `t2m` (not GRIB `2t`), 10-metre winds as `u10`/`v10` (not GRIB `10u`/`10v`), and dewpoint as `d2m` (not GRIB `2d`).

MasharikiWeather solves this with a bidirectional candidate resolver:
$$\text{Candidates}(V) = \{V, \text{canonical}(V)\} \cup \{k \mid \text{Registry}[k] = \text{canonical}(V)\} \cup \{\text{short}(V)\}$$

Every requested variable is mapped across all possible representations. The extractor slices whichever variable name is present in the target store and standardizes the output `xr.Dataset` to canonical CF names.

### ERA5 Variable Cross-Reference Matrix

| Physical Parameter | Canonical Name | Legacy CDS GRIB Code | CADL ARCO Zarr Code | Stream |
|---|---|---|---|---|
| 2-Metre Temperature | `2m_temperature` | `2t` | `t2m` | Surface (`sfc`) |
| 2-Metre Dewpoint Temperature | `2m_dewpoint_temperature` | `2d` | `d2m` | Surface (`sfc`) |
| 10m U-Wind Component | `10m_u_component_of_wind` | `10u` | `u10` | Surface (`sfc`) |
| 10m V-Wind Component | `10m_v_component_of_wind` | `10v` | `v10` | Surface (`sfc`) |
| 100m U-Wind Component | `100m_u_component_of_wind` | `100u` | `u100` | Surface (`sfc`) |
| 100m V-Wind Component | `100m_v_component_of_wind` | `100v` | `v100` | Surface (`sfc`) |
| Total Precipitation | `total_precipitation` | `tp` | `tp` | Surface (`sfc`) |
| Convective Precipitation | `convective_precipitation` | `cp` | `cp` | Surface (`sfc`) |
| Large-Scale Precipitation | `large_scale_precipitation` | `lsp` | `lsp` | Surface (`sfc`) |
| Surface Pressure | `surface_pressure` | `sp` | `sp` | Surface (`sfc`) |
| Mean Sea Level Pressure | `mean_sea_level_pressure` | `msl` | `msl` | Surface (`sfc`) |
| Surface Solar Radiation Downwards | `surface_solar_radiation_downwards` | `ssrd` | `ssrd` | Surface (`sfc`) |
| Top-of-Atmosphere Incident Solar Radiation | `toa_incident_solar_radiation` | `tisr` | `tisr` | Surface (`sfc`) |
| Total Cloud Cover | `total_cloud_cover` | `tcc` | `tcc` | Surface (`sfc`) |
| Total Column Water Vapour | `total_column_water_vapour` | `tcwv` | `tcwv` | Surface (`sfc`) |
| Convective Available Potential Energy | `convective_available_potential_energy` | `cape` | `cape` | Surface (`sfc`) |
| Convective Inhibition | `convective_inhibition` | `cin` | `cin` | Surface (`sfc`) |
| Sea Surface Temperature | `sea_surface_temperature` | `sst` | `sst` | Surface (`sfc`) |
| Volumetric Soil Water Layer 1 | `volumetric_soil_water_layer_1` | `swvl1` | `swvl1` | Surface (`sfc`) |
| Soil Temperature Level 1 | `soil_temperature_level_1` | `stl1` | `stl1` | Surface (`sfc`) |
| Significant Wave Height | `significant_height_of_combined_wind_waves_and_swell` | `swh` | `swh` | Waves (`wav`) |
| Mean Wave Direction | `mean_wave_direction` | `mwd` | `mwd` | Waves (`wav`) |
| Mean Wave Period | `mean_wave_period` | `mwp` | `mwp` | Waves (`wav`) |
| Geopotential (Pressure Levels) | `geopotential` | `z` | `z` | Pressure (`pl`) |
| Temperature (Pressure Levels) | `temperature` | `t` | `t` | Pressure (`pl`) |
| U-Wind Component (Pressure Levels) | `u_component_of_wind` | `u` | `u` | Pressure (`pl`) |
| V-Wind Component (Pressure Levels) | `v_component_of_wind` | `v` | `v` | Pressure (`pl`) |
| Specific Humidity (Pressure Levels) | `specific_humidity` | `q` | `q` | Pressure (`pl`) |
| Relative Humidity (Pressure Levels) | `relative_humidity` | `r` | `r` | Pressure (`pl`) |
| Vertical Velocity (Pressure Levels) | `vertical_velocity` | `w` | `w` | Pressure (`pl`) |

### Automated Multi-Stream Extraction

When requested variables span multiple CADL streams (such as combining surface `2m_temperature` from `sfc`, wave `significant_height_of_combined_wind_waves_and_swell` from `wav`, and upper-air `geopotential` from `pl`), the extractor detects the required streams, queries each target store independently, and merges the resulting slices into a unified `xr.Dataset`.

---

## Source Parameter Validation and UserWarnings (DEC-023)

To prevent runtime crashes and silent failures, each extractor maintains a capability registry defining the physical parameters it provides. When an unsupported variable is requested (e.g., requesting `temperature` from `source="imerg"`), MasharikiWeather issues an informative `UserWarning`:

```text
UserWarning: Variable 'temperature' is not provided by source 'imerg'.
Available variables: ['precipitation', 'precipitation_amount', 'precipitationcal', 'total_precipitation'].
```

### Supported Variable Inventory by Source

| Source Key | Supported Physical Parameters |
|---|---|
| `imerg` | `precipitationcal`, `precipitation_amount`, `precip`, `total_precipitation`, `precipitation`, `probabilityliquidprecipitation` |
| `chirps` | `precip`, `precipitation`, `precipitation_amount`, `total_precipitation` |
| `tamsat` | `rfe`, `rfe_filled`, `precipitation`, `precipitation_amount`, `soil_moisture` |
| `era5` | All surface, wave, and pressure level variables listed in the Cross-Reference Matrix |
| `cams` | `pm2_5`, `pm10`, `dust`, `ozone`, `nitrogen_dioxide`, `sulphur_dioxide`, `carbon_monoxide` |
| `merra2` | `T2M`, `U10M`, `V10M`, `PRECTOTCORR`, `PS`, `SLP`, `QV2M`, `AODANA` |
| `open_meteo` | `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `precipitation`, `surface_pressure`, `wind_speed_10m`, `wind_direction_10m`, `shortwave_radiation` |
| `nasa_power` | `T2M`, `T2MDEW`, `RH2M`, `PRECTOTCORR`, `WS10M`, `WD10M`, `PS`, `ALLSKY_SFC_SW_DWN` |
| `pvgis` | `Gb(i)` (direct irradiance), `Gd(i)` (diffuse irradiance), `T2m`, `WS10m` |
| `sarah3` | `SIS` (surface incoming shortwave), `SID` (surface incoming direct), `DNI` (direct normal irradiance) |
| `landsaf` | `LST` (land surface temperature), `DSSF` (downwelling surface shortwave flux), `FVC` (fractional vegetation cover) |
| `tropomi` | `nitrogendioxide_tropospheric_column`, `carbonmonoxide_total_column`, `ozone_total_column` |

---

## Code Examples

### 1. Default ERA5 Priority Cascade

```python
from masharikiweather.dataset import extract_gridded

# Default cascade: attempts Google Cloud ARCO, falls back to CDS NetCDF4, then CDS ARCO
ds_era5 = extract_gridded(
    source="era5",
    start_date="2023-01-01",
    end_date="2023-01-07",
    bbox=[-5.0, 33.0, 5.0, 42.0],  # [lat_min, lon_min, lat_max, lon_max]
    variables=["2m_temperature", "total_precipitation", "surface_pressure"],
    output_dir="./mashariki_data/gridded/era5",
    save_zarr=True,
)
```

### 2. Explicit Backend Selection

```python
# Force Google Cloud ARCO extraction
ds_google = extract_gridded(
    source="era5",
    start_date="2023-01-01",
    end_date="2023-01-07",
    variables=["2m_temperature", "total_precipitation"],
    use_google_arco=True,
    output_dir="./mashariki_data/gridded/era5_google",
)

# Force Copernicus CDS Legacy API with NetCDF4 format and adaptive chunking
ds_cds = extract_gridded(
    source="era5",
    start_date="2023-01-01",
    end_date="2023-01-07",
    variables=["2m_temperature", "total_precipitation"],
    use_legacy=True,
    data_format="netcdf4",
    chunk_days=5,
    output_dir="./mashariki_data/gridded/era5_cds",
)

# Force Copernicus CDS CADL ARCO via obstore
ds_arco = extract_gridded(
    source="era5",
    start_date="2023-01-01",
    end_date="2023-01-07",
    variables=["2m_temperature", "total_precipitation"],
    use_cds_arco=True,
    chunking="geo",
    output_dir="./mashariki_data/gridded/era5_cadl",
)
```

### 3. TAHMO Extraction with Server-Respecting Concurrency

```python
from masharikiweather.dataset import pull_station_data

# Extract TAHMO observations using safe thread concurrency (default 4 workers)
tahmo_files = pull_station_data(
    source="tahmo",
    years=[2023],
    variables=["2m_temperature", "total_precipitation", "relative_humidity"],
    max_workers=4,
    output_dir="./mashariki_data",
)
```

### 4. NASA IMERG Satellite Precipitation

```python
# Automatic transition between Final Run (< Nov 2025) and Late Run (>= Nov 2025)
ds_imerg = extract_gridded(
    source="imerg",
    start_date="2023-01-01",
    end_date="2023-01-02",
    bbox=[-5.0, 33.0, 5.0, 42.0],
    output_dir="./mashariki_data/gridded/imerg",
    save_zarr=True,
)
```

### 5. Combined In-Situ and Reanalysis Extraction

```python
from masharikiweather.dataset import pull_weather_data
from masharikiweather.extract.apis.base import EAST_AFRICA_BBOX

# Extract ground observations and gridded reanalysis in a single coordinated pipeline
result = pull_weather_data(
    start_date="2023-01-01",
    end_date="2023-01-03",
    bbox=EAST_AFRICA_BBOX,
    station_source="tahmo",
    gridded_source="era5",
    station_variables=["2m_temperature", "total_precipitation"],
    gridded_variables=["2m_temperature", "total_precipitation", "surface_pressure"],
    max_workers=4,
    output_dir="./mashariki_data",
)

print("Station observations saved to:", result["stations"])
print("Gridded reanalysis saved to:", result["gridded"])
```
