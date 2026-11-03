# TAHMO DataHub API Wrapper

A small Python client for the [TAHMO DataHub API](https://datahub.tahmo.org). It downloads weather station
measurements as pandas DataFrames, can select stations inside a latitude/longitude bounding box, and saves the
results as **CSV**, **Parquet** or **Zarr**.

## Project structure

```
TAHMO/
  __init__.py       the apiWrapper class
.gitignore
README.md
example.ipynb       notebook examples
example.py          command line and code examples
requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Python 3.8 or newer is recommended. `pyarrow` is needed for Parquet and `xarray` + `zarr` for Zarr. If you only
need CSV, `requests`, `numpy`, `pandas` and `python-dateutil` are enough.

## Credentials

You need a TAHMO DataHub API key and secret. There are three ways to provide them:

```python
from TAHMO import apiWrapper

api = apiWrapper('MY_KEY', 'MY_SECRET')      # 1. constructor
api.setCredentials('MY_KEY', 'MY_SECRET')    # 2. setter
api = apiWrapper()                           # 3. environment variables
```

For option 3, set `TAHMO_API_KEY` and `TAHMO_API_SECRET`. Do not hardcode the secret in files you commit
(`.gitignore` already excludes `.env` files and the `output/` folder).

## Quick start

```python
from TAHMO import apiWrapper

api = apiWrapper()
api.setCredentials('MY_KEY', 'MY_SECRET')

# One station
df = api.getMeasurements('TA00001', startDate='2024-01-01', endDate='2024-01-31',
                         variables=['pr', 'te'])

# All stations inside a bounding box
df = api.getMeasurementsInBounds(latMin=-4.7, latMax=5.0, lonMin=33.9, lonMax=41.9,
                                 startDate='2024-01-01', endDate='2024-01-31',
                                 variables=['pr'])

# Save it
api.saveData(df, 'output/kenya_rainfall.parquet')
```

## API reference

### Measurements

| Method | Description |
|---|---|
| `getMeasurements(station, startDate, endDate, variables=None, dataset='controlled')` | Measurements of one station. Returns a DataFrame indexed by time, one column per variable. |
| `getRawMeasurements(station, startDate, endDate, variables=None)` | Same as above with the unfiltered `raw` dataset. |
| `getMeasurementsForStations(stations, startDate, endDate, variables=None, dataset='controlled', skipErrors=True)` | Several stations in one DataFrame. Columns are a MultiIndex `(station, variable)`. |
| `getMeasurementsInBounds(latMin, latMax, lonMin, lonMax, startDate, endDate, variables=None, dataset='controlled', skipErrors=True)` | Finds the stations inside the box and downloads them. |

**Arguments**

- `station` / `stations`: station code such as `'TA00001'`, or a list of codes.
- `startDate`, `endDate`: both are required. Strings (`'2024-01-01'`, `'2024-01-01T06:00:00'`), `datetime`
  or `pandas.Timestamp`. An end date given **without a time** includes that whole day. Timezone-aware values are
  converted to UTC.
- `variables`: list of shortcodes such as `['pr', 'te']`. `None` returns every variable. Use `getVariables()` to
  see what exists.
- `dataset`: `'controlled'` (quality checked, default) or `'raw'`.
- `skipErrors`: if `True`, a station that fails is skipped and its message is stored in `api.lastErrors`.

**Behaviour to know**

- Periods longer than 365 days are split automatically into several API requests.
- Observations whose quality flag is not `1` are returned as `NaN`.
- If several sensors measure the same variable, the columns are named `variable_sensor`.
- Stations with no data in the period are left out of multi-station results.

### Stations and variables

| Method | Description |
|---|---|
| `getStations()` | Dict of all stations keyed by station code. |
| `getStationsInBounds(latMin, latMax, lonMin, lonMax)` | The stations whose coordinates are inside the box (edges included). Stations without coordinates are ignored. |
| `getVariables()` | Dict of all variables keyed by shortcode. |

Latitude runs from -90 to 90 and longitude from -180 to 180. Southern latitudes and western longitudes are
negative. Kenya is roughly `latMin=-4.7, latMax=5.0, lonMin=33.9, lonMax=41.9`.

> The coordinates are read from `station['location']['latitude']` and `['longitude']`, with a fallback to
> top-level `latitude`/`longitude` keys. If your station records use other names, print one with
> `next(iter(api.getStations().values()))` and adjust `_stationCoordinates` in `TAHMO/__init__.py`.

### Saving

```python
api.saveData(df, 'output/data.csv')                 # format taken from the extension
api.saveData(df, 'output/data', fmt='parquet')      # or give it explicitly
api.saveData(df, 'output/data.zarr')
```

| Format | Notes |
|---|---|
| `csv` | Opens in Excel. Largest files, no data types kept. |
| `parquet` | Compressed and fast, keeps data types. Recommended for large tables. Needs `pyarrow`. |
| `zarr` | Written as a **folder**, not a single file. Needs `xarray` and `zarr`. Timestamps are stored as UTC. |

Multi-station DataFrames have their `(station, variable)` columns flattened to names like `TA00001_pr`, and
the index is named `time`. `saveData` creates missing folders and returns the path it wrote.

## Command line

`example.py` is a ready-made command line tool (run it from the project root):

```bash
# See what is available
python example.py --list-variables
python example.py --list-stations

# One station to CSV
python example.py --stations TA00001 --start 2024-01-01 --end 2024-01-31 --vars pr te --format csv

# Several stations to Parquet
python example.py --stations TA00001 TA00002 --start 2024-01-01 --end 2024-06-30 --vars pr te --format parquet

# Everything inside a bounding box to Zarr
python example.py --bbox -4.7 5.0 33.9 41.9 --start 2024-01-01 --end 2024-03-31 --vars pr --format zarr --output output/kenya
```

If `--key`/`--secret` are not given and the environment variables are not set, you are prompted (the secret is
hidden as you type). See `example.ipynb` for the same workflow step by step in a notebook.

## Errors

API failures raise an `Exception` carrying the message returned by the API (for example wrong credentials), or
`API request failed with status code <n>` when the response has no message. Invalid dates and unsupported formats
raise `ValueError`.

## Notes

- Large boxes and long periods mean many API calls. Start with a small area and a short period.
- Be mindful of the DataHub's terms of use and rate limits when downloading a lot of data.
