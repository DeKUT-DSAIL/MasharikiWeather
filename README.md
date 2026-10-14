# IMERG toolkit

Download NASA GPM **IMERG** precipitation data and (optionally) convert it to
monthly **Zarr** stores with a single call.

```python
from imerg_wrapper import get_imerg

result = get_imerg(
    start_date="2025-01-01",
    end_date="2025-01-03",              # inclusive: the whole last day is downloaded
    bbox=(36.0, -1.5, 38.0, 0.5),       # west, south, east, north
    output_dir="./IMERG",
    dataset_name="GPM_3IMERGHH",        # default
    convert_to_zarr=True,
)
ds = result.open()                      # lazy xarray.Dataset, cropped to your dates + bbox
```

## Contents

| File | Purpose |
|---|---|
| `imerg_wrapper.py` | The toolkit. Download, HDF5 to Zarr conversion and reader in one file. |
| `example.py` | End-to-end demo: download, open, area-mean, accumulation, plot. |
| `requirements.txt` | Python dependencies. |


## Install

Zarr 3 requires **Python 3.11 or newer**. Use a fresh environment:

```bash
conda create -n imerg python=3.12 -y
conda activate imerg
pip install -r requirements.txt
```

> In PowerShell/cmd always quote version specifiers (`"zarr>=3"`). An unquoted `>`
> is a shell redirect and creates a stray file instead of installing the package.

## Earthdata account (required)

IMERG is free but needs a NASA Earthdata login. When you call `get_imerg()` it shows a
short guide and asks **"Do you have a NASA Earthdata account? [yes/no]"**:

* **yes** - it logs you in (username/password prompt) and downloads.
* **no** - it walks you through registration, then continues once you confirm.

To create an account manually:

1. Register at <https://urs.earthdata.nasa.gov/users/new> and verify your email.
2. Log in, open **Applications > Authorized Apps > Approve More Applications**, and approve
   **NASA GESDISC DATA ARCHIVE**. Without this, downloads fail with HTTP 401/403.

For unattended jobs (SLURM, cron) there is no one to answer prompts. Create the account
first, export `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` (or create `~/.netrc`), and call
`get_imerg(..., has_account=True)`.

## `get_imerg` arguments

| Argument | Default | Description |
|---|---|---|
| `start_date`, `end_date` | required | `"YYYY-MM-DD"`; both inclusive. |
| `bbox` | required | `(west, south, east, north)` in degrees. |
| `output_dir` | required | Root folder for all output. |
| `dataset_name` | `"GPM_3IMERGHH"` | CMR short name. |
| `convert_to_zarr` | `True` | Also build monthly Zarr stores (needs zarr>=3). |
| `max_workers` | CPU count - 1 (max 8) | Parallel download threads / file readers. Optional. |
| `has_account` | `None` | `True` skips the account question (batch jobs). |
| `version` | `"07"` | Dataset version; `None` = no filter (may mix versions). |
| `variables` | `None` (all) | Variables to keep in the Zarr, e.g. `["precipitation"]`. |
| `zarr_dir` | `<output_dir>/zarr` | Override Zarr location. |
| `overwrite_zarr` | `False` | Rebuild Zarr for the requested months. |
| `attrs_path` | `None` | JSON of per-variable attributes to attach. |
| `retries` | `3` | Attempts per month for search/download. |

Returns an `IMERGResult` with `files`, `failed_months`, `empty_months`, `zarr_stores`,
`zarr_errors`, `ok` and `.open(variables=None)`.

## Output layout

```
<output_dir>/GPM_3IMERGHH/<YYYY>/<MM>/*.HDF5        raw granules
<output_dir>/zarr/<YYYY>/IMERG_<YYYY>_<MM>.zarr     converted, cropped to bbox
```

## Other helpers

```python
from imerg_wrapper import convert_to_zarr_only, open_imerg_zarr

# convert files you already downloaded
convert_to_zarr_only("./IMERG/GPM_3IMERGHH", "./IMERG/zarr", bbox=(36, -1.5, 38, 0.5), years=[2025])

# reopen existing Zarr without any network access
ds = open_imerg_zarr("./IMERG/zarr", "2025-01-01", "2025-01-03", variables=["precipitation"])
```

## Good to know

* **Downloads are global.** IMERG granules are whole-globe files (tens of MB each, 48 per
  day for half-hourly data), so `bbox` does not shrink the download. It only crops the Zarr.
* **Partial months work.** Only the days you ask for are fetched. Asking for more days of a
  month that is already converted rebuilds that month's Zarr automatically.
* **Changed the bbox?** Pass `overwrite_zarr=True` to re-crop months that already exist.
* **Rain rate units.** `precipitation` is in mm/hr; multiply by 0.5 h per step for mm per half hour.
* Zarr conversion fails fast (before any download) if zarr<3 or Python<3.11.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip` cannot find `zarr>=3` | Python is older than 3.11. Create a new environment (see Install). |
| HTTP 401/403 while downloading | Approve **NASA GESDISC DATA ARCHIVE** in your Earthdata profile. |
| Login keeps failing | Use your Earthdata *username* (not email); reset the password on the Earthdata site. |
| `No interactive input available...` | Running as a batch job: pass `has_account=True` and set credentials via env vars/`~/.netrc`. |
| `result.failed_months` is not empty | Re-run the same call; finished files are reused. Check your connection and login. |
