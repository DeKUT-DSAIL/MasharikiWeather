"""
example.py - how to use imerg_wrapper.

When you run this, get_imerg() first shows how to create a free NASA Earthdata
account and asks whether you have one:
  - answer "yes"  -> it logs you in (username/password prompt) and downloads
  - answer "no"   -> it walks you through registration, then continues

    pip install earthaccess h5py xarray dask pandas tqdm matplotlib "zarr>=3"
    python example.py

Batch jobs (SLURM, cron) can't answer prompts: create the account beforehand,
export EARTHDATA_USERNAME / EARTHDATA_PASSWORD (or use ~/.netrc) and pass
has_account=True to get_imerg().
"""

from imerg_wrapper import get_imerg, convert_to_zarr_only, open_imerg_zarr

OUT = "./IMERG"                      # use "/content/IMERG" on Colab
BBOX = (36.0, -1.5, 38.0, 0.5)       # west, south, east, north (central Kenya)

# --------------------------------------------------------------------------- #
# 1. One call: download + convert to Zarr
# --------------------------------------------------------------------------- #
result = get_imerg(
    start_date="2025-01-01",
    end_date="2025-01-03",           # inclusive: all of Jan 3 is downloaded
    bbox=BBOX,
    output_dir=OUT,
    dataset_name="GPM_3IMERGHH",
    convert_to_zarr=True,            # needs zarr>=3 (Python >= 3.11)
    # max_workers is optional; it defaults to your CPU count - 1 (max 8)
)
print(result)

if not result.ok:
    print("Failed months:", result.failed_months)
    print("Zarr errors:  ", result.zarr_errors)
    raise SystemExit(1)

# --------------------------------------------------------------------------- #
# 2. Use the data (lazy xarray Dataset, cropped to BBOX)
# --------------------------------------------------------------------------- #
ds = result.open(variables=["precipitation"])
print(ds)

precip = ds["precipitation"]                         # mm/hr, half-hourly

# Area-mean rain rate through time
mean_rate = precip.mean(dim=["lat", "lon"])
print(mean_rate.to_series().describe())

# Accumulated rainfall over the period (mm): rate (mm/h) x 0.5 h per step
total_mm = (precip * 0.5).sum(dim="time")
print("Max 3-day accumulation in the box: %.1f mm" % float(total_mm.max()))

# Daily totals averaged over the box
daily_mm = (precip * 0.5).mean(dim=["lat", "lon"]).resample(time="1D").sum()
print(daily_mm.to_series())

# Quick plot (optional)
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    mean_rate.plot(ax=ax[0])
    ax[0].set_title("Area-mean rain rate (mm/h)")
    total_mm.T.plot(ax=ax[1])        # IMERG arrays are (time, lon, lat); transpose to lat x lon
    ax[1].set_title("3-day accumulation (mm)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/example_precip.png", dpi=120)
    print(f"Saved {OUT}/example_precip.png")
except ImportError:
    pass

# --------------------------------------------------------------------------- #
# 3. Other useful patterns
# --------------------------------------------------------------------------- #

# a) Download only, convert later (e.g. on a different machine)
#    get_imerg("2025-02-01", "2025-02-07", BBOX, OUT, convert_to_zarr=False)
#    convert_to_zarr_only(f"{OUT}/GPM_3IMERGHH", f"{OUT}/zarr", bbox=BBOX, years=[2025])

# b) Re-open Zarr data you already have, no network needed
#    ds = open_imerg_zarr(f"{OUT}/zarr", "2025-01-01", "2025-01-03")

# c) Changed the bbox for months already converted? Force a rebuild
#    get_imerg("2025-01-01", "2025-01-03", (35.0, -2.0, 39.0, 1.0), OUT, overwrite_zarr=True)

# d) Only keep the variables you need (smaller Zarr)
#    get_imerg(..., variables=["precipitation", "precipitationQualityIndex"])

# e) Run unattended (skips the account question; you must already have an account)
#    get_imerg("2025-01-01", "2025-01-03", BBOX, OUT, has_account=True)
