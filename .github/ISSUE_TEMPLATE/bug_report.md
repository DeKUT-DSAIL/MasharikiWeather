---
name: Bug Report
about: Report an issue with data extraction, tensor alignment, or environment failures.
title: "[BUG] "
labels: bug
assignees: 'Jane Doe'
---

## The Bug
**A clear and concise description of what is broken.**
*(e.g., "Calling `ds.get_windows()` with ERA5 data throws a KeyError on the latitude coordinate.")*

## Steps to Reproduce
**Provide the exact Python snippet to reproduce the error:**
```python
# Paste your initialization and extraction code here:
from masharikiweather import MasharikiWeatherDataset
ds = MasharikiWeatherDataset(...)
ds.get_windows(url="era5")
```
## Expected Behavior vs Actual Behavior
1. **Expected**: (e.g., "It should return a Windows dataclass with gridded_x shape (100, 24, 2, 9).")
2. **Actual**: (e.g., "It crashes with No group found in store fsspec.mapping.FSMap.")

## Environment Context
**Please verify our environment details:**
1. **OS**: (e.g., "Ubuntu 22.04")
2. **Python Version**: (e.g., "Python 3.10.12")
3. **Execution Environment**: (e.g., "Local Machine", "Google Colab", "WSL2")
4. **Relevant Libraries**: (e.g., "xarray 2023.12.0", "fsspec 2023.12.0")
For example: [`pip freeze | grep zarr`] or [`pip show zarr`] - Is it <3.0.0?
5. Credentials check: Are you sure your credentials are active and authorized (Yes/No)

## Error Traceback
**Paste the full error traceback here:**
```Plaintext
[Paste traceback here]
```





