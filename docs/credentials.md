# Credentials

Some weather data providers require free API accounts. MasharikiWeather discovers credentials automatically without requiring hardcoded keys.

---

## Overview

When an extractor needs an API key, it checks the following in order:

1. **Function Arguments:** Parameters passed to extractors (e.g., `api_key="..."`, `cds_key="..."`).
2. **Environment Variables:** Variables set in your shell session or script (e.g., `CDSAPI_KEY`, `TAHMO_API_KEY`).
3. **Configuration Files:** Standard configuration files in your home directory (`~/.cdsapirc`, `~/.netrc`).
4. **Local `.env` File (Optional):** A `.env` file in the current working directory.

If credentials are provided as arguments or environment variables, no configuration files are needed.

---

## Service Credentials

| Service | Environment Variables | Config File | Registration Link |
|---|---|---|---|
| **TAHMO DataHub** | `TAHMO_API_KEY`, `TAHMO_API_SECRET` | None | [TAHMO Portal](https://tahmo.org/) |
| **Copernicus CDS (ERA5, CAMS)** | `CDSAPI_KEY` | `~/.cdsapirc` | [Copernicus CDS User Guide](https://cds.climate.copernicus.eu/user-guide) |
| **NASA Earthdata (IMERG, MERRA-2)** | `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD` | `~/.netrc` | [NASA Earthdata Login](https://urs.earthdata.nasa.gov/) |
| **Copernicus CDS In-Situ** | None (Public access) | None | None |
| **NOAA GHCN** | None (Public access) | None | None |
| **Open-Meteo** | None (Public access) | None | None |

---

## Examples

### Environment Variables

```bash
export TAHMO_API_KEY="your-tahmo-api-key"
export TAHMO_API_SECRET="your-tahmo-api-secret"
export CDSAPI_KEY="your-cds-api-key"
export EARTHDATA_USERNAME="your-earthdata-username"
export EARTHDATA_PASSWORD="your-earthdata-password"
```

### Copernicus CDS Configuration File (`~/.cdsapirc`)

```text
url: https://cds.climate.copernicus.eu/api
key: your-cds-api-key
```

### NASA Earthdata Configuration File (`~/.netrc`)

```text
machine urs.earthdata.nasa.gov login your-earthdata-username password your-earthdata-password
```

### Python Arguments

```python
from masharikiweather.extract.apis.era5 import ERA5Extractor

extractor = ERA5Extractor(cds_key="your-cds-api-key")
```
