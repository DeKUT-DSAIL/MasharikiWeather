# Bring-Your-Own-Key (BYOK) Credential Configuration

MasharikiWeather adheres to a Bring-Your-Own-Key (BYOK) architecture. No commercial, governmental, or proprietary authentication secrets are hardcoded into the codebase, committed to Git repositories, or bundled with releases.

The unified authentication engine is implemented in `masharikiweather.extract.apis.credentials.CredentialsManager`. It automatically resolves credentials across four prioritized tiers without requiring user intervention when standard configuration files or environment variables are present.

---

## 1. Unified Resolution Precedence

When an extraction asset or function requests credentials for an external meteorological service, `CredentialsManager` inspects candidate locations in the following order of precedence:

```text
   +-------------------------------------------------------------+
   | 1. Explicit Arguments (api_key='...', cds_key='...', etc.)  |
   +-------------------------------------------------------------+
                                |
                             (not set)
                                v
   +-------------------------------------------------------------+
   | 2. Process Environment Variables (TAHMO_API_KEY, CDSAPI_KEY)|
   +-------------------------------------------------------------+
                                |
                             (not set)
                                v
   +-------------------------------------------------------------+
   | 3. Official Configuration Files (~/.cdsapirc, ~/.netrc)     |
   +-------------------------------------------------------------+
                                |
                             (not set)
                                v
   +-------------------------------------------------------------+
   | 4. Environment Files (.env, sibling AI4weather, ~/.env)     |
   +-------------------------------------------------------------+
                                |
                             (not set)
                                v
   [Raise CredentialsError with Troubleshooting Instructions]
```

1. **Explicit Function Arguments:** Values passed directly to functions or class constructors (such as `api_key`, `api_secret`, `cds_key`, `cds_url`, `username`, `password`, `token`). Explicit arguments always supersede all other sources.
2. **Process Environment Variables:** Standard environment variables exported in the active shell environment (e.g., `TAHMO_API_KEY`, `CDSAPI_KEY`, `EARTHDATA_USERNAME`).
3. **Official Service Configuration Files:** System configuration files recognized by upstream client libraries (e.g., `~/.cdsapirc` for ECMWF CDS, `~/.netrc` for NASA Earthdata).
4. **Environment (`.env`) Files:** If an optional `env_file` argument is provided, or by searching standardized candidate paths including:
   - Current working directory: `./.env`
   - Sibling AI4weather project directories: `../AI4weather/.env` and `../AI4weather/notebooks/.env`
   - User home directory paths: `~/AI4weather/.env`, `~/AI4weather/notebooks/.env`, and `~/.env`.

If credentials cannot be resolved across any tier, `CredentialsManager` raises a detailed `CredentialsError` containing actionable troubleshooting hints for the user.

---

## 2. Service Authentication Guides

### 2.1 TAHMO DataHub Ground Stations

TAHMO provides high-resolution ground weather station observations across sub-Saharan Africa via an authenticated REST API.

#### Required Secrets
* `api_key`: TAHMO DataHub client key string.
* `api_secret`: TAHMO DataHub client secret string.

#### Setup Methods

##### Method A: Environment File (`.env`)
Place your credentials in a `.env` file located in the current working directory or at `~/.env`:

```bash
# ~/.env
TAHMO_API_KEY=your_tahmo_api_key_here
TAHMO_API_SECRET=your_tahmo_api_secret_here
```

##### Method B: Shell Environment Variables
Export variables directly in your bash shell or `.bashrc`:

```bash
export TAHMO_API_KEY="your_tahmo_api_key_here"
export TAHMO_API_SECRET="your_tahmo_api_secret_here"
```

##### Method C: Direct Programmatic Specification
Pass credentials explicitly to `get_tahmo_credentials` or extraction methods:

```python
from masharikiweather.extract.apis.credentials import CredentialsManager

creds = CredentialsManager.get_tahmo_credentials(
    api_key="your_tahmo_api_key_here",
    api_secret="your_tahmo_api_secret_here",
)
print("Resolved TAHMO Key:", creds["api_key"][:4] + "****")
```

---

### 2.2 NASA IMERG Earthdata (Global Precipitation Measurement)

NASA's Global Precipitation Measurement (GPM) IMERG dataset provides satellite precipitation estimates through the NASA Earthdata Login portal. For detailed NASA Earthdata credential setup and token generation, refer to the [NASA earthaccess Documentation](https://earthaccess.readthedocs.io/en/stable/).

#### Required Secrets
* `username`: NASA Earthdata account username.
* `password`: NASA Earthdata account password.
* `token` *(optional)*: NASA Earthdata Bearer Token (alternative to username/password).

#### Setup Methods

##### Method A: Environment File (`.env`)
Store your Earthdata credentials in a local `.env` or `~/.env`:

```bash
# .env
EARTHDATA_USERNAME=your_earthdata_username
EARTHDATA_PASSWORD=your_earthdata_password
# Optional token alternative:
# EARTHDATA_TOKEN=your_earthdata_bearer_token
```

##### Method B: Standard `~/.netrc` Configuration
Create or edit your user netrc configuration file at `~/.netrc`:

```text
machine urs.earthdata.nasa.gov login <your_username> password <your_password>
```

Ensure correct file permissions:

```bash
chmod 600 ~/.netrc
```

##### Method C: Shell Environment Variables
```bash
export EARTHDATA_USERNAME="your_earthdata_username"
export EARTHDATA_PASSWORD="your_earthdata_password"
```

##### Method D: Direct Programmatic Specification
```python
from masharikiweather.extract.apis.credentials import CredentialsManager

creds = CredentialsManager.get_earthdata_credentials(
    username="your_earthdata_username",
    password="your_earthdata_password",
)
```

---

### 2.3 ECMWF Copernicus Climate Data Store (CDS & ERA5 Reanalysis)

The Copernicus Climate Data Store (CDS) hosts the ERA5 atmospheric reanalysis and Copernicus In-Situ global observations. For instructions on creating an account and generating your API key, refer to the official [Copernicus CDS User Guide](https://cds.climate.copernicus.eu/user-guide).

#### Required Secrets
* `url`: CDS API endpoint (default: `https://cds.climate.copernicus.eu/api`).
* `key`: Personal CDS API access key (UUID or Personal Access Token).

#### Setup Methods

##### Method A: Standard Configuration File (`~/.cdsapirc`)
Place your configuration in your user home directory at `~/.cdsapirc`:

```ini
url: https://cds.climate.copernicus.eu/api
key: your-cds-api-key-here
```

Ensure correct file permissions:

```bash
chmod 600 ~/.cdsapirc
```

##### Method B: Shell Environment Variables
```bash
export CDSAPI_URL="https://cds.climate.copernicus.eu/api"
export CDSAPI_KEY="your-cds-api-key-here"
```

##### Method C: Direct Programmatic Specification
```python
from masharikiweather.extract.apis.credentials import CredentialsManager

creds = CredentialsManager.get_cds_credentials(
    cds_url="https://cds.climate.copernicus.eu/api",
    cds_key="your-cds-api-key-here",
)
```

---

## 3. Error Handling and Diagnostic Feedback

If a requested credential cannot be resolved across any configured path, `CredentialsManager` raises `CredentialsError`:

```python
from masharikiweather.extract.apis.credentials import CredentialsError, CredentialsManager

try:
    creds = CredentialsManager.get_tahmo_credentials()
except CredentialsError as err:
    print(err)
```

Example diagnostic exception output:

```text
Credentials missing or invalid for 'TAHMO DataHub': Missing API key or API secret.
How to provide credentials:
  - Pass explicitly: get_tahmo_credentials(api_key='...', api_secret='...')
  - Set environment variables: export TAHMO_API_KEY='...' and export TAHMO_API_SECRET='...'
  - Provide an optional env_file argument or create a .env file containing TAHMO_API_KEY and TAHMO_API_SECRET.
```

This diagnostic format ensures users receive immediate actionable instructions without exposing sensitive credentials in stack traces.
