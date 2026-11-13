"""
Credentials and Authentication Management for MasharikiWeather APIs.

Provides a unified Bring Your Own Key (BYOK) credential manager that discovers
authentication secrets from explicit parameters, environment variables,
local .env files, and configuration files (~/.cdsapirc, ~/.netrc).
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


class CredentialsError(Exception):
    """Raised when required API credentials cannot be located."""

    def __init__(self, service: str, message: str, resolution_hints: Optional[List[str]] = None) -> None:
        """Initialize CredentialsError with service identifier, diagnostic explanation, and resolution guidance."""
        self.service = service
        self.resolution_hints = resolution_hints or []
        hint_text = "\n".join(f"  - {h}" for h in self.resolution_hints)
        full_msg = f"Credentials missing or invalid for '{service}': {message}"
        if self.resolution_hints:
            full_msg += f"\nHow to provide credentials:\n{hint_text}"
        super().__init__(full_msg)


def parse_date(date_val: Union[str, datetime, date, pd.Timestamp]) -> pd.Timestamp:
    """
    Parse date or timestamp input into a normalized timezone-naive pandas Timestamp.

    Parameters
    ----------
    date_val : Union[str, datetime, date, pd.Timestamp]
        Input temporal representation.

    Returns
    -------
    pd.Timestamp
        Timezone-naive pandas Timestamp representation.
    """
    ts = pd.to_datetime(date_val)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def parse_date_range(
    start_date: Union[str, datetime, date, pd.Timestamp],
    end_date: Union[str, datetime, date, pd.Timestamp],
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Validate and return start and end timestamps ensuring start <= end.

    Mathematical docstring:
    Temporal interval: [t_start, t_end] with t_start <= t_end.

    Parameters
    ----------
    start_date : Union[str, datetime, date, pd.Timestamp]
        Initial timestamp.
    end_date : Union[str, datetime, date, pd.Timestamp]
        Terminal timestamp.

    Returns
    -------
    Tuple[pd.Timestamp, pd.Timestamp]
        Normalized (start_timestamp, end_timestamp).
    """
    s = parse_date(start_date)
    e = parse_date(end_date)
    if s > e:
        raise ValueError(f"start_date ({s}) cannot exceed end_date ({e})")
    return s, e


def _parse_env_file(path: Union[str, Path]) -> Dict[str, str]:
    """
    Parse a key=value .env file without external dependencies.
    Ignores comments and blank lines. Strips surrounding quotes.
    """
    env_path = Path(path).expanduser().resolve()
    if not env_path.is_file():
        return {}

    parsed: Dict[str, str] = {}
    with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key:
                parsed[key] = val
    return parsed


def _parse_cdsapirc(path: Union[str, Path]) -> Dict[str, str]:
    """
    Parse standard CDS API configuration file (~/.cdsapirc).
    Expected format:
      url: https://cds.climate.copernicus.eu/api
      key: <cds-api-key>
    """
    rc_path = Path(path).expanduser().resolve()
    if not rc_path.is_file():
        return {}

    parsed: Dict[str, str] = {}
    with open(rc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" in stripped:
                k, v = stripped.split(":", 1)
                parsed[k.strip().lower()] = v.strip().strip("'\"")
    return parsed


class CredentialsManager:
    """
    Discovers, verifies, and manages credentials across supported weather and satellite services.
    Adheres to Bring Your Own Key (BYOK) architecture.
    """

    @classmethod
    def _get_candidate_env_paths(cls, env_file: Optional[Union[str, Path]] = None) -> List[Path]:
        """
        Generate candidate search paths for environment files.
        Checks explicit env_file if provided, followed by .env in current
        working directory, AI4weather sibling repositories, and user home directory.
        Never enforces a required file path.
        """
        candidates: List[Path] = []
        if env_file is not None:
            candidates.append(Path(env_file).expanduser().resolve())
        candidates.append(Path.cwd() / ".env")
        # Multi-repo layout: check AI4weather sibling directory
        candidates.append(Path.cwd().parent / "AI4weather" / ".env")
        candidates.append(Path.cwd().parent / "AI4weather" / "notebooks" / ".env")
        # User home directory locations
        candidates.append(Path.home() / "AI4weather" / ".env")
        candidates.append(Path.home() / "AI4weather" / "notebooks" / ".env")
        candidates.append(Path.home() / ".env")
        return candidates

    @classmethod
    def get_tahmo_credentials(
        cls,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        env_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, str]:
        """
        Discover TAHMO DataHub credentials.

        Resolution precedence:
        1. Explicit `api_key` and `api_secret` arguments.
        2. Environment variables `TAHMO_API_KEY` and `TAHMO_API_SECRET`.
        3. Optional `env_file` argument, plus .env in current directory or user home directory.

        Mathematical docstring:
        Output authentication mapping: {'api_key': str, 'api_secret': str}

        Returns
        -------
        Dict[str, str]
            Dictionary containing 'api_key' and 'api_secret'.
        """
        key = api_key or os.environ.get("TAHMO_API_KEY") or os.environ.get("TAHMO_KEY")
        secret = api_secret or os.environ.get("TAHMO_API_SECRET") or os.environ.get("TAHMO_SECRET")

        if key and secret:
            return {"api_key": key, "api_secret": secret}

        candidate_paths = cls._get_candidate_env_paths(env_file=env_file)
        if os.environ.get("TAHMO_ENV_FILE"):
            candidate_paths.insert(0, Path(os.environ["TAHMO_ENV_FILE"]).expanduser().resolve())

        for path in candidate_paths:
            if path.is_file():
                parsed = _parse_env_file(path)
                k = parsed.get("TAHMO_API_KEY") or parsed.get("TAHMO_KEY")
                s = parsed.get("TAHMO_API_SECRET") or parsed.get("TAHMO_SECRET")
                if k and not key:
                    key = k
                if s and not secret:
                    secret = s
                if key and secret:
                    return {"api_key": key, "api_secret": secret}

        hints = [
            "Pass explicitly: get_tahmo_credentials(api_key='...', api_secret='...')",
            "Set environment variables: export TAHMO_API_KEY='...' and export TAHMO_API_SECRET='...'",
            "Provide an optional env_file argument or create a .env file containing TAHMO_API_KEY and TAHMO_API_SECRET.",
        ]
        raise CredentialsError("TAHMO DataHub", "Missing API key or API secret.", hints)

    @classmethod
    def get_cds_credentials(
        cls,
        url: Optional[str] = None,
        key: Optional[str] = None,
        cds_url: Optional[str] = None,
        cds_key: Optional[str] = None,
        rc_file: Optional[Union[str, Path]] = None,
        cds_rc_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Discover Copernicus Climate Data Store (CDS) credentials.

        Resolution precedence:
        1. Explicit function arguments (`key`, `url`, `cds_key`, `cds_url`).
        2. Environment variables `CDSAPI_KEY` and `CDSAPI_URL`.
        3. Explicitly specified `rc_file` or `cds_rc_file`.
        4. If not explicitly provided, returns None for url and key so that
           cdsapi.Client() initializes with its native automatic discovery
           (~/.cdsapirc or standard environment variables).

        Default URL:
        If key is present and url is not specified, defaults to
        'https://cds.climate.copernicus.eu/api'.

        Mathematical docstring:
        Output authentication mapping: {'url': Optional[str], 'key': Optional[str]}

        Returns
        -------
        Dict[str, Optional[str]]
            Dictionary containing 'url' and 'key'.
        """
        default_url = "https://cds.climate.copernicus.eu/api"
        u = url or cds_url or os.environ.get("CDSAPI_URL")
        k = key or cds_key or os.environ.get("CDSAPI_KEY")

        if k:
            return {"url": u or default_url, "key": k}

        target_rc = rc_file or cds_rc_file
        if target_rc is not None and Path(target_rc).expanduser().is_file():
            parsed = _parse_cdsapirc(target_rc)
            found_url = parsed.get("url")
            found_key = parsed.get("key")
            if found_key:
                return {"url": found_url or default_url, "key": found_key}

        # Return None so cdsapi.Client() executes its native automatic discovery
        return {"url": u, "key": None}

    @classmethod
    def get_cds_api_key(
        cls,
        key: Optional[str] = None,
        rc_file: Optional[Union[str, Path]] = None,
        env_file: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Discover Copernicus CDS API key string from explicit argument, environment, or ~/.cdsapirc.

        Parameters
        ----------
        key : Optional[str]
            Explicitly provided CDS API key.
        rc_file : Optional[Union[str, Path]]
            Custom path to .cdsapirc file.
        env_file : Optional[Union[str, Path]]
            Optional path to .env file.

        Returns
        -------
        str
            Discovered CDS API key string.
        """
        if key:
            return key
        if os.environ.get("CDSAPI_KEY"):
            return os.environ["CDSAPI_KEY"]

        target_rc = Path(rc_file).expanduser().resolve() if rc_file else Path("~/.cdsapirc").expanduser().resolve()
        if target_rc.is_file():
            parsed = _parse_cdsapirc(target_rc)
            if parsed.get("key"):
                return parsed["key"]

        candidate_paths = cls._get_candidate_env_paths(env_file=env_file)
        for path in candidate_paths:
            if path.is_file():
                parsed_env = _parse_env_file(path)
                k = parsed_env.get("CDSAPI_KEY") or parsed_env.get("CDS_KEY")
                if k:
                    return k

        hints = [
            "Export environment variable: export CDSAPI_KEY='your-cds-api-key'",
            "Add your key to ~/.cdsapirc: key: <your-cds-api-key>",
            "Pass cds_key='your-cds-api-key' directly to the extractor or method.",
            "Consult the official Copernicus CDS User Guide: https://cds.climate.copernicus.eu/user-guide",
        ]
        raise CredentialsError("Copernicus CDS", "Missing CDS API key.", hints)

    @classmethod
    def get_earthdata_credentials(
        cls,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        env_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, str]:
        """
        Discover NASA Earthdata / Earth Access credentials.

        Resolution precedence:
        1. Explicit `username`, `password`, or `token` arguments.
        2. Environment variables `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`, `EARTHDATA_TOKEN`, `EDL_TOKEN`.
        3. Netrc configuration in $HOME/.netrc or ~/.netrc.
        4. Optional `env_file` argument, plus .env in current directory or user home directory.

        Mathematical docstring:
        Output authentication mapping: {'username': str, 'password': str} (and optional 'token': str)

        Returns
        -------
        Dict[str, str]
            Dictionary containing credentials.
        """
        user = username or os.environ.get("EARTHDATA_USERNAME") or os.environ.get("EARTHDATA_USER")
        pwd = password or os.environ.get("EARTHDATA_PASSWORD") or os.environ.get("EARTHDATA_PASS")
        tok = token or os.environ.get("EARTHDATA_TOKEN") or os.environ.get("EDL_TOKEN")

        if user and pwd:
            res = {"username": user, "password": pwd}
            if tok:
                res["token"] = tok
            return res
        if tok and not user and not pwd:
            return {"token": tok, "username": "", "password": tok}

        netrc_candidates: List[Path] = []
        if "HOME" in os.environ:
            netrc_candidates.append(Path(os.environ["HOME"]) / ".netrc")
        netrc_candidates.append(Path.home() / ".netrc")

        for netrc_path in netrc_candidates:
            if netrc_path.is_file():
                try:
                    import netrc
                    n = netrc.netrc(str(netrc_path))
                    auth = n.authenticators("urs.earthdata.nasa.gov")
                    if not auth:
                        auth = n.authenticators("default")
                    if auth:
                        u, _, p = auth
                        if u and p:
                            res = {"username": u, "password": p}
                            if tok:
                                res["token"] = tok
                            return res
                except Exception:
                    pass

        candidate_paths = cls._get_candidate_env_paths(env_file=env_file)
        if os.environ.get("EARTHDATA_ENV_FILE"):
            candidate_paths.insert(0, Path(os.environ["EARTHDATA_ENV_FILE"]).expanduser().resolve())

        for path in candidate_paths:
            if path.is_file():
                parsed = _parse_env_file(path)
                u = parsed.get("EARTHDATA_USERNAME") or parsed.get("EARTHDATA_USER")
                p = parsed.get("EARTHDATA_PASSWORD") or parsed.get("EARTHDATA_PASS")
                t = parsed.get("EARTHDATA_TOKEN") or parsed.get("EDL_TOKEN")
                if u and not user:
                    user = u
                if p and not pwd:
                    pwd = p
                if t and not tok:
                    tok = t
                if user and pwd:
                    res = {"username": user, "password": pwd}
                    if tok:
                        res["token"] = tok
                    return res
                if tok and not user and not pwd:
                    return {"token": tok, "username": "", "password": tok}

        if tok:
            return {"token": tok, "username": user or "", "password": pwd or tok}

        hints = [
            "Pass explicitly: get_earthdata_credentials(username='...', password='...') or get_earthdata_credentials(token='...')",
            "Set environment variables: export EARTHDATA_USERNAME='...' and export EARTHDATA_PASSWORD='...' (or export EARTHDATA_TOKEN='...')",
            "Configure ~/.netrc ($HOME/.netrc) with machine urs.earthdata.nasa.gov login <username> password <password>",
            "Provide an optional env_file argument or create a .env file with EARTHDATA_USERNAME and EARTHDATA_PASSWORD.",
            "Register for a free NASA Earthdata account at https://urs.earthdata.nasa.gov/",
        ]
        raise CredentialsError("NASA Earthdata / IMERG", "Missing Earthdata username/password or token.", hints)


def get_credentials(service: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Unified entry point to discover and retrieve credentials for an external service.

    Parameters
    ----------
    service : str
        Service identifier ('tahmo', 'earthdata', 'imerg', 'cds', 'era5', etc.).
    **kwargs : Any
        Explicit credential arguments and aliases passed directly to the service retriever.
        Supported keywords:
        - TAHMO: api_key, api_secret, env_file
        - CDS/ERA5/In-Situ: key, url, cds_key, cds_url, rc_file, cds_rc_file
        - Earthdata/IMERG: username, password, token, env_file

    Returns
    -------
    Dict[str, str]
        Credentials dictionary for the requested service.
    """
    srv = service.strip().lower()
    if srv in ("tahmo", "tahmo_datahub"):
        return CredentialsManager.get_tahmo_credentials(
            api_key=kwargs.get("api_key"),
            api_secret=kwargs.get("api_secret"),
            env_file=kwargs.get("env_file"),
        )
    elif srv in ("earthdata", "imerg", "nasa", "earth_access"):
        return CredentialsManager.get_earthdata_credentials(
            username=kwargs.get("username") or kwargs.get("user"),
            password=kwargs.get("password") or kwargs.get("pwd"),
            token=kwargs.get("token"),
            env_file=kwargs.get("env_file"),
        )
    elif srv in ("cds", "era5", "copernicus", "cds_insitu"):
        return CredentialsManager.get_cds_credentials(
            url=kwargs.get("url"),
            key=kwargs.get("key"),
            cds_url=kwargs.get("cds_url"),
            cds_key=kwargs.get("cds_key"),
            rc_file=kwargs.get("rc_file"),
            cds_rc_file=kwargs.get("cds_rc_file"),
        )
    else:
        raise ValueError(
            f"Unknown service '{service}'. Supported: 'tahmo', 'earthdata', 'imerg', 'cds', 'era5', 'cds_insitu'."
        )


get_cds_api_key = CredentialsManager.get_cds_api_key

