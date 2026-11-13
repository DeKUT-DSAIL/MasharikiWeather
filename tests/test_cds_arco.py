"""
Tests for Copernicus CDS ARCO (Analysis-Ready Cloud-Optimized) ERA5 Zarr backend.

Validates:
1. Query-Aware Routing Matrix mapping exact surface, wave, and pressure level URLs.
2. Chunking inference (geoChunked for time-series / point queries vs timeChunked for global/wide snapshots).
3. Manual chunking override via chunking="geo" or chunking="time".
4. Obstore HTTPStore + ObjectStore connection with Bearer authorization.
5. End-to-end extraction and retrieval without crashing.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from masharikiweather.dataset import extract_gridded, pull_gridded_data
from masharikiweather.extract.apis.base import AFRICA_BBOX, SpatialBoundingBox
from masharikiweather.extract.apis.credentials import get_cds_api_key
from masharikiweather.extract.apis.era5 import (
    DEFAULT_CDS_ARCO_PL_GEO_URL,
    DEFAULT_CDS_ARCO_PL_TIME_URL,
    DEFAULT_CDS_ARCO_SFC_GEO_URL,
    DEFAULT_CDS_ARCO_SFC_TIME_URL,
    DEFAULT_CDS_ARCO_WAV_GEO_URL,
    DEFAULT_CDS_ARCO_WAV_TIME_URL,
    ECMWF_CDS_ARCO_URLS,
    ERA5Extractor,
    resolve_cds_arco_url,
)


# ==============================================================================
# 1. Exact CADL URL Matrix Verification
# ==============================================================================

def test_ecmwf_cds_arco_urls_exact_matrix() -> None:
    """Verify exact CADL ARCO URLs for surface, wave, and pressure level stores."""
    # 1. Surface (sfc)
    assert ECMWF_CDS_ARCO_URLS["sfc"]["geo"] == (
        "https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr"
    )
    assert ECMWF_CDS_ARCO_URLS["sfc"]["time"] == (
        "https://arco.datastores.ecmwf.int/cadl-arco-time-002/arco/reanalysis_era5_single_levels/sfc/timeChunked.zarr"
    )

    # 2. Waves (wav)
    assert ECMWF_CDS_ARCO_URLS["wav"]["geo"] == (
        "https://arco.datastores.ecmwf.int/cadl-arco-geo-003/arco/reanalysis_era5_single_levels/wav/geoChunked.zarr"
    )
    assert ECMWF_CDS_ARCO_URLS["wav"]["time"] == (
        "https://arco.datastores.ecmwf.int/cadl-arco-time-003/arco/reanalysis_era5_single_levels/wav/timeChunked.zarr"
    )

    # 3. Pressure Levels (pl)
    assert ECMWF_CDS_ARCO_URLS["pl"]["geo"] == (
        "https://arco.datastores.ecmwf.int/cadl-arco-geo-048/arco/reanalysis_era5_pressure_levels/pl/geoChunked.zarr"
    )
    assert ECMWF_CDS_ARCO_URLS["pl"]["time"] == (
        "https://arco.datastores.ecmwf.int/cadl-arco-time-048/arco/reanalysis_era5_pressure_levels/pl/timeChunked.zarr"
    )


# ==============================================================================
# 2. Query-Aware Routing Matrix: Streams & Chunking Toggles
# ==============================================================================

def test_routing_matrix_stream_inference() -> None:
    """Verify that variables and levels infer the correct stream (sfc, wav, pl)."""
    point_bbox = SpatialBoundingBox(lat_min=-1.28, lat_max=-1.28, lon_min=36.82, lon_max=36.82)

    # Surface variables route to sfc
    url, stream, _ = resolve_cds_arco_url(
        variables=["2m_temperature", "total_precipitation"],
        bbox=point_bbox,
        start_date="2023-01-01",
        end_date="2023-01-05",
    )
    assert stream == "sfc"
    assert url == DEFAULT_CDS_ARCO_SFC_GEO_URL

    # Wave variables route to wav
    url_wav, stream_wav, _ = resolve_cds_arco_url(
        variables=["significant_height_of_combined_wind_waves_and_swell"],
        bbox=point_bbox,
        start_date="2023-01-01",
        end_date="2023-01-05",
    )
    assert stream_wav == "wav"
    assert url_wav == DEFAULT_CDS_ARCO_WAV_GEO_URL

    # Wave short code routes to wav
    url_swh, stream_swh, _ = resolve_cds_arco_url(
        variables=["swh", "mwd"],
        bbox=point_bbox,
        start_date="2023-01-01",
        end_date="2023-01-05",
    )
    assert stream_swh == "wav"
    assert url_swh == DEFAULT_CDS_ARCO_WAV_GEO_URL

    # Pressure variables or explicit levels route to pl
    url_pl, stream_pl, _ = resolve_cds_arco_url(
        variables=["temperature", "specific_humidity"],
        levels=[850, 500],
        bbox=point_bbox,
        start_date="2023-01-01",
        end_date="2023-01-05",
    )
    assert stream_pl == "pl"
    assert url_pl == DEFAULT_CDS_ARCO_PL_GEO_URL

    # Explicit stream override takes precedence
    url_exp, stream_exp, _ = resolve_cds_arco_url(
        stream="wav",
        bbox=point_bbox,
        start_date="2023-01-01",
        end_date="2023-01-05",
    )
    assert stream_exp == "wav"
    assert url_exp == DEFAULT_CDS_ARCO_WAV_GEO_URL


def test_routing_matrix_query_aware_chunking_toggles() -> None:
    """Verify that routing toggles between geoChunked and timeChunked based on query shape."""
    point_bbox = SpatialBoundingBox(lat_min=-1.28, lat_max=-1.28, lon_min=36.82, lon_max=36.82)
    small_bbox = SpatialBoundingBox(lat_min=-1.5, lat_max=-1.0, lon_min=36.5, lon_max=37.0)
    wide_bbox = AFRICA_BBOX

    # Case A: Point / small bbox over extended time -> geoChunked
    url_pt, _, chunk_pt = resolve_cds_arco_url(
        stream="sfc",
        bbox=point_bbox,
        start_date="2023-01-01 00:00:00",
        end_date="2023-01-07 23:00:00",
    )
    assert chunk_pt == "geo"
    assert url_pt == DEFAULT_CDS_ARCO_SFC_GEO_URL

    url_sm, _, chunk_sm = resolve_cds_arco_url(
        stream="sfc",
        bbox=small_bbox,
        start_date="2023-01-01 00:00:00",
        end_date="2023-01-07 23:00:00",
    )
    assert chunk_sm == "geo"
    assert url_sm == DEFAULT_CDS_ARCO_SFC_GEO_URL

    # Case B: Single time stamp over wide spatial extent -> timeChunked
    url_time, _, chunk_time = resolve_cds_arco_url(
        stream="sfc",
        bbox=wide_bbox,
        start_date="2023-01-01 12:00:00",
        end_date="2023-01-01 12:00:00",
    )
    assert chunk_time == "time"
    assert url_time == DEFAULT_CDS_ARCO_SFC_TIME_URL

    # Case C: Wave store timeChunked for continental snapshot
    url_wav_time, _, chunk_wav_time = resolve_cds_arco_url(
        stream="wav",
        bbox=wide_bbox,
        start_date="2023-01-01 00:00:00",
        end_date="2023-01-01 00:00:00",
    )
    assert chunk_wav_time == "time"
    assert url_wav_time == DEFAULT_CDS_ARCO_WAV_TIME_URL

    # Case D: Pressure level timeChunked for continental snapshot
    url_pl_time, _, chunk_pl_time = resolve_cds_arco_url(
        stream="pl",
        bbox=wide_bbox,
        start_date="2023-01-01 00:00:00",
        end_date="2023-01-01 00:00:00",
    )
    assert chunk_pl_time == "time"
    assert url_pl_time == DEFAULT_CDS_ARCO_PL_TIME_URL


def test_routing_matrix_manual_chunking_override() -> None:
    """Verify that chunking='geo' or chunking='time' strictly overrides shape heuristics."""
    wide_bbox = AFRICA_BBOX
    point_bbox = SpatialBoundingBox(lat_min=-1.28, lat_max=-1.28, lon_min=36.82, lon_max=36.82)

    # Force geo on wide bbox + single timestamp
    url_geo_forced, _, chunk_geo = resolve_cds_arco_url(
        stream="sfc",
        bbox=wide_bbox,
        start_date="2023-01-01 00:00:00",
        end_date="2023-01-01 00:00:00",
        chunking="geo",
    )
    assert chunk_geo == "geo"
    assert url_geo_forced == DEFAULT_CDS_ARCO_SFC_GEO_URL

    # Force time on single point location
    url_time_forced, _, chunk_time = resolve_cds_arco_url(
        stream="sfc",
        bbox=point_bbox,
        start_date="2023-01-01 00:00:00",
        end_date="2023-01-10 00:00:00",
        chunking="time",
    )
    assert chunk_time == "time"
    assert url_time_forced == DEFAULT_CDS_ARCO_SFC_TIME_URL


# ==============================================================================
# 3. Obstore HTTPStore Authentication Header & Zarr Extraction
# ==============================================================================

def test_extract_cds_arco_slice_obstore_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify extract_cds_arco_slice passes Authorization Bearer header to obstore HTTPStore."""
    monkeypatch.setenv("CDSAPI_KEY", "cds-test-secret-bearer-key")

    mock_ds = xr.Dataset(
        data_vars={
            "t2m": (("time", "latitude", "longitude"), np.ones((2, 3, 3), dtype=np.float32) * 295.0),
        },
        coords={
            "time": pd.date_range("2023-01-01 00:00:00", periods=2, freq="1h"),
            "latitude": [-1.5, -1.25, -1.0],
            "longitude": [36.5, 36.75, 37.0],
        },
    )

    with patch("obstore.store.HTTPStore") as mock_http, \
         patch("zarr.storage.ObjectStore") as mock_obj, \
         patch("xarray.open_zarr", return_value=mock_ds) as mock_open:

        extractor = ERA5Extractor(use_cds_arco=True)
        box = SpatialBoundingBox(lat_min=-1.5, lat_max=-1.0, lon_min=36.5, lon_max=37.0)

        res = extractor.extract_cds_arco_slice(
            bbox=box,
            start_date="2023-01-01 00:00:00",
            end_date="2023-01-01 01:00:00",
            variables=["2m_temperature"],
            chunking="geo",
        )

        mock_http.assert_called_once()
        target_url = mock_http.call_args[0][0]
        assert target_url == DEFAULT_CDS_ARCO_SFC_GEO_URL
        headers = mock_http.call_args[1]["client_options"]["default_headers"]
        assert headers["Authorization"] == "Bearer cds-test-secret-bearer-key"
        mock_obj.assert_called_once()
        mock_open.assert_called_once()

        assert "lat" in res.coords
        assert "lon" in res.coords
        assert "time" in res.dims
        assert "2m_temperature" in res.data_vars


# ==============================================================================
# 4. Live ECMWF ARCO Endpoint Connection Test
# ==============================================================================

def test_live_ecmwf_cds_arco_obstore_connection() -> None:
    """
    CRITICAL: Validates live obstore connection to ECMWF CADL ARCO Zarr store.
    Authenticates with CDSAPI_KEY and retrieves a real data slice without crashing.
    """
    try:
        key = get_cds_api_key()
    except Exception:
        pytest.skip("No CDS credentials found in environment or ~/.cdsapirc. Skipping live network test.")

    if not key:
        pytest.skip("CDSAPI_KEY is empty. Skipping live network test.")

    extractor = ERA5Extractor(use_cds_arco=True)
    box = SpatialBoundingBox(lat_min=-1.5, lat_max=-1.0, lon_min=36.5, lon_max=37.25)

    # Execute lightweight query against ECMWF geo-chunked Zarr store
    ds = extractor.extract_cds_arco_slice(
        bbox=box,
        start_date="2023-01-01 00:00:00",
        end_date="2023-01-01 01:00:00",
        variables=["2m_temperature"],
        chunking="geo",
    )

    assert isinstance(ds, xr.Dataset)
    assert "2m_temperature" in ds.data_vars
    assert "lat" in ds.coords
    assert "lon" in ds.coords
    assert "time" in ds.dims

    # Compute a small slice to verify numeric arrays are decoded without crashing
    vals = ds["2m_temperature"].compute().values
    assert vals.size > 0
    assert not np.isnan(vals).all()
    # Physical sanity check: Kelvin temperature across Africa between 250K and 330K
    assert float(np.nanmean(vals)) > 250.0
    assert float(np.nanmean(vals)) < 330.0


# ==============================================================================
# 5. Dataset Interface Propagation Tests
# ==============================================================================

def test_extract_gridded_cds_arco_routing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify extract_gridded routes use_cds_arco=True and passes chunking parameter."""
    monkeypatch.setenv("CDSAPI_KEY", "cds-test-dummy-key")

    mock_slice = xr.Dataset(
        data_vars={"2m_temperature": (("time", "lat", "lon"), np.ones((2, 2, 2), dtype=np.float32) * 290.0)},
        coords={"time": pd.date_range("2023-01-01", periods=2, freq="1h"), "lat": [0.0, 1.0], "lon": [36.0, 37.0]},
    )

    with patch.object(ERA5Extractor, "extract_cds_arco_slice", return_value=mock_slice) as mock_arco:
        res = extract_gridded(
            source="era5",
            start_date="2023-01-01",
            end_date="2023-01-02",
            bbox=[0.0, 36.0, 1.0, 37.0],
            variables=["2m_temperature"],
            output_dir=tmp_path / "era5_arco",
            save_zarr=False,
            use_cds_arco=True,
            chunking="time",
        )
        mock_arco.assert_called_once()
        call_kwargs = mock_arco.call_args[1]
        assert call_kwargs["chunking"] == "time"
        assert isinstance(res, xr.Dataset)


def test_pull_gridded_data_cds_arco_forwarding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify pull_gridded_data forwards use_cds_arco and chunking to extract_gridded."""
    with patch("masharikiweather.dataset.extract_gridded", return_value=xr.Dataset()) as mock_extract:
        pull_gridded_data(
            source="era5",
            start_date="2023-01-01",
            end_date="2023-01-02",
            output_dir=tmp_path / "gridded",
            save_zarr=False,
            use_cds_arco=True,
            chunking="geo",
            stream="wav",
        )
        mock_extract.assert_called_once()
        kwargs = mock_extract.call_args[1]
        assert kwargs["use_cds_arco"] is True
        assert kwargs["chunking"] == "geo"
        assert kwargs["stream"] == "wav"


def test_extract_cds_arco_slice_missing_obstore_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify informative ImportError is raised when obstore is not installed."""
    import sys
    monkeypatch.setenv("CDSAPI_KEY", "cds-test-dummy-key")
    monkeypatch.setitem(sys.modules, "obstore", None)
    monkeypatch.setitem(sys.modules, "obstore.store", None)

    extractor = ERA5Extractor(use_cds_arco=True)
    box = SpatialBoundingBox(lat_min=-1.0, lat_max=1.0, lon_min=36.0, lon_max=38.0)
    with pytest.raises(ImportError, match="obstore is required for CDS ARCO Zarr loading"):
        extractor.extract_cds_arco_slice(
            bbox=box,
            start_date="2023-01-01",
            end_date="2023-01-02",
            variables=["2m_temperature"],
        )

