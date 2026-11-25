"""
Unit tests for extraction resilience, progress reporting, rate limiting,
manifest-backed caching, and pre-flight dry-run estimation.
"""

from __future__ import annotations

import builtins
import io
import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from masharikiweather.dataset import extract_gridded, extract_stations, pull_weather_data
from masharikiweather.extract.apis.base import BaseExtractor, SpatialBoundingBox
from masharikiweather.extract.apis.era5 import ERA5Extractor
from masharikiweather.extract.apis.ghcnd import GHCNExtractor
from masharikiweather.extract.apis.imerg import IMERGExtractor
from masharikiweather.extract.apis.tahmo import TAHMOExtractor
from masharikiweather.extract.apis.resilience import (
    GHCN_MAX_WORKERS_CAP,
    MAX_WORKERS_CAP,
    TAHMO_MAX_WORKERS_CAP,
    ExtractionCache,
    ExtractionEstimate,
    ExtractionManifestSummary,
    RateLimiter,
    _NoOpProgressBar,
    allocate_worker_budget,
    append_to_zarr,
    clamp_max_workers,
    estimate_extraction,
    get_available_memory_mb,
    make_progress_bar,
    retry_with_backoff,
    stream_merge_parquet,
)


# ---------------------------------------------------------------------------
# 1. Source-Adaptive Progress Bar Tests
# ---------------------------------------------------------------------------


class TestProgressBar:
    """Tests for make_progress_bar and _NoOpProgressBar."""

    @pytest.mark.parametrize(
        "source,expected_unit,expected_desc",
        [
            ("tahmo", "station", "TAHMO stations"),
            ("era5", "chunk", "ERA5 chunks"),
            ("era5_arco", "chunk", "ERA5 ARCO chunks"),
            ("imerg", "granule", "IMERG granules"),
            ("open_meteo", "tile", "Open-Meteo tiles"),
            ("openmeteo", "tile", "Open-Meteo tiles"),
            ("ghcn", "station", "GHCN stations"),
        ],
    )
    def test_source_adaptive_desc_and_unit(
        self,
        source: str,
        expected_unit: str,
        expected_desc: str,
    ) -> None:
        buf = io.StringIO()
        pbar = make_progress_bar(
            [1, 2, 3],
            source=source,
            total=3,
            show_progress=True,
            file=buf,
        )
        try:
            assert pbar.unit == expected_unit
            assert pbar.desc == expected_desc
        finally:
            pbar.close()

    def test_show_progress_false_and_initial_on_resume(self) -> None:
        pbar = make_progress_bar(
            total=10,
            initial=4,
            source="tahmo",
            show_progress=False,
        )
        try:
            assert pbar.disable is True
            assert pbar.n == 4
            assert pbar.total == 10
        finally:
            pbar.close()

    def test_noop_fallback_when_tqdm_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "tqdm.auto" or name.startswith("tqdm"):
                raise ImportError("Simulated missing tqdm")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        pbar = make_progress_bar(
            ["a", "b", "c"],
            source="imerg",
            total=3,
            initial=1,
            show_progress=False,
        )
        assert isinstance(pbar, _NoOpProgressBar)
        assert pbar.unit == "granule"
        assert pbar.desc == "IMERG granules"
        assert pbar.disable is True
        assert pbar.n == 1

        collected: List[str] = []
        with pbar as ctx:
            for item in ctx:
                collected.append(item)
            ctx.set_description("updated")
            ctx.set_postfix(cached=1)
            ctx.refresh()
            ctx.write("log line")
            ctx.update(2)

        assert collected == ["a", "b", "c"]
        assert pbar.desc == "updated"
        assert pbar.n == 6


# ---------------------------------------------------------------------------
# 2. ExtractionCache & manifest.json Checkpointing Tests
# ---------------------------------------------------------------------------


class TestExtractionCache:
    """Tests for ExtractionCache determinism, atomic I/O, resume, and thread safety."""

    def test_request_hash_determinism_and_sensitivity(self, tmp_path: Path) -> None:
        bbox = SpatialBoundingBox(lat_min=-4.0, lat_max=4.0, lon_min=33.0, lon_max=42.0)
        base_params: Dict[str, Any] = {
            "variables": ["precipitation", "temperature"],
            "start_date": "2023-01-01",
            "end_date": "2023-01-05",
            "bbox": bbox,
        }
        reordered_params: Dict[str, Any] = {
            "bbox": bbox,
            "end_date": "2023-01-05",
            "start_date": "2023-01-01",
            "variables": ["precipitation", "temperature"],
        }

        c1 = ExtractionCache(tmp_path, "tahmo", base_params)
        c2 = ExtractionCache(tmp_path, "tahmo", reordered_params)
        assert c1.request_hash == c2.request_hash

        diff_vars = dict(base_params, variables=["precipitation"])
        diff_start = dict(base_params, start_date="2023-01-02")
        diff_end = dict(base_params, end_date="2023-01-06")
        diff_bbox = dict(
            base_params,
            bbox=SpatialBoundingBox(lat_min=-5.0, lat_max=4.0, lon_min=33.0, lon_max=42.0),
        )

        hashes = {
            c1.request_hash,
            ExtractionCache(tmp_path, "tahmo", diff_vars).request_hash,
            ExtractionCache(tmp_path, "tahmo", diff_start).request_hash,
            ExtractionCache(tmp_path, "tahmo", diff_end).request_hash,
            ExtractionCache(tmp_path, "tahmo", diff_bbox).request_hash,
        }
        assert len(hashes) == 5

    def test_atomic_writes_tmp_file_does_not_mark_completed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache = ExtractionCache(tmp_path, "tahmo", {"start_date": "2023-01-01"})
        df = pd.DataFrame({"val": [1.0, 2.0]})

        def _crash_replace(src: Any, dst: Any) -> None:
            raise OSError("Simulated crash before atomic rename")

        monkeypatch.setattr(os, "replace", _crash_replace)
        with pytest.raises(OSError, match="Simulated crash"):
            cache.save_parquet("TA001", df)

        tmp_files = list(cache.cache_dir.glob("*.tmp"))
        assert len(tmp_files) == 1
        assert cache.is_done("TA001") is False

        monkeypatch.undo()
        manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
        assert "TA001" not in manifest["partitions"]

    def test_parquet_dataset_and_bytes_round_trips(self, tmp_path: Path) -> None:
        cache = ExtractionCache(tmp_path, "era5", {"start_date": "2023-01-01"})

        df = pd.DataFrame(
            {"station": ["S1", "S2"], "temp": [22.5, 24.1]},
            index=pd.Index([0, 1], name="idx"),
        )
        pq_path = cache.save_parquet("part_pq", df)
        assert pq_path.is_file()
        assert cache.is_done("part_pq") is True
        loaded_df = cache.load_parquet("part_pq")
        pd.testing.assert_frame_equal(loaded_df, df)

        ds = xr.Dataset(
            {"t2m": (("time", "lat", "lon"), np.ones((2, 2, 2), dtype=np.float32))},
            coords={"time": [0, 1], "lat": [0.0, 1.0], "lon": [35.0, 36.0]},
        )
        nc_path = cache.save_dataset("part_nc", ds, fmt="netcdf")
        assert nc_path.is_file()
        assert cache.is_done("part_nc") is True
        with cache.load_dataset("part_nc") as loaded_ds:
            xr.testing.assert_identical(loaded_ds, ds)

        payload = b"\x00\x01mashariki-raw-bytes\xff"
        raw_path = cache.save_bytes("part_bin", payload, suffix=".bin")
        assert raw_path.is_file()
        assert cache.is_done("part_bin") is True
        assert cache.get_path("part_bin") == raw_path
        assert raw_path.read_bytes() == payload

    def test_mark_empty_and_mark_failed_resume_behavior(self, tmp_path: Path) -> None:
        params = {"start_date": "2023-01-01", "end_date": "2023-01-02"}
        cache = ExtractionCache(tmp_path, "tahmo", params, resume=True)
        cache.mark_empty("EMPTY_STN")
        cache.mark_failed("BAD_STN", RuntimeError("HTTP 503"))

        resumed = ExtractionCache(tmp_path, "tahmo", params, resume=True)
        assert resumed.is_done("EMPTY_STN") is True
        assert resumed.is_done("BAD_STN") is False

        summary = resumed.summary()
        assert isinstance(summary, ExtractionManifestSummary)
        assert summary.empty == 1
        assert summary.failed == 1
        assert summary.failed_items == {"BAD_STN": "HTTP 503"}
        assert resumed.get_failed() == {"BAD_STN": "HTTP 503"}

    def test_overwrite_bypasses_cache(self, tmp_path: Path) -> None:
        params = {"start_date": "2023-01-01"}
        cache = ExtractionCache(tmp_path, "tahmo", params, resume=True)
        cache.save_parquet("TA001", pd.DataFrame({"x": [1]}))
        cache.mark_empty("TA002")
        assert cache.is_done("TA001") is True
        assert cache.is_done("TA002") is True

        overwrite_cache = ExtractionCache(tmp_path, "tahmo", params, resume=True, overwrite=True)
        assert overwrite_cache.is_done("TA001") is False
        assert overwrite_cache.is_done("TA002") is False

    def test_thread_safe_concurrent_writes(self, tmp_path: Path) -> None:
        cache = ExtractionCache(tmp_path, "tahmo", {"start_date": "2023-01-01"})

        def _worker(idx: int) -> None:
            key = f"STN_{idx:03d}"
            if idx % 5 == 0:
                cache.mark_empty(key)
            elif idx % 7 == 0:
                cache.mark_failed(key, f"err_{idx}")
            else:
                cache.save_parquet(key, pd.DataFrame({"val": [float(idx)]}))

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_worker, range(30)))

        manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
        assert len(manifest["partitions"]) == 30

        summary = cache.summary()
        assert summary.total == 30
        assert summary.completed + summary.empty + summary.failed == 30

    def test_cleanup_keep_on_failure_behavior(self, tmp_path: Path) -> None:
        params_fail = {"case": "with_failure"}
        c_fail = ExtractionCache(tmp_path, "tahmo", params_fail)
        c_fail.save_parquet("OK", pd.DataFrame({"v": [1]}))
        c_fail.mark_failed("ERR", "timeout")

        assert c_fail.cleanup(keep_on_failure=True) is False
        assert c_fail.cache_dir.exists() is True

        assert c_fail.cleanup(keep_on_failure=False) is True
        assert c_fail.cache_dir.exists() is False

        params_ok = {"case": "clean"}
        c_ok = ExtractionCache(tmp_path, "tahmo", params_ok)
        c_ok.save_parquet("OK", pd.DataFrame({"v": [1]}))
        assert c_ok.cleanup(keep_on_failure=True) is True
        assert c_ok.cache_dir.exists() is False


# ---------------------------------------------------------------------------
# 3. RateLimiter & retry_with_backoff Tests
# ---------------------------------------------------------------------------


class TestRateLimiterAndRetry:
    """Tests for RateLimiter and retry_with_backoff with mocked time.sleep."""

    def test_rate_limiter_enforces_spacing_after_burst(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = {"t": 100.0}
        sleeps: List[float] = []

        def _fake_monotonic() -> float:
            return clock["t"]

        def _fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock["t"] += seconds

        monkeypatch.setattr("masharikiweather.extract.apis.resilience.time.monotonic", _fake_monotonic)
        monkeypatch.setattr("masharikiweather.extract.apis.resilience.time.sleep", _fake_sleep)

        limiter = RateLimiter(requests_per_second=2.0, burst=2)
        assert limiter.acquire() == 0.0
        assert limiter.acquire() == 0.0
        assert sleeps == []

        waited = limiter.acquire()
        assert waited == pytest.approx(0.5)
        assert sleeps == [pytest.approx(0.5)]

        with limiter:
            pass
        assert len(sleeps) == 2
        assert sleeps[1] == pytest.approx(0.5)

    def test_retry_with_backoff_exponential_and_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: List[float] = []
        retries_seen: List[int] = []
        attempts = {"count": 0}

        monkeypatch.setattr("masharikiweather.extract.apis.resilience.time.sleep", sleeps.append)

        def _flaky() -> str:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ConnectionError(f"transient #{attempts['count']}")
            return "ok"

        result = retry_with_backoff(
            _flaky,
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            jitter=False,
            retry_on=(ConnectionError,),
            on_retry=lambda attempt, exc, delay: retries_seen.append(attempt),
        )

        assert result == "ok"
        assert attempts["count"] == 3
        assert retries_seen == [1, 2]
        assert sleeps == [1.0, 2.0]

    def test_retry_with_backoff_honors_retry_after(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: List[float] = []
        monkeypatch.setattr("masharikiweather.extract.apis.resilience.time.sleep", sleeps.append)

        class _RateLimitError(Exception):
            def __init__(self, retry_after: str) -> None:
                super().__init__("429 Too Many Requests")
                self.response = SimpleNamespace(headers={"Retry-After": retry_after})

        calls = {"n": 0}

        def _call() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise _RateLimitError("12")
            return "recovered"

        out = retry_with_backoff(
            _call,
            max_retries=2,
            base_delay=1.0,
            max_delay=60.0,
            jitter=False,
            retry_on=(_RateLimitError,),
        )
        assert out == "recovered"
        assert sleeps == [12.0]

    def test_retry_with_backoff_stops_immediately_on_non_retryable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sleeps: List[float] = []
        monkeypatch.setattr("masharikiweather.extract.apis.resilience.time.sleep", sleeps.append)

        calls = {"n": 0}

        def _fatal() -> None:
            calls["n"] += 1
            raise ValueError("Invalid parameter")

        with pytest.raises(ValueError, match="Invalid parameter"):
            retry_with_backoff(
                _fatal,
                max_retries=3,
                base_delay=1.0,
                retry_on=(TimeoutError,),
            )

        assert calls["n"] == 1
        assert sleeps == []


# ---------------------------------------------------------------------------
# 4. Pre-Flight Dry-Run & BaseExtractor Helper Tests
# ---------------------------------------------------------------------------


class TestDryRunAndBaseHelpers:
    """Tests for estimate_extraction, dry_run flags, and BaseExtractor helpers."""

    @pytest.mark.parametrize(
        "source,expected_unit",
        [
            ("tahmo", "station"),
            ("era5", "chunk"),
            ("era5_arco", "chunk"),
            ("imerg", "granule"),
            ("open_meteo", "tile"),
        ],
    )
    def test_estimate_extraction_unit_names_and_metrics(
        self,
        source: str,
        expected_unit: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _no_network(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("Network access is forbidden during dry-run estimation")

        monkeypatch.setattr(socket, "create_connection", _no_network)

        est = estimate_extraction(
            source=source,
            start_date="2023-01-01",
            end_date="2023-01-03",
            bbox=[-1.5, 36.0, 1.5, 38.0],
            variables=["precipitation", "temperature"],
        )
        assert isinstance(est, ExtractionEstimate)
        assert est.unit_name == expected_unit
        assert est.num_units > 0
        assert est.estimated_requests > 0
        assert est.estimated_size_mb > 0.0
        assert est.summary_text != ""
        assert expected_unit in est.summary_text

    def test_extract_stations_dry_run_no_network_no_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _no_network(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("Network access is forbidden during dry_run=True")

        monkeypatch.setattr(socket, "create_connection", _no_network)
        out_dir = tmp_path / "stations_out_should_not_exist"

        est = extract_stations(
            source="tahmo",
            start_date="2023-01-01",
            end_date="2023-01-05",
            bbox=[-2.0, 35.0, 2.0, 39.0],
            variables=["pr"],
            output_dir=out_dir,
            stations=["TA00001", "TA00002", "TA00003"],
            dry_run=True,
        )

        assert isinstance(est, ExtractionEstimate)
        assert est.unit_name == "station"
        assert est.num_units == 3
        assert est.estimated_requests > 0
        assert est.estimated_size_mb > 0.0
        assert out_dir.exists() is False

    def test_extract_gridded_dry_run_no_network_no_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _no_network(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("Network access is forbidden during dry_run=True")

        monkeypatch.setattr(socket, "create_connection", _no_network)
        out_dir = tmp_path / "gridded_out_should_not_exist"

        est = extract_gridded(
            source="era5",
            start_date="2023-01-01",
            end_date="2023-01-10",
            bbox=[-2.0, 35.0, 2.0, 39.0],
            variables=["2m_temperature", "total_precipitation"],
            output_dir=out_dir,
            chunk_days=5,
            dry_run=True,
        )

        assert isinstance(est, ExtractionEstimate)
        assert est.unit_name == "chunk"
        assert est.num_units == 2
        assert est.estimated_requests == 2
        assert est.estimated_size_mb > 0.0
        assert out_dir.exists() is False

    def test_base_extractor_resilience_helpers(self, tmp_path: Path) -> None:
        class _DummyExtractor(BaseExtractor):
            extractor_name = "imerg"

            def extract(self, *args: Any, **kwargs: Any) -> Any:
                return None

        ext = _DummyExtractor()
        buf = io.StringIO()
        pbar = ext.progress_bar(total=5, show_progress=True, file=buf)
        try:
            assert pbar.unit == "granule"
            assert pbar.disable is False
        finally:
            pbar.close()

        cache = ext.create_cache(tmp_path, {"start_date": "2023-01-01"})
        assert isinstance(cache, ExtractionCache)
        assert cache.source == "imerg"

        limiter = ext.create_rate_limiter(requests_per_second=5.0, burst=2)
        assert isinstance(limiter, RateLimiter)

        est = ext.estimate_extraction(
            start_date="2023-01-01",
            end_date="2023-01-01",
            bbox=[-1.0, 36.0, 1.0, 38.0],
        )
        assert isinstance(est, ExtractionEstimate)
        assert est.unit_name == "granule"
        assert est.num_units == 48


# ---------------------------------------------------------------------------
# 5. Extractor Integration Tests (Mocked Network)
# ---------------------------------------------------------------------------


class TestExtractorIntegration:
    """Mocked integration tests for wired extractors."""

    def test_tahmo_extract_failure_retention_and_resume(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ext = TAHMOExtractor(
            output_dir=tmp_path,
            api_key="test_key",
            api_secret="test_secret",
            request_delay=0.0,
            max_workers=1,
        )

        called_run1: List[str] = []

        def _make_df(val: float) -> pd.DataFrame:
            idx = pd.date_range("2023-01-01", periods=2, freq="1h", name="time")
            return pd.DataFrame({"pr": [val, val + 1.0]}, index=idx)

        def _fake_extract_run1(
            station_code: str,
            start_date: Any,
            end_date: Any,
            variables: Any = None,
        ) -> pd.DataFrame:
            called_run1.append(station_code)
            if station_code == "TA00003":
                raise ValueError("Persistent station error")
            return _make_df(1.0 if station_code == "TA00001" else 2.0)

        monkeypatch.setattr(ext, "extract_station", _fake_extract_run1)

        res1 = ext.extract(
            start_date="2023-01-01",
            end_date="2023-01-02",
            stations=["TA00001", "TA00002", "TA00003"],
            parquet_layout="station",
            generate_metadata=False,
            show_progress=False,
            resume=True,
            keep_cache=False,
        )

        assert sorted(called_run1) == ["TA00001", "TA00002", "TA00003"]
        assert "TA00001" in res1 and "TA00002" in res1
        assert "TA00003" not in res1

        manifests = list((tmp_path / ".extract_cache" / "tahmo").glob("*/manifest.json"))
        assert len(manifests) == 1
        manifest_data = json.loads(manifests[0].read_text(encoding="utf-8"))
        assert manifest_data["partitions"]["TA00001"]["status"] == "completed"
        assert manifest_data["partitions"]["TA00002"]["status"] == "completed"
        assert manifest_data["partitions"]["TA00003"]["status"] == "failed"

        called_run2: List[str] = []

        def _fake_extract_run2(
            station_code: str,
            start_date: Any,
            end_date: Any,
            variables: Any = None,
        ) -> pd.DataFrame:
            called_run2.append(station_code)
            return _make_df(3.0)

        monkeypatch.setattr(ext, "extract_station", _fake_extract_run2)

        res2 = ext.extract(
            start_date="2023-01-01",
            end_date="2023-01-02",
            stations=["TA00001", "TA00002", "TA00003"],
            parquet_layout="station",
            generate_metadata=False,
            show_progress=False,
            resume=True,
            keep_cache=False,
        )

        assert called_run2 == ["TA00003"]
        assert set(res2.keys()) == {"TA00001", "TA00002", "TA00003"}
        assert list((tmp_path / ".extract_cache" / "tahmo").glob("*/manifest.json")) == []

    def test_extractors_dry_run_no_network(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _no_network(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("Network access is forbidden during dry_run=True")

        monkeypatch.setattr(socket, "create_connection", _no_network)
        bbox = SpatialBoundingBox(lat_min=-2.0, lat_max=2.0, lon_min=35.0, lon_max=39.0)

        tahmo_est = TAHMOExtractor(
            output_dir=tmp_path / "tahmo",
            api_key="k",
            api_secret="s",
        ).extract(
            "2023-01-01",
            "2023-01-03",
            stations=["TA00001", "TA00002"],
            dry_run=True,
        )
        assert isinstance(tahmo_est, ExtractionEstimate)
        assert tahmo_est.source == "tahmo"
        assert tahmo_est.unit_name == "station"
        assert tahmo_est.num_units == 2

        era5_est = ERA5Extractor(output_dir=tmp_path / "era5").extract(
            "2023-01-01",
            "2023-01-05",
            bbox=bbox,
            variables=["2m_temperature"],
            dry_run=True,
        )
        assert isinstance(era5_est, ExtractionEstimate)
        assert era5_est.source == "era5"
        assert era5_est.unit_name == "chunk"
        assert era5_est.num_units == 1

        imerg_est = IMERGExtractor(output_dir=tmp_path / "imerg").extract(
            "2023-01-01",
            "2023-01-02",
            bbox=bbox,
            dry_run=True,
        )
        assert isinstance(imerg_est, ExtractionEstimate)
        assert imerg_est.source == "imerg"
        assert imerg_est.unit_name == "granule"
        assert imerg_est.num_units == 96

        ghcn_est = GHCNExtractor(output_dir=tmp_path / "ghcn").extract(
            "2023-01-01",
            "2023-01-03",
            stations=["KE000063740"],
            dry_run=True,
        )
        assert isinstance(ghcn_est, ExtractionEstimate)
        assert ghcn_est.source == "ghcnd"
        assert ghcn_est.unit_name == "station"
        assert ghcn_est.num_units == 1


# ---------------------------------------------------------------------------
# 6. Max Workers Clamping and Multi-Source Kwarg Forwarding Tests
# ---------------------------------------------------------------------------


class TestMaxWorkersClampAndKwargForwarding:
    """Validate max_workers clamping and strict kwarg filtering across multi-source pulls."""

    def test_clamp_max_workers_bounds(self) -> None:
        assert MAX_WORKERS_CAP == 16
        assert TAHMO_MAX_WORKERS_CAP == 16
        assert GHCN_MAX_WORKERS_CAP == 24
        assert clamp_max_workers(None) == 4
        assert clamp_max_workers(8) == 8
        assert clamp_max_workers(48, source="imerg") == 48

        with pytest.warns(UserWarning, match="below minimum"):
            assert clamp_max_workers(0) == 1

        with pytest.warns(UserWarning, match="exceeds maximum allowed"):
            assert clamp_max_workers(200, source="tahmo") == 16

        with pytest.warns(UserWarning, match="exceeds maximum allowed"):
            assert clamp_max_workers(200, source="ghcnd") == 24

    def test_pull_weather_data_filters_kwargs_and_clamps_max_workers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        recorded: Dict[str, Any] = {}

        class _StubTAHMOExtractor:
            def __init__(
                self,
                output_dir: Any,
                api_key: Any = None,
                api_secret: Any = None,
                max_workers: int = 4,
                show_progress: bool = True,
                resume: bool = True,
                overwrite: bool = False,
                keep_cache: bool = False,
            ) -> None:
                recorded["tahmo_init_workers"] = max_workers
                self.output_dir = Path(output_dir)

            def extract(
                self,
                start_date: Any,
                end_date: Any,
                bbox: Any = None,
                variables: Any = None,
                save_parquet: bool = True,
                max_workers: int = 4,
                show_progress: bool = True,
                resume: bool = True,
                overwrite: bool = False,
                keep_cache: bool = False,
            ) -> Dict[str, Path]:
                recorded["tahmo_extract_workers"] = max_workers
                p = self.output_dir / "2023.parquet"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"PAR1")
                return {"2023": p}

        class _StubGHCNExtractor:
            def __init__(
                self,
                output_dir: Any,
                resolution: str = "daily",
                show_progress: bool = True,
                resume: bool = True,
                overwrite: bool = False,
                keep_cache: bool = False,
            ) -> None:
                recorded["ghcn_init_called"] = True
                self.output_dir = Path(output_dir)

            def extract(
                self,
                start_date: Any,
                end_date: Any,
                bbox: Any = None,
                variables: Any = None,
                save_parquet: bool = True,
                show_progress: bool = True,
                resume: bool = True,
                overwrite: bool = False,
                keep_cache: bool = False,
            ) -> Dict[str, Path]:
                recorded["ghcn_extract_called"] = True
                p = self.output_dir / "ghcn.parquet"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"PAR1")
                return {"ghcn": p}

        class _StubERA5Extractor:
            def __init__(
                self,
                output_dir: Any,
                use_legacy: Any = None,
                use_cds_arco: Any = None,
                use_google_arco: Any = None,
                data_format: str = "netcdf4",
                prefer_arco: Any = None,
                force_cds: Any = None,
                cds_key: Any = None,
                cds_url: Any = None,
                cds_arco_url: Any = None,
                chunking: Any = None,
                stream: Any = None,
                show_progress: bool = True,
                resume: bool = True,
                overwrite: bool = False,
                keep_cache: bool = False,
            ) -> None:
                recorded["era5_init_called"] = True
                self.output_dir = Path(output_dir)

            def extract(
                self,
                start_date: Any,
                end_date: Any,
                bbox: Any = None,
                variables: Any = None,
                levels: Any = None,
                save_zarr: bool = True,
                use_legacy: Any = None,
                use_cds_arco: Any = None,
                use_google_arco: Any = None,
                data_format: str = "netcdf4",
                force_cds: Any = None,
                prefer_arco: Any = None,
                chunk_days: int = 5,
                chunking: Any = None,
                stream: Any = None,
                cds_key: Any = None,
                cds_url: Any = None,
                cds_arco_url: Any = None,
                show_progress: bool = True,
                resume: bool = True,
                overwrite: bool = False,
                keep_cache: bool = False,
            ) -> Path:
                recorded["era5_extract_called"] = True
                p = self.output_dir / "era5.zarr"
                p.mkdir(parents=True, exist_ok=True)
                return p

        import masharikiweather.extract.apis.tahmo as tahmo_mod
        import masharikiweather.extract.apis.ghcnd as ghcnd_mod
        import masharikiweather.extract.apis.era5 as era5_mod

        monkeypatch.setattr(tahmo_mod, "TAHMOExtractor", _StubTAHMOExtractor)
        monkeypatch.setattr(ghcnd_mod, "GHCNExtractor", _StubGHCNExtractor)
        monkeypatch.setattr(era5_mod, "ERA5Extractor", _StubERA5Extractor)

        with pytest.warns(UserWarning, match="exceeds maximum allowed"):
            res = pull_weather_data(
                start_date="2023-01-01",
                end_date="2023-01-02",
                bbox=[-2.0, 35.0, 2.0, 39.0],
                station_source=["tahmo", "ghcn"],
                gridded_source=["era5"],
                output_dir=tmp_path,
                max_workers=200,
                show_progress=False,
            )

        assert recorded["tahmo_init_workers"] == 16
        assert recorded["tahmo_extract_workers"] == 16
        assert recorded["ghcn_init_called"] is True
        assert recorded["ghcn_extract_called"] is True
        assert recorded["era5_init_called"] is True
        assert recorded["era5_extract_called"] is True
        assert res["stations"]["tahmo"] is not None
        assert res["stations"]["ghcn"] is not None
        assert res["gridded"]["era5"] is not None

        # Sequential mode (concurrent_sources=False) gives TAHMO the full clamped cap (16)
        recorded.clear()
        with pytest.warns(UserWarning, match="exceeds maximum allowed"):
            pull_weather_data(
                start_date="2023-01-01",
                end_date="2023-01-02",
                bbox=[-2.0, 35.0, 2.0, 39.0],
                station_source=["tahmo", "ghcn"],
                gridded_source=["era5"],
                output_dir=tmp_path / "seq",
                max_workers=200,
                concurrent_sources=False,
                show_progress=False,
            )
        assert recorded["tahmo_init_workers"] == 16
        assert recorded["tahmo_extract_workers"] == 16

        # Direct extract_stations / extract_gridded calls with max_workers=200 must also not raise TypeError
        ghcn_res = extract_stations(
            source="ghcn",
            start_date="2023-01-01",
            end_date="2023-01-02",
            output_dir=tmp_path / "direct_ghcn",
            max_workers=200,
            show_progress=False,
        )
        assert isinstance(ghcn_res, dict)

        era5_res = extract_gridded(
            source="era5",
            start_date="2023-01-01",
            end_date="2023-01-02",
            output_dir=tmp_path / "direct_era5",
            max_workers=200,
            show_progress=False,
        )
        assert isinstance(era5_res, Path)


# ---------------------------------------------------------------------------
# 7. RAM-Safe Streaming and Worker Budget Tests
# ---------------------------------------------------------------------------


class TestRamSafeStreamingAndBudget:
    """Validate memory query, worker budget allocation, Parquet streaming merge, and Zarr append."""

    def test_get_available_memory_mb(self) -> None:
        mem = get_available_memory_mb()
        if mem is not None:
            assert mem > 0.0

    def test_allocate_worker_budget(self) -> None:
        outer, per_src = allocate_worker_budget(3, 200)
        assert (outer, per_src) == (3, 66)

        with pytest.warns(UserWarning, match="exceeds maximum allowed"):
            outer_capped, per_capped = allocate_worker_budget(3, 200, max_cap=MAX_WORKERS_CAP)
        assert (outer_capped, per_capped) == (3, 5)
        assert outer_capped * per_capped <= MAX_WORKERS_CAP

        outer_1, per_1 = allocate_worker_budget(1, 8)
        assert (outer_1, per_1) == (1, 8)

    def test_stream_merge_parquet_heterogeneous_schemas(self, tmp_path: Path) -> None:
        p1 = tmp_path / "s1.parquet"
        p2 = tmp_path / "s2.parquet"
        pd.DataFrame({"station_id": ["S1", "S1"], "te": [21.0, 22.0]}).to_parquet(p1)
        pd.DataFrame({"station_id": ["S2"], "pr": [1.5]}).to_parquet(p2)

        out = tmp_path / "merged.parquet"
        merged_path = stream_merge_parquet([p1, p2], out, batch_size=1, delete_inputs=True)
        assert merged_path == out
        assert out.is_file()
        assert not p1.exists()
        assert not p2.exists()

        df = pd.read_parquet(out)
        assert len(df) == 3
        assert set(["station_id", "te", "pr"]).issubset(df.columns)

    def test_append_to_zarr(self, tmp_path: Path) -> None:
        store = tmp_path / "test.zarr"
        t1 = pd.date_range("2023-01-01", periods=2, freq="h")
        t2 = pd.date_range("2023-01-01T02:00:00", periods=2, freq="h")
        ds1 = xr.Dataset(
            {"t2m": (("time", "lat"), np.ones((2, 2)))},
            coords={"time": t1, "lat": [0.0, 1.0]},
        )
        ds2 = xr.Dataset(
            {"t2m": (("time", "lat"), np.full((2, 2), 2.0))},
            coords={"time": t2, "lat": [0.0, 1.0]},
        )

        append_to_zarr(ds1, store)
        append_to_zarr(ds2, store)

        opened = xr.open_zarr(store)
        assert opened.sizes["time"] == 4
        assert float(opened["t2m"].isel(time=0, lat=0).values) == 1.0
        assert float(opened["t2m"].isel(time=3, lat=0).values) == 2.0


def test_pull_weather_data_extracts_only_requested_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify pull_weather_data does not pull default era5/tahmo when omitted."""
    called: List[str] = []

    def _fake_station(**kwargs: Any) -> Dict[str, Path]:
        called.append(f"station:{kwargs['source']}")
        return {}

    def _fake_gridded(**kwargs: Any) -> Path:
        called.append(f"gridded:{kwargs['source']}")
        return tmp_path / "out.zarr"

    monkeypatch.setattr("masharikiweather.dataset.pull_station_data", _fake_station)
    monkeypatch.setattr("masharikiweather.dataset.pull_gridded_data", _fake_gridded)

    with pytest.raises(ValueError, match="station_source"):
        pull_weather_data("2023-01-01", "2023-01-02", output_dir=tmp_path)

    res_st = pull_weather_data("2023-01-01", "2023-01-02", station_source=["ghcn"], output_dir=tmp_path)
    assert called == ["station:ghcn"]
    assert res_st["gridded"] == {}

    called.clear()
    res_gr = pull_weather_data("2023-01-01", "2023-01-02", gridded_source=["imerg"], output_dir=tmp_path)
    assert called == ["gridded:imerg"]
    assert res_gr["stations"] == {}



