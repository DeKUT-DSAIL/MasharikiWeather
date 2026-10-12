"""
Tests for GHCN API Wrapper, Hourly Engine, and Haversine Alignment.
Compatible with standard library unittest and pytest.
"""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd
from masharikiweather.ghcn_apiwrapper import (
    GHCN,
    GHCNDaily,
    GHCNHourly,
    GHCNDailyLexicon,
    GHCNHourlyLexicon,
    DEFAULT_KENYA_BOUNDS,
    haversine_distance,
    align_to_reference_grid,
    find_nearest_stations,
    find_nearest_grid_point,
)


class TestGHCNApiWrapper(unittest.TestCase):
    def test_lexicons(self):
        """Verify daily and hourly element mappings and conversions."""
        # Daily
        tmax_elem, tmax_func = GHCNDailyLexicon["t2m_max"]
        self.assertEqual(tmax_elem, "TMAX")
        self.assertEqual(tmax_func(250), 25.0)

        tp_elem, tp_func = GHCNDailyLexicon["tp"]
        self.assertEqual(tp_elem, "PRCP")
        self.assertEqual(tp_func(150), 15.0)

        # Hourly
        t2m_elem, t2m_func = GHCNHourlyLexicon["t2m"]
        self.assertEqual(t2m_elem, "temperature")
        self.assertEqual(t2m_func(18.5), 18.5)

    def test_haversine_distance(self):
        """Test Haversine distance between known coordinates (Nairobi to Mombasa ~440km)."""
        nairobi_lat, nairobi_lon = np.array([-1.286389]), np.array([36.817223])
        mombasa_lat, mombasa_lon = np.array([-4.05466]), np.array([39.66359])

        dist = haversine_distance(nairobi_lat, nairobi_lon, mombasa_lat, mombasa_lon)
        self.assertTrue(430.0 < dist[0] < 460.0, f"Distance {dist[0]} km out of expected range")

    def test_align_to_reference_grid(self):
        """Verify alignment reports nearest label with haversine and euclidean metrics."""
        source_df = pd.DataFrame([
            {"station": "KE001", "lat": -1.28, "lon": 36.82},  # Nairobi
            {"station": "KE002", "lat": -0.05, "lon": 37.65},  # Meru
        ])

        target_meta = pd.DataFrame([
            {"lat_lon_label": "TA_NRB", "latitude": -1.29, "longitude": 36.81},
            {"lat_lon_label": "TA_MRU", "latitude": -0.04, "longitude": 37.64},
            {"lat_lon_label": "TA_MBA", "latitude": -4.05, "longitude": 39.66},
        ])

        # 1. Haversine
        aligned_h = align_to_reference_grid(source_df, target_meta, target_name="tahmo", metric="haversine")
        self.assertEqual(len(aligned_h), 2)
        self.assertEqual(aligned_h.loc[aligned_h["ghcn_station"] == "KE001", "tahmo_nearest_label"].values[0], "TA_NRB")
        self.assertIn("distance_km", aligned_h.columns)

        # 2. Euclidean
        aligned_e = align_to_reference_grid(source_df, target_meta, target_name="tahmo", metric="euclidean")
        self.assertEqual(len(aligned_e), 2)
        self.assertIn("distance_deg", aligned_e.columns)

    def test_find_nearest_grid_point(self):
        """Verify gridded product nearest coordinate search (matching Stage 6 notebook helper)."""
        gridded_df = pd.DataFrame({
            "-1.280_36.810": [1.2, 0.0, 3.4],
            "0.500_37.000": [0.0, 0.0, 0.1],
        }, index=pd.date_range("2024-01-01", periods=3))

        series, col = find_nearest_grid_point(-1.282, 36.812, gridded_df, "era5")
        self.assertEqual(col, "-1.280_36.810")
        self.assertIsNotNone(series)
        self.assertEqual(len(series), 3)

    def test_pipeline_methods(self):
        """Test pipeline location mappings and direct .align() method."""
        meta_df = pd.DataFrame({
            "ID": ["KE001", "KE002"],
            "LAT": [-1.283, -0.051],
            "LON": [36.817, 37.652],
            "ELEV": [1600.0, 1200.0],
            "STATE": ["", ""],
            "NAME": ["NAIROBI", "MERU"],
            "GSN": ["", ""],
            "HCN": ["", ""],
            "WMO": ["", ""],
        })

        with tempfile.TemporaryDirectory() as tmp_dir:
            meta_path = os.path.join(tmp_dir, "ghcnd-stations.txt")
            lines = [
                f"{row['ID']:<11}{row['LAT']:>8.4f} {row['LON']:>9.4f} {row['ELEV']:>6.1f} {row['STATE']:<2} {row['NAME']:<30} {row['GSN']:<3} {row['HCN']:<3} {row['WMO']:<5}"
                for _, row in meta_df.iterrows()
            ]
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            # Mock country definitions and inventory so tests don't make remote HTTP requests
            with open(os.path.join(tmp_dir, "ghcnd-countries.txt"), "w", encoding="utf-8") as f:
                f.write("KE Kenya\n")

            pd.DataFrame(columns=["ID", "first_year", "last_year", "data_start_year", "data_end_year", "status", "data_coverage"]).to_parquet(
                os.path.join(tmp_dir, "ghcnd_station_inventory.parquet"), index=False
            )

            pipeline = GHCN(resolution="daily", stations=["KE001", "KE002"], cache_dir=tmp_dir)

            # Test get_location_mappings()
            mappings = pipeline.get_location_mappings()
            self.assertEqual(mappings.get("KE001"), "-1.283_36.817")
            self.assertEqual(mappings.get("KE002"), "-0.051_37.652")

            # Test pipeline.align()
            target_meta = pd.DataFrame([
                {"lat_lon_label": "TA001", "latitude": -1.285, "longitude": 36.815},
            ])
            aligned = pipeline.align(target_meta, target_name="tahmo")
            self.assertEqual(len(aligned), 2)
            self.assertEqual(aligned.loc[aligned["ghcn_station"] == "KE001", "tahmo_nearest_label"].values[0], "TA001")

    def test_to_wide_conversion(self):
        """Verify converting Earth2Studio long observations to wide matrices."""
        from masharikiweather.ghcn_apiwrapper import to_wide

        long_df = pd.DataFrame([
            {"time": "2026-01-01", "station": "KE001", "lat_lon": "-1.28_36.82", "variable": "tp", "observation": 12.5},
            {"time": "2026-01-01", "station": "KE002", "lat_lon": "-0.05_37.65", "variable": "tp", "observation": 0.0},
            {"time": "2026-01-02", "station": "KE001", "lat_lon": "-1.28_36.82", "variable": "tp", "observation": 5.0},
        ])
        long_df["time"] = pd.to_datetime(long_df["time"])

        # Wide by station
        wide_station = to_wide(long_df, column_key="station")
        self.assertEqual(wide_station.shape, (2, 2))
        self.assertEqual(wide_station.loc[pd.to_datetime("2026-01-01"), "KE001"], 12.5)

        # Wide by lat_lon
        wide_latlon = to_wide(long_df, column_key="lat_lon")
        self.assertIn("-1.28_36.82", wide_latlon.columns)

        # Variable specific dict
        dict_wide = to_wide(long_df, aggfunc={"tp": "sum"})
        self.assertIn("tp", dict_wide)

    def test_qc_sanitization(self):
        """Test physical plausibility sanitization filters out corruptions."""
        from masharikiweather.ghcn_apiwrapper import clean_hourly_dataframe, clean_daily_dataframe

        # Corrupted hourly data (similar to Nairobi METAR glitches)
        hourly_raw = pd.DataFrame([
            {"temperature": -18.3, "dew_point_temperature": 14.1, "wind_speed": 63.3},  # Sign-flipped temp & wind spike
            {"temperature": 22.0, "dew_point_temperature": 42.0, "wind_speed": 5.0},    # Impossible dew point
            {"temperature": 25.0, "dew_point_temperature": 16.0, "wind_speed": 3.5},    # Valid observation
        ])
        cleaned_hourly = clean_hourly_dataframe(hourly_raw, apply_qc=True)

        # Corrupted temp (-18.3 with tropical dew point) should be dropped/nan
        self.assertTrue(pd.isna(cleaned_hourly.loc[0, "temperature"]))
        # Corrupted dew point (42.0 > 22.0) should be dropped/nan
        self.assertTrue(pd.isna(cleaned_hourly.loc[1, "dew_point_temperature"]))
        # Wind spike (63.3 > 45.0) should be dropped/nan
        self.assertTrue(pd.isna(cleaned_hourly.loc[0, "wind_speed"]))
        # Valid row intact
        self.assertEqual(cleaned_hourly.loc[2, "temperature"], 25.0)
        self.assertEqual(cleaned_hourly.loc[2, "dew_point_temperature"], 16.0)

        # Daily QC
        daily_raw = pd.DataFrame([
            {"variable": "tp", "observation": -9999.0},
            {"variable": "tp", "observation": 15.0},
            {"variable": "t2m_max", "observation": 95.0},  # Impossible 95C
        ])
        cleaned_daily = clean_daily_dataframe(daily_raw, apply_qc=True)
        self.assertEqual(len(cleaned_daily), 1)
        self.assertEqual(cleaned_daily.iloc[0]["observation"], 15.0)

    def test_relative_humidity(self):
        """Verify Magnus formula relative humidity calculation."""
        from masharikiweather.ghcn_apiwrapper import calculate_relative_humidity
        temp = pd.Series([20.0, 30.0])
        dew = pd.Series([20.0, 15.0])
        rh = calculate_relative_humidity(temp, dew)
        # When temp == dew point, RH should be 100%
        self.assertAlmostEqual(rh.iloc[0], 100.0, places=1)
        # 30C temp, 15C dew point -> approx 40-42% RH
        self.assertTrue(39.0 <= rh.iloc[1] <= 44.0)

    def test_variable_resolution(self):
        """Test case-insensitive variable alias resolution."""
        from masharikiweather.ghcn_apiwrapper import resolve_daily_variable, resolve_hourly_variable

        elem, conv, can = resolve_daily_variable("Maximum_Temperature")
        self.assertEqual(elem, "TMAX")
        self.assertEqual(conv(300), 30.0)

        elem, conv, can = resolve_daily_variable("RAIN")
        self.assertEqual(elem, "PRCP")

        col, conv, can = resolve_hourly_variable("Dew_Point")
        self.assertEqual(col, "dew_point_temperature")

        col, conv, can = resolve_hourly_variable("RH")
        self.assertEqual(can, "rh")


    def test_get_station_series_safe(self):
        """Verify get_station_series safely extracts series and prevents KeyError on empty/missing stations."""
        from masharikiweather.ghcn_apiwrapper import get_station_series

        # Empty DataFrame
        empty_df = pd.DataFrame()
        s_empty = get_station_series(empty_df, station="BC007916970", variable="rain")
        self.assertTrue(s_empty.empty)

        # MultiIndex columns wide df
        cols = pd.MultiIndex.from_tuples([("NIM00065201", "rain"), ("NIM00065201", "tmax")])
        df_multi = pd.DataFrame([[12.5, 29.0]], index=pd.to_datetime(["2025-07-01"]), columns=cols)
        s_rain = get_station_series(df_multi, station="NIM00065201", variable="rain")
        self.assertEqual(len(s_rain), 1)
        self.assertEqual(s_rain.iloc[0], 12.5)

        # Non-existent station in MultiIndex df (does not raise KeyError)
        s_missing = get_station_series(df_multi, station="BC007916970", variable="rain")
        self.assertTrue(s_missing.empty)


    def test_check_station_coverage(self):
        """Verify check_station_coverage detects active vs decommissioned stations."""
        from masharikiweather.ghcn_apiwrapper import check_station_coverage

        # BC000068032 is active up to 2025; BC007916970 stopped reporting in 1989
        cov = check_station_coverage(
            stations=["BC000068032", "BC007916970"],
            variables=["rain"],
            start_date="2024-07-01",
            end_date="2025-08-24",
        )
        self.assertEqual(len(cov), 2)
        bc_active = cov[cov["station"] == "BC000068032"].iloc[0]
        bc_hist = cov[cov["station"] == "BC007916970"].iloc[0]

        self.assertTrue(bc_active["is_reporting_in_window"])
        self.assertEqual(bc_active["validity_status"], "Valid")
        self.assertIn("1921", bc_active["data_coverage"])

        self.assertFalse(bc_hist["is_reporting_in_window"])
        self.assertIn("Historical", bc_hist["validity_status"])
        self.assertIn("1989", bc_hist["data_coverage"])

    def test_variable_activeness_and_lifecycle(self):
        """Verify independent per-variable activeness tracking across stations."""
        from masharikiweather.ghcn_apiwrapper import (
            check_variable_activeness,
            get_stations_variable_activeness,
            GHCNDaily,
        )

        # Pandamatenga (BCM00068030): rain ended 2023, while tmax active thru 2026
        # Maun (BC000068032): both rain and tmax active thru 2025
        df_vars = check_variable_activeness(stations=["BCM00068030", "BC000068032"])
        self.assertEqual(len(df_vars), 2)

        panda = df_vars[df_vars["station"] == "BCM00068030"].iloc[0]
        maun = df_vars[df_vars["station"] == "BC000068032"].iloc[0]

        # Pandamatenga has ended rain, active tmax
        self.assertEqual(panda["rain_status"], "Ended 2023")
        self.assertIn("Active", panda["tmax_status"])
        self.assertIn("tmax", panda["active_variables"])
        self.assertIn("rain (2023)", panda["ended_variables"])

        # Maun has active rain and active tmax
        self.assertIn("Active", maun["rain_status"])
        self.assertIn("Active", maun["tmax_status"])

        # Test validate_variable_activeness for query window 2024-07-01 to 2025-08-24
        pipeline = GHCNDaily(stations=["BCM00068030"])
        diag = pipeline.validate_variable_activeness(
            variables=["rain", "tmax", "tmin"],
            start_date="2024-07-01",
            end_date="2025-08-24",
        )
        self.assertEqual(len(diag), 3)
        rain_diag = diag[diag["variable"] == "rain"].iloc[0]
        tmax_diag = diag[diag["variable"] == "tmax"].iloc[0]

        # Rain should be flagged as decommissioned prior to query window
        self.assertFalse(rain_diag["is_reporting_in_window"])
        self.assertIn("Decommissioned", rain_diag["query_status"])

        # Tmax should be flagged as active
        self.assertTrue(tmax_diag["is_reporting_in_window"])
        self.assertIn("Active", tmax_diag["query_status"])

    def test_african_climate_variables(self):
        """Verify extended variables relevant to African climates resolve and clean properly."""
        from masharikiweather.ghcn_apiwrapper import resolve_daily_variable, clean_daily_dataframe

        # Multi-day precipitation
        elem, conv, can = resolve_daily_variable("multi_day_prcp")
        self.assertEqual(elem, "MDPR")
        self.assertEqual(conv(250), 25.0)

        # Pan evaporation
        elem, conv, can = resolve_daily_variable("pan_evaporation")
        self.assertEqual(elem, "EVAP")
        self.assertEqual(conv(85), 8.5)

        # Observation-time temperature
        elem, conv, can = resolve_daily_variable("obs_temp")
        self.assertEqual(elem, "TOBS")
        self.assertEqual(conv(225), 22.5)

        # Convective wind gust
        elem, conv, can = resolve_daily_variable("wind_gust")
        self.assertEqual(elem, "WSFG")
        self.assertEqual(conv(320), 32.0)

        # Convective thunderstorm & Harmattan haze
        elem, conv, can = resolve_daily_variable("thunder")
        self.assertEqual(elem, "WT03")
        elem, conv, can = resolve_daily_variable("harmattan_haze")
        self.assertEqual(elem, "WT08")

        # Diurnal temperature range
        elem, conv, can = resolve_daily_variable("diurnal_range")
        self.assertEqual(elem, "DTR")

        # QC bounds
        df_qc = pd.DataFrame([
            {"variable": "pan_evaporation", "observation": 12.0},
            {"variable": "pan_evaporation", "observation": 85.0},  # Unphysical > 50 mm/day
            {"variable": "obs_temp", "observation": 28.5},
            {"variable": "multi_day_prcp", "observation": 120.0},
        ])
        cleaned = clean_daily_dataframe(df_qc, apply_qc=True)
        self.assertEqual(len(cleaned), 3)
        self.assertNotIn(85.0, cleaned["observation"].values)

    def test_hourly_station_resolution_and_methods(self):
        """Verify hourly station identifier resolution (ICAO codes, WMO IDs) and ingestion."""
        from masharikiweather import GHCNHourly, query_ghcn_hourly, get_station_series

        # 1. Resolve ICAO codes
        canon_id_hkjk, meta_hkjk = GHCNHourly.resolve_station("HKJK")
        self.assertEqual(canon_id_hkjk, "KEI0000HKJK")
        self.assertEqual(meta_hkjk.get("icao"), "HKJK")

        canon_id_hknw, meta_hknw = GHCNHourly.resolve_station("HKNW")
        self.assertEqual(canon_id_hknw, "KEI0000HKNW")
        self.assertEqual(meta_hknw.get("icao"), "HKNW")

        # 2. Query hourly observations with ICAO aliases
        df_hourly = query_ghcn_hourly(
            stations=["HKJK", "HKNW"],
            variables=["t2m", "d2m"],
            start_date="2024-07-01",
            end_date="2024-07-02",
            format="long",
        )
        self.assertFalse(df_hourly.empty)
        self.assertIn("station", df_hourly.columns)
        self.assertIn("ghcn_id", df_hourly.columns)

        # 3. get_station_series safe extraction by ICAO code
        s_hkjk = get_station_series(df_hourly, station="HKJK", variable="t2m")
        self.assertFalse(s_hkjk.empty)
        self.assertTrue(10.0 <= s_hkjk.mean() <= 30.0)


if __name__ == "__main__":
    unittest.main()



