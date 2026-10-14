"""
Unit Tests for MasharikiWeather Multi-Source Climate, Satellite & Reanalysis Wrappers
======================================================================================
Tests NASA POWER, PVGIS, SARAH-3, CAMS, MERRA-2, LANDSAF, TROPOMI, and TAHMO prep.
"""

import unittest
import numpy as np
import pandas as pd

from masharikiweather import (
    NASAPower,
    query_nasa_power,
    PVGIS,
    query_pvgis,
    SARAH3,
    query_sarah3,
    CAMS,
    query_cams_aerosols,
    OpenMeteo,
    query_open_meteo,
    MERRA2,
    query_merra2,
    search_merra2_parameters,
    list_merra2_parameters,
    get_merra2_metadata,
    list_merra2_collections,
    LANDSAF,
    query_landsaf,
    search_landsaf_parameters,
    list_landsaf_parameters,
    get_landsaf_metadata,
    list_landsaf_products,
    TROPOMI,
    query_tropomi,
    search_tropomi_parameters,
    list_tropomi_parameters,
    get_tropomi_metadata,
    query_dataset,
)


class TestDatasetWrappers(unittest.TestCase):

    def test_nasa_power(self):
        """Test NASA POWER zero-key REST API."""
        # Query 2 days for Kenya centroid
        df = query_nasa_power(
            region="kenya",
            variables=["ghi", "t2m"],
            start_date="2024-01-01",
            end_date="2024-01-02",
            resolution="hourly",
            format="wide",
        )
        self.assertFalse(df.empty)
        self.assertIsInstance(df.index, pd.DatetimeIndex)

    def test_pvgis(self):
        """Test EU JRC PVGIS hourly API."""
        df = query_pvgis(
            lat=-1.286, lon=36.817, # Nairobi
            variables=["ghi", "t2m"],
            start_date="2020-01-01",
            end_date="2020-01-02",
            database="PVGIS-SARAH2",
            format="wide",
        )
        self.assertFalse(df.empty)
        self.assertIsInstance(df.index, pd.DatetimeIndex)

    def test_sarah3(self):
        """Test CM SAF SARAH-3 satellite solar irradiance."""
        df = query_sarah3(
            lat=-1.286, lon=36.817,
            variables=["ghi", "direct", "diffuse"],
            start_date="2020-01-01",
            end_date="2020-01-02",
            format="wide",
        )
        self.assertFalse(df.empty)
        self.assertIsInstance(df.index, pd.DatetimeIndex)

    def test_cams_aerosols(self):
        """Test CAMS Global Aerosols zero-key API."""
        df = query_cams_aerosols(
            region="kenya",
            variables=["aod", "dust", "pm10"],
            start_date="2024-01-01",
            end_date="2024-01-02",
            format="wide",
        )
        self.assertFalse(df.empty)
        self.assertIsInstance(df.index, pd.DatetimeIndex)

    def test_open_meteo(self):
        """Test Open-Meteo zero-key high-resolution weather API."""
        df = query_open_meteo(
            lat=-0.398, lon=36.958, # DeKUT Nyeri
            variables=["temp", "precip", "rh", "wind_speed"],
            start_date="2024-01-01",
            end_date="2024-01-02",
            format="wide",
        )
        self.assertFalse(df.empty)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertIn("temp", df.columns)

    def test_merra2(self):
        """Test MERRA-2 aerosol and meteorological reanalysis wrapper."""
        # 1. Parameter and collection discovery
        params_df = list_merra2_parameters()
        self.assertFalse(params_df.empty)
        self.assertIn("Variable", params_df.columns)

        colls_df = list_merra2_collections()
        self.assertFalse(colls_df.empty)
        self.assertIn("Collection ID", colls_df.columns)

        meta = get_merra2_metadata("swgdn")
        self.assertEqual(meta["unit"], "W/m²")
        self.assertEqual(meta["merra_name"], "SWGDN")

        # 2. Query DeKUT Nyeri for aerosols and meteorology
        df = query_merra2(
            lat=-0.398, lon=36.958, # DeKUT Nyeri
            variables=["aod_total", "aod_dust", "swgdn", "t2m"],
            start_date="2020-01-01",
            end_date="2020-01-02",
            format="wide",
        )
        self.assertFalse(df.empty)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertIn("aod_total", df.columns)
        self.assertIn("swgdn", df.columns)

    def test_landsaf(self):
        """Test EUMETSAT LSA SAF Downward Solar Radiation wrapper."""
        # 1. Parameter and product discovery
        params_df = list_landsaf_parameters()
        self.assertFalse(params_df.empty)
        self.assertIn("Variable", params_df.columns)

        prods_df = list_landsaf_products()
        self.assertFalse(prods_df.empty)
        self.assertIn("Product Code", prods_df.columns)

        meta = get_landsaf_metadata("dssf_tot")
        self.assertEqual(meta["unit"], "W/m²")
        self.assertEqual(meta["product_code"], "MDSSFTD")

        # 2. Query DeKUT Nyeri for solar fluxes and surface temperature
        df = query_landsaf(
            lat=-0.398, lon=36.958, # DeKUT Nyeri
            variables=["dssf_tot", "dssf_dir", "dssf_diff", "fraction_diffuse", "dslf", "lst"],
            start_date="2020-01-01",
            end_date="2020-01-02",
            frequency="30min",
            format="wide",
        )
        self.assertFalse(df.empty)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertIn("dssf_tot", df.columns)
        self.assertIn("lst", df.columns)

    def test_tropomi(self):
        """Test Sentinel-5P TROPOMI Level-2 atmospheric wrapper."""
        # 1. Parameter discovery
        params_df = list_tropomi_parameters()
        self.assertFalse(params_df.empty)
        self.assertIn("Variable", params_df.columns)

        search_res = search_tropomi_parameters(query="no2")
        self.assertTrue(len(search_res) > 0)

        meta = get_tropomi_metadata("ch4")
        self.assertEqual(meta["unit"], "ppb")
        self.assertEqual(meta["product_id"], "L2__CH4___")

        # 2. Query DeKUT Nyeri for multiple Level-2 trace gases and aerosols
        df = query_tropomi(
            lat=-0.398, lon=36.958, # DeKUT Nyeri
            variables=["no2", "co", "ch4", "aer_ai", "so2"],
            start_date="2024-01-01",
            end_date="2024-01-03",
            format="wide",
        )
        self.assertFalse(df.empty)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertIn("nitrogen_dioxide", df.columns)
        self.assertIn("methane", df.columns)

    def test_query_dispatcher(self):
        """Test unified query_dataset dispatcher."""
        # Dispatch to NASA POWER
        df_power = query_dataset(
            source="nasa_power",
            region="kenya",
            variables=["ghi"],
            start_date="2024-01-01",
            end_date="2024-01-02",
        )
        self.assertFalse(df_power.empty)

        # Dispatch to PVGIS
        df_pvgis = query_dataset(
            source="pvgis",
            lat=-1.286, lon=36.817,
            variables=["ghi"],
            start_date="2020-01-01",
            end_date="2020-01-02",
        )
        self.assertFalse(df_pvgis.empty)

        # Dispatch to SARAH-3
        df_sarah = query_dataset(
            source="sarah3",
            lat=-1.286, lon=36.817,
            variables=["ghi"],
            start_date="2020-01-01",
            end_date="2020-01-02",
        )
        self.assertFalse(df_sarah.empty)


if __name__ == "__main__":
    unittest.main()
