"""
Unified Satellite Extraction Engine across Multi-Mission Constellations for MasharikiWeather.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..base import AFRICA_BBOX, BaseExtractor, SpatialBoundingBox
from .common import _parse_date
from .landsaf import LANDSAF, LANDSAF_PRODUCTS, LANDSAFWrapper, Landsaf, LandsafWrapper, get_landsaf_data
from .nasa_power import (
    NASA_POWER_CATALOG,
    NASA_POWER_PARAMETERS,
    NASAPower,
    NASAPowerWrapper,
    NasaPower,
    NasaPowerWrapper,
    get_nasa_power_data,
)
from .pvgis import PVGIS, PVGIS_DATABASES, PVGIS_RADIATION_VARS, PVGISWrapper, get_pvgis_data
from .sarah3 import SARAH3, SARAH3_PARAMETERS, SARAH3Wrapper, get_sarah3_data
from .tropomi import TROPOMI, TROPOMI_PRODUCTS, TROPOMIWrapper, get_tropomi_data

logger = logging.getLogger("MasharikiWeather.Satellite.Extractor")

class SatelliteExtractor(BaseExtractor):
    """
    Unified Satellite Extraction Engine across NASA POWER, PVGIS, SARAH-3, LANDSAF, and TROPOMI.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "./data/satellite",
        source: str = "nasa_power",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source = source.lower().strip()

    def extract(
        self,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        locations: Optional[Any] = None,
        bbox: Optional[SpatialBoundingBox] = None,
        variables: Optional[Union[str, Sequence[str]]] = None,
        format: str = "wide",
        save_parquet: bool = True,
        **kwargs: Any,
    ) -> Union[pd.DataFrame, Path]:
        """
        Execute satellite observation extraction and optionally save to Parquet.

        Mathematical docstring:
        Input:
            start_date, end_date: temporal range [T_start, T_end].
            locations: sequence of N point coordinates or SpatialBoundingBox [lat_min, lat_max, lon_min, lon_max].
        Output shape:
        - Wide format: (T_timestamps, N_locations * C_variables)
        - Long format: (N_records, 6) where columns are [time, station/location, lat, lon, variable, value].
        """
        locs = locations
        if locs is None:
            box = bbox if bbox is not None else AFRICA_BBOX
            lats = np.linspace(box.lat_min, box.lat_max, 3)
            lons = np.linspace(box.lon_min, box.lon_max, 3)
            locs = [{"name": f"pt_{i}_{j}", "lat": float(la), "lon": float(lo)}
                    for i, la in enumerate(lats) for j, lo in enumerate(lons)]

        s_str = _parse_date(start_date, "")
        e_str = _parse_date(end_date, "")

        vars_arg: Optional[Union[str, List[str]]]
        if variables == "all":
            vars_arg = "all"
        elif isinstance(variables, str):
            vars_arg = [variables]
        elif variables is not None:
            vars_arg = list(variables)
        else:
            vars_arg = None

        if self.source in ("nasa_power", "power"):
            from . import NASAPower
            inst = NASAPower(cache_dir=str(self.output_dir / "cache"))
            df = inst(
                variables=vars_arg,
                locations=locs,
                start_date=s_str,
                end_date=e_str,
                format=format,
                **kwargs,
            )
        elif self.source in ("pvgis", "jrc_pvgis"):
            from . import PVGIS
            inst = PVGIS(cache_dir=str(self.output_dir / "cache"))
            df = inst(
                locations=locs,
                start_date=s_str,
                end_date=e_str,
                start_year=pd.to_datetime(start_date).year,
                end_year=pd.to_datetime(end_date).year,
                format=format,
                **kwargs,
            )
        elif self.source in ("sarah3", "sarah_3", "cm_saf"):
            from . import SARAH3
            inst = SARAH3(cache_dir=str(self.output_dir / "cache"))
            df = inst(
                variables=vars_arg,
                locations=locs,
                start_date=s_str,
                end_date=e_str,
                format=format,
                **kwargs,
            )
        elif self.source in ("landsaf", "lsa_saf"):
            from . import LANDSAF
            inst = LANDSAF(cache_dir=str(self.output_dir / "cache"))
            df = inst(
                variables=vars_arg,
                locations=locs,
                start_date=s_str,
                end_date=e_str,
                format=format,
                **kwargs,
            )
        elif self.source in ("tropomi", "sentinel5p", "s5p"):
            from . import TROPOMI
            inst = TROPOMI(cache_dir=str(self.output_dir / "cache"))
            df = inst(
                variables=vars_arg,
                locations=locs,
                start_date=s_str,
                end_date=e_str,
                format=format,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown satellite source '{self.source}'. Supported: 'nasa_power', 'pvgis', 'sarah3', 'landsaf', 'tropomi'.")

        if save_parquet:
            st_str = pd.to_datetime(start_date).strftime("%Y%m%d")
            en_str = pd.to_datetime(end_date).strftime("%Y%m%d")
            out_file = self.output_dir / f"{self.source}_{st_str}_{en_str}.parquet"
            if isinstance(df.columns, pd.MultiIndex):
                flat_cols = ['_'.join(str(c) for c in col).strip('_') for col in df.columns]
                flat_df = df.copy()
                flat_df.columns = flat_cols
                flat_df.reset_index().to_parquet(out_file, index=False)
            else:
                df.reset_index().to_parquet(out_file, index=False)
            return out_file

        return df

