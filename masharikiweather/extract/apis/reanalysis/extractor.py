"""
Unified Reanalysis Extraction Engine across Atmospheric Reanalyses for MasharikiWeather.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..base import AFRICA_BBOX, BaseExtractor, SpatialBoundingBox
from .cams import CAMS, CAMSWrapper, get_cams_data
from .common import _parse_date
from .merra2 import MERRA2, MERRA2Wrapper, get_merra2_data
from .open_meteo import OpenMeteo, OpenMeteoWrapper, get_open_meteo_data

logger = logging.getLogger("MasharikiWeather.Reanalysis.Extractor")


class ReanalysisExtractor(BaseExtractor):
    """
    Unified Reanalysis Extraction Engine across CAMS, MERRA-2, and Open-Meteo.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "./data/reanalysis",
        source: str = "open_meteo",
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
        Execute reanalysis extraction from selected source and optionally save to Parquet.

        Mathematical docstring:
        Input:
            start_date, end_date: temporal range [T_start, T_end].
            locations: sequence of N point coordinates or SpatialBoundingBox [lat_min, lat_max, lon_min, lon_max].
        Output shape:
            - Wide format: (T_timestamps, N_locations * C_variables) or (T_timestamps, C_variables)
            - Long format: (N_records, 6) where columns are [time, station/location, lat, lon, variable, value].

        Parameters
        ----------
        start_date, end_date : str, datetime, or pd.Timestamp
            Date range boundaries.
        locations : sequence of dict or tuple, optional
            List of target locations.
        bbox : SpatialBoundingBox, optional
            Spatial bounding box to sample if locations are not explicitly specified.
        variables : sequence of str, optional
            Target variables to retrieve.
        format : str, default 'wide'
            'wide' or 'long' DataFrame layout.
        save_parquet : bool, default True
            Whether to write output to a local Parquet file.
        **kwargs : Any
            Additional source-specific extraction arguments.

        Returns
        -------
        pd.DataFrame or Path
            Extracted tabular observations or Path to saved Parquet artifact.
        """
        locs = locations
        if locs is None:
            box = bbox if bbox is not None else AFRICA_BBOX
            # Generate representative grid points within bounding box
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

        if self.source in ("cams", "copernicus_cams"):
            from . import CAMS
            inst = CAMS(cache_dir=str(self.output_dir / "cache"))
            df = inst(
                variables=vars_arg,
                locations=locs,
                start_date=s_str,
                end_date=e_str,
                format=format,
                **kwargs,
            )
        elif self.source in ("merra2", "nasa_merra2"):
            from . import MERRA2
            inst = MERRA2(cache_dir=str(self.output_dir / "cache"))
            df = inst(
                variables=vars_arg,
                locations=locs,
                start_date=s_str,
                end_date=e_str,
                format=format,
                **kwargs,
            )
        elif self.source in ("open_meteo", "openmeteo"):
            from . import OpenMeteo
            inst = OpenMeteo(cache_dir=str(self.output_dir / "cache"))
            df = inst(
                variables=vars_arg,
                locations=locs,
                start_date=s_str,
                end_date=e_str,
                format=format,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown reanalysis source '{self.source}'. Supported: 'open_meteo', 'cams', 'merra2'.")

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


__all__ = ["ReanalysisExtractor"]
