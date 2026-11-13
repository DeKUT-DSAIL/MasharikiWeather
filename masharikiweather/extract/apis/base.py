"""
Base Extraction Interfaces and Spatial/Temporal Validators for MasharikiWeather.

Defines standardized data classes and abstract extraction contracts enforcing
CF-compliant coordinate standards, spatial bounding box validation, temporal interval
chunking, and framework-agnostic persistence.
"""

from __future__ import annotations

import calendar
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Sequence, Tuple, Union

import pandas as pd


@dataclass(frozen=True)
class SpatialBoundingBox:
    """
    Standard spatial bounding box container with coordinate sanity enforcement.

    Parameters
    ----------
    lat_min : float
        Southernmost latitude in degrees north [-90.0, 90.0].
    lat_max : float
        Northernmost latitude in degrees north [-90.0, 90.0].
    lon_min : float
        Westernmost longitude in degrees east [-180.0, 180.0].
    lon_max : float
        Easternmost longitude in degrees east [-180.0, 180.0].
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.lat_min <= 90.0):
            raise ValueError(f"lat_min out of bounds [-90, 90]: {self.lat_min}")
        if not (-90.0 <= self.lat_max <= 90.0):
            raise ValueError(f"lat_max out of bounds [-90, 90]: {self.lat_max}")
        if self.lat_min > self.lat_max:
            raise ValueError(f"lat_min ({self.lat_min}) cannot exceed lat_max ({self.lat_max})")

        if not (-180.0 <= self.lon_min <= 180.0):
            raise ValueError(f"lon_min out of bounds [-180, 180]: {self.lon_min}")
        if not (-180.0 <= self.lon_max <= 180.0):
            raise ValueError(f"lon_max out of bounds [-180, 180]: {self.lon_max}")
        if self.lon_min > self.lon_max:
            raise ValueError(f"lon_min ({self.lon_min}) cannot exceed lon_max ({self.lon_max})")

    @classmethod
    def from_tuple(cls, bounds: Sequence[float], format_order: str = "min_max") -> SpatialBoundingBox:
        """
        Construct bounding box from numeric sequence.

        Supported formats:
        - 'min_max': [lat_min, lat_max, lon_min, lon_max]
        - 'north_west_south_east': [lat_max, lon_min, lat_min, lon_max] (CDS/ARCO order)
        - 'west_south_east_north': [lon_min, lat_min, lon_max, lat_max] (GIS/Earthdata order)
        """
        if len(bounds) != 4:
            raise ValueError(f"Expected 4 coordinate values, received {len(bounds)}")

        if format_order == "min_max":
            return cls(lat_min=float(bounds[0]), lat_max=float(bounds[1]),
                       lon_min=float(bounds[2]), lon_max=float(bounds[3]))
        elif format_order == "north_west_south_east":
            return cls(lat_max=float(bounds[0]), lon_min=float(bounds[1]),
                       lat_min=float(bounds[2]), lon_max=float(bounds[3]))
        elif format_order == "west_south_east_north":
            return cls(lon_min=float(bounds[0]), lat_min=float(bounds[1]),
                       lon_max=float(bounds[2]), lat_max=float(bounds[3]))
        else:
            raise ValueError(f"Unknown format_order: {format_order}")

    def to_area_list(self) -> List[float]:
        """
        Return coordinates in CDS / ARCO format: [North, West, South, East].
        """
        return [self.lat_max, self.lon_min, self.lat_min, self.lon_max]

    def to_area_str_list(self) -> List[str]:
        """
        Return coordinates in CDS string format: ['North', 'West', 'South', 'East'].
        """
        return [str(self.lat_max), str(self.lon_min), str(self.lat_min), str(self.lon_max)]

    def to_bbox_tuple(self) -> Tuple[float, float, float, float]:
        """
        Return coordinates in GIS / Earthdata format: (West, South, East, North).
        """
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)

    def to_dict(self) -> Dict[str, float]:
        """
        Return dictionary representation.
        """
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }

    def contains(self, lat: float, lon: float) -> bool:
        """
        Check whether given point resides within bounding box.
        """
        return (self.lat_min <= lat <= self.lat_max) and (self.lon_min <= lon <= self.lon_max)


def synthetic_fallback_allowed() -> bool:
    """
    Return True only when the environment variable MW_ALLOW_SYNTHETIC_FALLBACK is set
    to a truthy value. Climatological stand-in values are never produced otherwise.
    """
    return os.environ.get("MW_ALLOW_SYNTHETIC_FALLBACK", "").strip().lower() in ("1", "true", "yes")


# Canonical preset regions for Africa and continent
AFRICA_BBOX = SpatialBoundingBox(lat_min=-35.0, lat_max=38.0, lon_min=-20.0, lon_max=55.0)
KENYA_BBOX = SpatialBoundingBox(lat_min=-4.7, lat_max=5.5, lon_min=33.8, lon_max=41.9)
PAN_AFRICA_BBOX = AFRICA_BBOX


@dataclass(frozen=True)
class TemporalRange:
    """
    Standard temporal range container with validation and chunking utilities.

    Parameters
    ----------
    start : pd.Timestamp
        Start timestamp (inclusive).
    end : pd.Timestamp
        End timestamp (inclusive).
    """

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if not isinstance(self.start, pd.Timestamp):
            s = pd.to_datetime(self.start)
            if s.tzinfo is not None:
                s = s.tz_localize(None)
            object.__setattr__(self, "start", s)
        if not isinstance(self.end, pd.Timestamp):
            e = pd.to_datetime(self.end)
            if e.tzinfo is not None:
                e = e.tz_localize(None)
            object.__setattr__(self, "end", e)
        if self.start > self.end:
            raise ValueError(f"start date ({self.start}) cannot exceed end date ({self.end})")

    @classmethod
    def from_dates(
        cls,
        start_date: Union[str, datetime, date, pd.Timestamp],
        end_date: Union[str, datetime, date, pd.Timestamp],
    ) -> TemporalRange:
        """
        Construct TemporalRange from strings or datetime objects.
        """
        s = pd.to_datetime(start_date).tz_localize(None) if pd.to_datetime(start_date).tzinfo is not None else pd.to_datetime(start_date)
        e = pd.to_datetime(end_date).tz_localize(None) if pd.to_datetime(end_date).tzinfo is not None else pd.to_datetime(end_date)
        return cls(start=s, end=e)

    def to_iso_tuple(self) -> Tuple[str, str]:
        """
        Return ISO format strings (YYYY-MM-DDTHH:MM:SS).
        """
        return (self.start.isoformat(), self.end.isoformat())

    def to_date_str_tuple(self) -> Tuple[str, str]:
        """
        Return date strings (YYYY-MM-DD).
        """
        return (self.start.strftime("%Y-%m-%d"), self.end.strftime("%Y-%m-%d"))

    def years(self) -> List[int]:
        """
        Return unique years covering range.
        """
        return list(range(self.start.year, self.end.year + 1))

    def split_by_month(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Split temporal range into monthly sub-windows [month_start, month_end].
        Properly handles month boundaries, year boundaries, and leap years (Feb 28 vs Feb 29).
        """
        windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        cur = self.start
        while cur <= self.end:
            year = cur.year
            month = cur.month
            _, num_days = calendar.monthrange(year, month)
            month_end = min(pd.Timestamp(datetime(year, month, num_days, 23, 59, 59)), self.end)
            windows.append((cur, month_end))
            next_start = pd.Timestamp(datetime(year, month, num_days)) + timedelta(days=1)
            cur = next_start
        return windows

    def split_by_days(self, n_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Split temporal range into discrete n-day blocks.
        """
        if n_days < 1:
            raise ValueError(f"n_days must be >= 1, received {n_days}")
        windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        cur = self.start
        delta = timedelta(days=n_days)
        while cur <= self.end:
            chunk_end = min(cur + delta - timedelta(seconds=1), self.end)
            windows.append((cur, chunk_end))
            cur = chunk_end + timedelta(seconds=1)
        return windows


class BaseExtractor(ABC):
    """
    Abstract Base Class for all MasharikiWeather extraction engines.
    """

    @abstractmethod
    def extract(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute remote or cloud extraction and return dataset or local storage paths.
        """
        pass

    def validate_spatial_bounds(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> SpatialBoundingBox:
        """
        Validate and construct standard SpatialBoundingBox.
        """
        return SpatialBoundingBox(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

    def validate_temporal_range(
        self,
        start_date: Union[str, datetime, date, pd.Timestamp],
        end_date: Union[str, datetime, date, pd.Timestamp],
    ) -> TemporalRange:
        """
        Validate and construct standard TemporalRange.
        """
        return TemporalRange.from_dates(start_date=start_date, end_date=end_date)
