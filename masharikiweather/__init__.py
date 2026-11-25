"""
MasharikiWeather: Spatiotemporal data fusion for Africa.
"""

from . import extract
from .dataset import (
    MasharikiWeatherDataset,
    extract_stations,
    extract_gridded,
    pull_station_data,
    pull_gridded_data,
    pull_weather_data,
    AFRICA_BBOX,
    EAST_AFRICA_BBOX,
    EASTERN_AFRICA_BBOX,
    EAST_AFRICA_COUNTRIES,
    EASTERN_AFRICA_COUNTRIES,
    SpatialBoundingBox,
    is_eastern_africa_country,
    parse_spatial_bbox,
)
from .rechunker import rechunk_dataset

__version__ = "0.2.8"
__author__ = "Teofilo Ligawa, Samuel Mbugua, James Muiru, Hilda Muthoni, Ciira wa Maina"
__authors__ = [
    "Teofilo Ligawa",
    "Samuel Mbugua",
    "James Muiru",
    "Hilda Muthoni",
    "Ciira wa Maina",
]
__affiliation__ = "Centre for Data Science and Artificial Intelligence (DSAIL), Dedan Kimathi University of Technology"

__all__ = [
    "MasharikiWeatherDataset",
    "rechunk_dataset",
    "extract_stations",
    "extract_gridded",
    "pull_station_data",
    "pull_gridded_data",
    "pull_weather_data",
    "extract",
    "AFRICA_BBOX",
    "EAST_AFRICA_BBOX",
    "EASTERN_AFRICA_BBOX",
    "EAST_AFRICA_COUNTRIES",
    "EASTERN_AFRICA_COUNTRIES",
    "SpatialBoundingBox",
    "is_eastern_africa_country",
    "parse_spatial_bbox",
]
