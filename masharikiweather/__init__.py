"""
MasharikiWeather: Spatiotemporal data fusion for Africa.
"""

from . import extract
from .dataset import (
    MasharikiWeatherDataset,
    Windows,
    extract_stations,
    extract_gridded,
    pull_station_data,
    pull_gridded_data,
    pull_weather_data,
)
from .rechunker import rechunk_dataset

__version__ = "0.1.9"
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
    "Windows",
    "rechunk_dataset",
    "extract_stations",
    "extract_gridded",
    "pull_station_data",
    "pull_gridded_data",
    "pull_weather_data",
    "extract",
]
