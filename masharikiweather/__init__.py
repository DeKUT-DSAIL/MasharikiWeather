"""
MasharikiWeather: Spatiotemporal data fusion for East Africa.
"""

from .dataset import MasharikiWeatherDataset, Windows
from .rechunker import rechunk_dataset
from .dataloader import (
    EarthSystemAdapter,
    PyGDataLoader,
    create_dataloader,
    create_distributed_dataloader,
    project_to_local_crs,
    BatchDescriptor,
)

__version__ = "0.1.5"
__all__ = [
    "MasharikiWeatherDataset",
    "Windows",
    "rechunk_dataset",
    "EarthSystemAdapter",
    "PyGDataLoader",
    "create_dataloader",
    "create_distributed_dataloader",
    "project_to_local_crs",
    "BatchDescriptor",
]