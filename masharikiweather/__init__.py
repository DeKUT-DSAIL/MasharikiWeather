"""
MasharikiWeather: Spatiotemporal data fusion for East Africa.
"""

__all__ = []

try:
    from .dataset import MasharikiWeatherDataset, Windows
    from .rechunker import rechunk_dataset
    __all__.extend(["MasharikiWeatherDataset", "Windows", "rechunk_dataset"])
except ImportError:
    MasharikiWeatherDataset = None  # type: ignore
    Windows = None  # type: ignore
    rechunk_dataset = None  # type: ignore

try:
    from .ghcn_apiwrapper import (
        GHCN,
        GHCNDaily,
        GHCNHourly,
        GHCNDailyLexicon,
        GHCNHourlyLexicon,
        DEFAULT_KENYA_BOUNDS,
        AFRICAN_REGIONS,
        CONTINENT_REGIONS,
        GLOBAL_REGIONS,
        haversine_distance,
        align_to_reference_grid,
        find_nearest_stations,
        find_nearest_grid_point,
        to_wide,
        get_stations,
        get_african_stations,
        query_ghcn_daily,
        query_ghcn_hourly,
        check_ghcn_availability,
        check_station_coverage,
        check_variable_activeness,
        get_stations_variable_activeness,
        get_variable_lifecycle,
        calculate_relative_humidity,
        get_station_series,
    )
    __all__.extend([
        "GHCN",
        "GHCNDaily",
        "GHCNHourly",
        "GHCNDailyLexicon",
        "GHCNHourlyLexicon",
        "DEFAULT_KENYA_BOUNDS",
        "AFRICAN_REGIONS",
        "CONTINENT_REGIONS",
        "GLOBAL_REGIONS",
        "haversine_distance",
        "align_to_reference_grid",
        "find_nearest_stations",
        "find_nearest_grid_point",
        "to_wide",
        "get_stations",
        "get_african_stations",
        "query_ghcn_daily",
        "query_ghcn_hourly",
        "check_ghcn_availability",
        "check_station_coverage",
        "check_variable_activeness",
        "get_stations_variable_activeness",
        "get_variable_lifecycle",
        "calculate_relative_humidity",
        "get_station_series",
    ])
except ImportError:
    pass

try:
    from . import api_wrappers
    for _name in getattr(api_wrappers, "__all__", []):
        globals()[_name] = getattr(api_wrappers, _name)
        if _name not in __all__:
            __all__.append(_name)
except ImportError:
    pass

__version__ = "0.2.0"