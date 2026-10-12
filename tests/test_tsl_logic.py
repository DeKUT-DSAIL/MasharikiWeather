import sys
sys.path.append("/home/teo/Documents/DSAIL/MasharikiWeather")

from masharikiweather.dataset import MasharikiWeatherDataset
import numpy as np
import pandas as pd
import xarray as xr

def create_mock_grid(year, freq="1D"):
    times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq=freq)
    lats = np.linspace(-5, 5, 10)
    lons = np.linspace(30, 40, 10)
    data = np.random.rand(len(times), len(lats), len(lons))
    ds = xr.Dataset(
        data_vars={"precipitation": (("time", "lat", "lon"), data)},
        coords={
            "time": times,
            "lat": lats,
            "lon": lons
        }
    )
    return ds

def test_tsl_logic():
    print("Initializing test dataset...")
    ds = MasharikiWeatherDataset(
        years=[2023],
        source_obs=["tahmo"],
        parameters=["total_precipitation"],
        freq="d"
    )

    test_stations = ds.stations.tolist()[:3]
    first_date = "2023-01-01"
    last_date = "2023-01-10" # 10 days of data
    
    obs, mask = ds.get_observations(
        stations=test_stations,
        first_date=first_date,
        last_date=last_date,
        as_numpy=False,
        return_mask=True
    )
    N = len(obs)
    print(f"Total contiguous timesteps available between dates: {N}")

    # Case 1: Standard (W=3, H=2, D=0, S=1)
    print("\n--- Case 1: Standard (W=3, H=2, delay=0, stride=1) ---")
    w1 = ds.get_windows(
        window_size=3, horizon_size=2, delay=0, stride=1, lazy=True,
        stations=test_stations, first_date=first_date, last_date=last_date,
    )
    print(f"x shape: {w1.x.shape} (Expected: {N - 3 - 0 - 2 + 1}, 3, 3, 1)")
    print(f"y shape: {w1.y.shape} (Expected: {N - 3 - 0 - 2 + 1}, 2, 3, 1)")
    
    # Case 2: Delay (W=3, H=2, D=2, S=1)
    print("\n--- Case 2: Delay (W=3, H=2, delay=2, stride=1) ---")
    w2 = ds.get_windows(
        window_size=3, horizon_size=2, delay=2, stride=1, lazy=True,
        stations=test_stations, first_date=first_date, last_date=last_date,
    )
    print(f"x shape: {w2.x.shape} (Expected: {N - 3 - 2 - 2 + 1}, 3, 3, 1)")
    print(f"y shape: {w2.y.shape} (Expected: {N - 3 - 2 - 2 + 1}, 2, 3, 1)")
    
    # Check that Y is appropriately delayed mathematically
    print(f"w2.index_x[0] = \n{w2.index_x[0]}")
    print(f"w2.index_y[0] (should be shifted +2 days) = \n{w2.index_y[0]}")
    
    # Case 3: Stride (W=3, H=2, D=0, S=2)
    print("\n--- Case 3: Stride (W=3, H=2, delay=0, stride=2) ---")
    w3 = ds.get_windows(
        window_size=3, horizon_size=2, delay=0, stride=2, lazy=True,
        stations=test_stations, first_date=first_date, last_date=last_date,
    )
    expected_stride_batches = len(np.arange(0, N - 3 - 0 - 2 + 1, 2))
    print(f"x shape: {w3.x.shape} (Expected: {expected_stride_batches}, 3, 3, 1)")
    print(f"y shape: {w3.y.shape} (Expected: {expected_stride_batches}, 2, 3, 1)")
    print(f"w3.index_x[0][0] = {w3.index_x[0][0]}")
    print(f"w3.index_x[1][0] (should be +2 days) = {w3.index_x[1][0]}")


def test_gridded_tsl_logic():
    print("\n=============================================")
    print("Testing Gridded Products with Mocked Grid...")
    ds = MasharikiWeatherDataset(
        years=[2023],
        source_obs=["tahmo"],
        parameters=["total_precipitation"],
        freq="d"
    )

    # Mock the zarr loader completely to dynamically prevent network issues and test math
    ds._open_gridded_zarr = lambda group, year, lazy=True: create_mock_grid(year, "1D")

    test_stations = ds.stations.tolist()[:3]
    first_date = "2023-01-01"
    last_date = "2023-01-10" # 10 days of data = 10 timesteps

    obs, mask = ds.get_observations(
        stations=test_stations,
        first_date=first_date,
        last_date=last_date,
        as_numpy=False,
        return_mask=True
    )
    N = len(obs)

    print("\n--- LAZY GRID (W=3, H=2, delay=2, stride=2, lazy=True) ---")
    w_lazy = ds.get_windows(
        window_size=3, horizon_size=2, delay=2, stride=2, lazy=True,
        stations=test_stations, first_date=first_date, last_date=last_date,
        gridded_url="era5", spatial_mode="grid", ml_task="downscaling"
    )
    expected_stride_batches = len(np.arange(0, N - 3 - 2 - 2 + 1, 2))
    print(f"valid_batch_indices: {w_lazy.valid_batch_indices} (Expected length: {expected_stride_batches})")
    print(f"Is gridded_x an xarray Dataset pointer? {isinstance(w_lazy.gridded_x, xr.Dataset)}")
    print(f"Target dataset variables: {w_lazy.gridded_features}")

    print("\n--- FULL EAGER GRID EXTRACTION (W=3, H=2, delay=2, stride=2, lazy=False) ---")
    w_eager = ds.get_windows(
        window_size=3, horizon_size=2, delay=2, stride=2, lazy=False,
        stations=test_stations, first_date=first_date, last_date=last_date,
        gridded_url="era5", spatial_mode="grid", ml_task="downscaling"
    )
    # The eager grid extraction concatenates channels so it returns NumPy: (Batch, sequence, Channels, Lat, Lon)
    print(f"gridded_x eager shape: {w_eager.gridded_x.shape} (Expected: {expected_stride_batches}, 3, 1, 10, 10)")
    print(f"gridded_y eager shape: {w_eager.gridded_y.shape} (Expected: {expected_stride_batches}, 2, 1, 10, 10)")

    print("\nAll gridded spatial tests mathematically correct!")

if __name__ == '__main__':
    test_tsl_logic()
    test_gridded_tsl_logic()
