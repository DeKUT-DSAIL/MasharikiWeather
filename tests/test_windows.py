"""
demo_universal_windows.py

Demonstration script querying the 8 primary spatial ML tasks supported by `get_windows`
under the new N-Grid Heterogeneous Bipartite Architecture.
"""
import argparse
from masharikiweather.dataset import MasharikiWeatherDataset


def _print_windows(win, label=""):
    """Helper to print all populated Windows fields."""
    if label:
        print(label)
    print(f"  station_x: {win.station_x.shape}")
    if win.station_y is not None and len(win.station_y) > 0:
        print(f"  station_y: {win.station_y.shape}")
    else:
        print(f"  station_y: None (blocked)")
        
    for name, gx in win.grids_x.items():
        print(f"  grids_x['{name}']: {gx.shape if hasattr(gx, 'shape') else type(gx)}")
        
    for name, gy in win.grids_y.items():
        print(f"  grids_y['{name}']: {gy.shape if hasattr(gy, 'shape') else type(gy)}")
        
    if getattr(win, 'target_node', None):
        print(f"  target_node: {win.target_node}")

    if win.station_static is not None:
        print(f"  station_static: {win.station_static.shape} {win.station_static.dtype}")
        
    if getattr(win, 'native_topography', None) is not None:
        shape = win.native_topography.shape if hasattr(win.native_topography, 'shape') else type(win.native_topography)
        print(f"  native_topography: {shape}")
        
    if getattr(win, 'native_topography_categorical', None) is not None:
        shape = win.native_topography_categorical.shape if hasattr(win.native_topography_categorical, 'shape') else type(win.native_topography_categorical)
        print(f"  native_topography_categorical: {shape}")


def main():
    parser = argparse.ArgumentParser(description="Query ML tasks supported by get_windows.")
    parser.add_argument("--token", type=str, required=False, help="Hugging Face token.")
    parser.add_argument("--target-url", type=str, default="chirps", help="Target grid HF group (e.g. 'chirps').")
    args = parser.parse_args()

    print("Initializing Dataset...")
    ds = MasharikiWeatherDataset(
        token=args.token,
        years=[2023],
        source_obs=["tahmo"],
        parameters=["total_precipitation"],
        freq="d"
    )

    test_stations = ds.stations.tolist()[:3]
    first_date = "2023-01-01"
    last_date = "2023-01-05"

    # --- 1. Grid-to-Point Downscaling ---
    print("\n--- 1. Grid-to-Point Downscaling ---")
    win = ds.get_windows(
        window_size=2, horizon_size=1, stations=test_stations,
        first_date=first_date, last_date=last_date,
        gridded_url="era5", spatial_mode="stations", ml_task="downscaling",
        crop_to_stations=False, target_node="tahmo"
    )
    _print_windows(win)

    # --- 2. Grid-to-Grid Forecasting ---
    print("\n--- 2. Grid-to-Grid Forecasting (like GraphCast)---")
    win = ds.get_windows(
        window_size=2, horizon_size=1, stations=test_stations,
        first_date=first_date, last_date=last_date,
        gridded_url="era5", spatial_mode="grid", ml_task="forecasting",
        crop_to_stations=False, target_node="era5"
    )
    _print_windows(win)

    # --- 3. Field Correction ---
    print("\n--- 3. Field Correction (ERA5 correction with station anchors) ---")
    win = ds.get_windows(
        window_size=2, horizon_size=1, stations=test_stations,
        first_date=first_date, last_date=last_date,
        gridded_url="era5", spatial_mode="grid", ml_task="correction",
        crop_to_stations=False, target_node="era5"
    )
    _print_windows(win)

    # --- 4. Grid-to-Point Forecasting ---
    print("\n--- 4. Grid-to-Point Forecasting ---")
    win = ds.get_windows(
        window_size=2, horizon_size=1, stations=test_stations,
        first_date=first_date, last_date=last_date,
        gridded_url="era5", spatial_mode="stations", ml_task="forecasting",
        crop_to_stations=False, target_node="tahmo"
    )
    _print_windows(win)

    # --- 5. Supervised Grid-to-Finer-Grid Downscaling ---
    print(f"\n--- 5. Supervised Downscaling: ERA5 -> {args.target_url} ---")
    win = ds.get_windows(
        window_size=2, horizon_size=1, stations=test_stations,
        first_date=first_date, last_date=last_date,
        gridded_url=["era5", args.target_url], spatial_mode="grid", ml_task="downscaling",
        crop_to_stations=False, target_node=args.target_url
    )
    _print_windows(win)

    # --- 6. Multi-Grid Correction ---
    print(f"\n--- 6. Multi-Grid Correction (ERA5 + {args.target_url} + Stations correcting ERA5) ---")
    win = ds.get_windows(
        window_size=2, horizon_size=1, stations=test_stations,
        first_date=first_date, last_date=last_date,
        gridded_url=["era5", args.target_url], spatial_mode="grid", ml_task="correction",
        crop_to_stations=False, target_node="era5"
    )
    _print_windows(win)

    # --- 7. Station-only Forecasting ---
    print("\n--- 7. Station-only Forecasting (No Grids) ---")
    win = ds.get_windows(
        window_size=2, horizon_size=1, stations=test_stations,
        first_date=first_date, last_date=last_date,
        gridded_url=None, spatial_mode="stations", ml_task="forecasting",
        target_node="tahmo"
    )
    _print_windows(win)

    # --- 8. Multi-Grid + Station Forecast ---
    print(f"\n--- 8. Multi-Grid + Station Forecast (ERA5 + {args.target_url} + Stations -> Station Future) ---")
    win = ds.get_windows(
        window_size=2, horizon_size=1, stations=test_stations,
        first_date=first_date, last_date=last_date,
        gridded_url=["era5", args.target_url], spatial_mode="stations", ml_task="forecasting",
        crop_to_stations=False, target_node="tahmo"
    )
    _print_windows(win)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


# pipeline flexibility
# Example: Rift Valley correction with per-grid variable selection
# win = ds.get_windows(
#     gridded_url=["era5", "imerg"],
#     gridded_variables={
#         "era5": ["2m_temperature", "total_precipitation"],
#         "imerg": ["total_precipitation", "randomError"],
#     },
#     target_node="era5",
#     ml_task="correction",
#     bbox=(-1.5, 35.0, 0.5, 37.5),   # Rift Valley
#     window_size=24, horizon_size=0
# )

