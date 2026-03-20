"""
demo_universal_windows.py

Demonstration script querying the 5 primary spatial ML tasks supported by `get_windows`:

1. Grid-to-Point Downscaling
2. Grid-to-Grid Forecasting (e.g. GraphCast)
3. Grid + Point to Grid Correction (e.g. Precipitation Correction)
4. Grid-to-Point Forecasting (station-based)
5. Supervised Grid-to-Finer-Grid Downscaling (e.g. ERA5 -> CHIRPS via CorrDiff/CGAN)
"""
import argparse
from masharikiweather.dataset import MasharikiWeatherDataset


def _print_windows(win, label=""):
    """Helper to print all populated Windows fields."""
    if label:
        print(label)
    print(f"  x: {win.x.shape}")
    print(f"  y: {win.y.shape}")
    if win.gridded_x is not None:
        print(f"  gridded_x: {win.gridded_x.shape}")
    if win.gridded_y is not None:
        print(f"  gridded_y: {win.gridded_y.shape}")
    else:
        print(f"  gridded_y: None (blocked)")
    if win.target_gridded_y is not None:
        print(f"  target_gridded_y: {win.target_gridded_y.shape} (hi-res target)")
        print(f"  target_grid_lats: {win.target_grid_lats.shape}, target_grid_lons: {win.target_grid_lons.shape}")
        print(f"  target_temporal_freq: {win.target_temporal_freq}")
    if win.static_stations is not None:
        print(f"  static_stations: {win.static_stations.shape} {win.static_stations.dtype}")
    if win.static_stations_categorical is not None:
        print(f"  static_stations_cat: {win.static_stations_categorical.shape} {win.static_stations_categorical.dtype}")
    if win.static_grid is not None:
        print(f"  static_grid: {win.static_grid.shape} {win.static_grid.dtype}")
    if win.static_grid_categorical is not None:
        print(f"  static_grid_cat: {win.static_grid_categorical.shape} {win.static_grid_categorical.dtype}")
    if win.static_grid_input is not None:
        print(f"  static_grid_input: {win.static_grid_input.shape} {win.static_grid_input.dtype} (encoder-res)")
    if win.static_grid_input_categorical is not None:
        print(f"  static_grid_input_cat: {win.static_grid_input_categorical.shape} {win.static_grid_input_categorical.dtype}")
    if win.station_mask_2d is not None:
        print(f"  station_mask_2d: {win.station_mask_2d.shape}")


def main():
    parser = argparse.ArgumentParser(description="Query ML tasks supported by get_windows.")
    parser.add_argument("--token", type=str, required=False, help="Hugging Face token.")
    parser.add_argument("--target-url", type=str, default=None, help="Target grid HF group (e.g. 'chirps').")
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
        gridded_url="era5", spatial_mode="stations", ml_task="downscaling"
    )
    _print_windows(win)

    # --- 2. Grid-to-Grid Forecasting ---
    print("\n--- 2. Grid-to-Grid Forecasting (like GraphCast)---")
    win = ds.get_windows(
        window_size=2, horizon_size=1, stations=test_stations,
        first_date=first_date, last_date=last_date,
        gridded_url="era5", spatial_mode="grid", ml_task="forecasting"
    )
    _print_windows(win)

    # --- 3. Field Correction ---
    print("\n--- 3. Field Correction (ERA5 correction with station anchors) ---")
    win = ds.get_windows(
        window_size=2, horizon_size=1, stations=test_stations,
        first_date=first_date, last_date=last_date,
        gridded_url="era5", spatial_mode="grid", ml_task="correction"
    )
    _print_windows(win)

    # --- 4. Grid-to-Point Forecasting ---
    print("\n--- 4. Grid-to-Point Forecasting ---")
    win = ds.get_windows(
        window_size=2, horizon_size=1, stations=test_stations,
        first_date=first_date, last_date=last_date,
        gridded_url="era5", spatial_mode="stations", ml_task="forecasting"
    )
    _print_windows(win)

    # --- 5. Supervised Grid-to-Finer-Grid Downscaling (if target URL provided) ---
    if args.target_url:
        print(f"\n--- 5. Supervised Downscaling: ERA5 -> {args.target_url} ---")
        win = ds.get_windows(
            window_size=2, horizon_size=1, stations=test_stations,
            first_date=first_date, last_date=last_date,
            gridded_url="era5", spatial_mode="grid", ml_task="downscaling",
            target_gridded_url=args.target_url,
            target_gridded_variables=["precipitation"],
            target_temporal_freq="D"
        )
        _print_windows(win)
    else:
        print("\n--- 5. Supervised Downscaling: SKIPPED (pass --target-url chirps to test) ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

# # Temporal split (temporal validation): train 2019-2021, val 2022, test 2023+
# train_win = ds.get_windows(..., first_date="2019-01-01", last_date="2021-12-31")
# val_win   = ds.get_windows(..., first_date="2022-01-01", last_date="2022-12-31")
# test_win  = ds.get_windows(..., split="test")  # Uses test_set_start (2023-01-01+)

# Spatiotemporal validation
# # Step 1: Hold out spatial test stations
# split = ds.get_spatial_split(test_fraction=0.2, seed=42)

# # Step 2: Further split training stations into train + val
# train_val_split = ds.get_spatial_split(
#     test_fraction=0.15,  # 15% of remaining for validation
#     seed=42,
#     stations=split["train"],  # Only split the training stations
# )

# # Step 3: Three disjoint sets
# train_stations = train_val_split["train"]
# val_stations   = train_val_split["test"]   # Held-out validation stations
# test_stations  = split["test"]              # Held-out test stations

# # Step 4: Combine with temporal splits for the strictest protocol
# train_win = ds.get_windows(stations=train_stations, first_date="2019-01-01", last_date="2022-06-30", ...)
# val_win   = ds.get_windows(stations=val_stations,   first_date="2022-07-01", last_date="2022-12-31", ...)
# test_win  = ds.get_windows(stations=test_stations,   split="test", ...)
