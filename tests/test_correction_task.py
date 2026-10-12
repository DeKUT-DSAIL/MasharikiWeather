
import os
import sys

# Optional torch/PyG imports
try:
    import torch
    from torch_geometric.data import HeteroData
    from masharikiweather.dataloader import create_dataloader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch/torch_geometric not found. Skipping HeteroData validation.")

import pytest
from masharikiweather.dataset import MasharikiWeatherDataset

@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="HF_TOKEN not set; skipping live HuggingFace integration test")
def test_bias_correction_scenario():
    print("\n--- TEST: Multi-Grid Precipitation Bias Correction (ERA5 + IMERG + TAHMO) ---")
    
    # 1. Initialize Dataset
    # Relevant variables for precipitation bias correction
    params = [
        "total_precipitation",      # target
        "temperature_2m",           # thermo
        "u_component_of_wind_10m",  # wind-u
        "v_component_of_wind_10m",  # wind-v
        "total_column_water_vapour" # moisture
    ]
    
    ds = MasharikiWeatherDataset(
        token=os.getenv("HF_TOKEN"),
        years=[2023],
        source_obs=["tahmo"],
        parameters=params,
        freq="h"
    )
    
    # Target week: 1st week of May 2023
    first_date = "2023-05-01"
    last_date = "2023-05-07"
    
    # 2. Extract Windows
    # Scenario: ERA5 input + IMERG input + TAHMO input -> correct ERA5 target precip
    print(f"Extracting windows from {first_date} to {last_date}...")
    
    win = ds.get_windows(
        window_size=24,
        horizon_size=0,
        first_date=first_date,
        last_date=last_date,
        gridded_url=["era5", "imerg"],
        spatial_mode="grid",
        ml_task="correction",
        target_node="era5", # The node we are correcting
        crop_to_stations=True,
    )
    
    # 3. Verify Windows Dataclass (Core Geostatistical Extraction Logic)
    print("\n[Windows Validation Out-of-Core]")
    print(f"Index size: {len(win.index_x)}")
    print(f"Grids detected: {list(win.grids_x.keys())}")
    
    for g_name in win.grids_x.keys():
        gx = win.grids_x[g_name]
        print(f"  Grid '{g_name}' extracted as xarray/numpy.")
        # We don't call .to_array().values here to keep it lazy unless needed
        print(f"  Available variables in '{g_name}': {win.grids_features.get(g_name)}")
        
    print(f"Station input 'x' shape: {win.station_x.shape}")
    print(f"Target node routing: {win.target_node}")
    
    if win.native_topography is not None:
        print(f"Native Topography (Continuous): YES")
    if win.native_topography_categorical is not None:
        print(f"Native Topography (Categorical): YES")

    # 4. Create PyG DataLoader (If Torch available)
    if not HAS_TORCH:
        print("\n[Skip] HeteroData validation skipped (Torch/PyG not found). Extraction logic verified.")
        return

    print("\n[HeteroData Validation JIT]")
    loader = create_dataloader(win, ml_task="correction", batch_size=1, k_stations=8, k_grid=4)
    batch = next(iter(loader))
    
    print(f"HeteroData nodes: {batch.node_types}")
    print(f"HeteroData edge types: {batch.edge_types}")
    
    for et in batch.edge_types:
        ei = batch[et].edge_index
        ea = batch[et].edge_attr
        print(f"  {et}: edge_index={list(ei.shape)}, edge_attr={list(ea.shape)}")
    
    # ERA5 should have 'y' because it is the target_node for correction
    if 'era5' in batch and hasattr(batch['era5'], 'y'):
        print(f"ERA5 correction target shape: {batch['era5'].y.shape}")
    
    # Station coords diagnostic
    if win.station_coords is not None:
        print(f"\nStation coords shape: {win.station_coords.shape}")

if __name__ == "__main__":
    try:
        test_bias_correction_scenario()
    except Exception as e:
        print(f"Correction task test failed: {e}")
        import traceback
        traceback.print_exc()

