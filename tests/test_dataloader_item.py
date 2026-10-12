from masharikiweather.dataset import MasharikiWeatherDataset
from masharikiweather.dataloader import create_dataloader
import torch

def test_getitem():
    ds = MasharikiWeatherDataset(dummy_test=True)
    windows = ds.get_windows(start_date="2023-01-01", end_date="2023-01-05", 
                             k_days_history=1, horizon=1, val_size=0, 
                             grids=["era5", "imerg"])
    
    loader = create_dataloader(windows, ml_task="downscaling", batch_size=2)
    hetero_data = loader.dataset[0]
    
    print("Graph built successfully.")
    print(hetero_data)
    
    if "era5" in hetero_data.node_types:
        print("era5 node shape:", hetero_data["era5"].x.shape)
        if hasattr(hetero_data["era5"], "static_u"):
            print("era5 static_u shape:", hetero_data["era5"].static_u.shape)
        else:
            print("era5 has NO static_u!")
            
if __name__ == "__main__":
    test_getitem()
