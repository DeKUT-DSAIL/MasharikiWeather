import xarray as xr
import os
import shutil
from dask.diagnostics import ProgressBar

def rechunk_dataset(input_path: str, output_path: str, time_chunks: int = 24, swap: bool = False):
    """
    Physically re-chunks a local Zarr dataset to Daily (24-step) blocks.
    
    Args:
        input_path (str): Path to the existing (fragmented) Zarr store.
        output_path (str): Path to the new (optimized) Zarr store.
        time_chunks (int): Number of time steps per chunk (default: 24).
        swap (bool): If True, deletes the original input_path after successful re-chunking.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input Zarr not found: {input_path}")
        
    print(f"Optimizing Mashariki I/O: {input_path} -> {output_path}")
    ds = xr.open_zarr(input_path)
    
    # Define new chunks: All spatial dimensions as full (None), time as 24
    chunks = {'time': time_chunks}
    for coord in ds.coords:
        if coord not in ['time']:
            chunks[coord] = -1 # Full scale for spatial dims
            
    # Apply re-chunking
    ds_rechunked = ds.chunk(chunks)
    
    # Configure encoding for Zarr (compression + alignment)
    encoding = {v: {'chunks': (time_chunks, *ds[v].shape[1:])} for v in ds.data_vars}
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with ProgressBar():
        ds_rechunked.to_zarr(output_path, mode='w', encoding=encoding, consolidated=True)
        
    print(f"Optimization complete: {output_path}")
    
    if swap:
        print(f"Swapping storage: Deleting original {input_path}...")
        shutil.rmtree(input_path)
        print("Swap complete. Storage space recovered.")

    return True
