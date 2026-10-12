import xarray as xr
import os
import shutil
from dask.diagnostics import ProgressBar

DEFAULT_TIME_CHUNKS = 24

def get_zarr_encoding(ds: xr.Dataset, time_chunks: int = DEFAULT_TIME_CHUNKS):
    """
    Standardizes encoding for High-Performance Mashariki Zarrs.
    Ensures correct chunk alignment for O(1) Daily temporal fetches.
    
    Builds per-variable chunk tuples based on the variable's actual
    dimensions, not a hardcoded assumption of (time, lat, lon).
    ERA5 variables may include 'level', and some may lack 'time'.
    """
    encoding = {}
    for var in ds.data_vars:
        var_dims = ds[var].dims
        var_shape = ds[var].shape
        # Build chunk tuple: use time_chunks for 'time', full size for everything else
        var_chunks = tuple(
            time_chunks if dim == 'time' else size
            for dim, size in zip(var_dims, var_shape)
        )
        encoding[var] = {
            'chunks': var_chunks,
        }
    return encoding

def rechunk_dataset(input_path: str, output_path: str, time_chunks: int = DEFAULT_TIME_CHUNKS, swap: bool = False):
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
    ds = xr.open_zarr(input_path, decode_timedelta=False, zarr_format=2)
    
    # Define new chunks using ONLY actual dimensions (ds.dims),
    # not ds.coords which includes scalar metadata like 'number', 'step', 'surface'.
    chunks = {}
    for dim in ds.dims:
        if dim == 'time':
            chunks[dim] = time_chunks
        else:
            chunks[dim] = -1  # Full scale for spatial dims
            
    # Apply re-chunking lazily
    ds_rechunked = ds.chunk(chunks)
    
    # Configure encoding for Zarr (compression + alignment)
    encoding = get_zarr_encoding(ds, time_chunks)
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Atomic Write Pattern: Ensure we don't leave partial Zarrs
    temp_path = f"{output_path}.tmp"
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
        
    try:
        with ProgressBar():
            ds_rechunked.to_zarr(temp_path, mode='w', encoding=encoding, consolidated=True, zarr_format=2)
        
        # Rename tmp to final
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.rename(temp_path, output_path)
        print(f"Optimization complete: {output_path}")
        
        if swap:
            print(f"Swapping storage: Deleting original {input_path}...")
            shutil.rmtree(input_path)
            print("Swap complete. Storage space recovered.")
        return True
        
    except Exception as e:
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        print(f"Optimization FAILED: {e}")
        return False
