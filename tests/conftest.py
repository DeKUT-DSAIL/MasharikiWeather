"""
Pytest configuration and global environment setups for MasharikiWeather.
"""

from __future__ import annotations

import os

# Limit OpenBLAS / OpenMP threading to avoid system pthread starvation on large HPC nodes
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
