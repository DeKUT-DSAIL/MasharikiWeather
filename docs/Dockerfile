FROM python:3.11-slim
# or desired python version

ENV DEBIAN_FRONTEND=noninteractive

# 1. System Dependencies
# Added build-essential (gcc, g++, make) which is required to compile ecCodes
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential cmake curl git wget \
    gdal-bin libgdal-dev \
    libproj-dev proj-data proj-bin \
    libgeos-dev libhdf5-dev libnetcdf-dev libpng-dev libopenjp2-7 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Build ecCodes (2.39.0) - Required for parsing GRIB files via cfgrib
WORKDIR /opt
RUN wget https://confluence.ecmwf.int/download/attachments/45757960/eccodes-2.39.0-Source.tar.gz && \
    tar -xzf eccodes-2.39.0-Source.tar.gz && \
    cd eccodes-2.39.0-Source && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DENABLE_FORTRAN=OFF \
        -DENABLE_PYTHON=OFF \
        -DENABLE_JPG=ON \
        -DENABLE_NETCDF=ON \
        .. && \
    make -j"$(nproc)" && make install && \
    cd /opt && rm -rf eccodes-2.39.0-Source*

# 3. Environment Variables for ecCodes
ENV ECCODES_DIR=/usr/local
ENV PATH="${ECCODES_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ECCODES_DIR}/lib:${LD_LIBRARY_PATH}"
ENV ECCODES_DEFINITION_PATH="${ECCODES_DIR}/share/eccodes/definitions"
ENV ECCODES_SAMPLES_PATH="${ECCODES_DIR}/share/eccodes/samples"

# 4. GDAL/PROJ Environment
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

# 5. Fast Python Package Installer (uv)
# We can use UV_SYSTEM_PYTHON directly since this is a dedicated Python container
ENV UV_SYSTEM_PYTHON=1
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# 6. Core Geospatial C-Bindings
RUN uv pip install --no-build-isolation --no-cache \
    "numpy<2" \
    "gdal==$(gdal-config --version)"

# 7. Core Data Pipelines, Orchestration, & APIs (Zero ML Bloat)
# Added dagster for orchestration
RUN uv pip install --no-cache \
    "xarray" "netCDF4" "h5netcdf" "zarr" "dask[complete]" "cfgrib" \
    "pandas" "geopandas" "pyarrow" \
    "cdsapi" "earthaccess" "python-dotenv" \
    "dagster" "masharikiweather"

WORKDIR /workspace
CMD ["/bin/bash"]
