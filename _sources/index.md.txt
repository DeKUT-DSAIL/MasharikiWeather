# MasharikiWeather

MasharikiWeather is a Python library for downloading, processing, and loading weather and climate data for Africa.

---

## Features

* **Data Extraction:** Downloads station observations, satellite feeds, and gridded reanalysis with rate limiting and retries.
* **Credentials:** Reads API keys from arguments, environment variables, or config files.
* **Dataset Interface:** Loads station data from Parquet files and gridded data from Zarr directories.
* **Resampling:** Aggregates time series with sum for precipitation and mean for state variables.
* **Topography:** Slices elevation, slope, and aspect angles for station coordinates.

---

## Contents

```{toctree}
:maxdepth: 2

architecture
extraction
credentials
dataset_user_guide
```

---

## Citation

```bibtex
@misc{masharikiweather2026,
  author = {Teofilo Ligawa and Samuel Mbugua and James Muiru and Hilda Muthoni and Ciira {wa Maina}},
  title = {MasharikiWeather: A Machine-Learning-Ready Weather and Climate Dataset for Africa},
  year = {2026},
  institution = {Centre for Data Science and Artificial Intelligence (DSAIL), Dedan Kimathi University of Technology},
  publisher = {Dedan Kimathi University of Technology},
  url = {https://github.com/DeKUT-DSAIL/MasharikiWeather}
}
```
