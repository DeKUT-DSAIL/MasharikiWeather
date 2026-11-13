# MasharikiWeather Documentation

Welcome to the documentation for **MasharikiWeather**, an open-source weather and climate data library designed for Africa.

MasharikiWeather provides tools to download, align, and process weather observations, satellite data, and reanalysis products into formats ready for machine learning and statistical analysis.

---

## Key Features

* **Data Extraction:** Downloads data from weather station networks, satellite products, and atmospheric reanalysis models with retries and rate limits.
* **Credentials:** Resolves API keys from function arguments, environment variables, or local configuration files.
* **Dataset Interface:** Loads station data stored as Parquet files and gridded fields stored as Zarr directories.
* **Temporal Resampling:** Resamples time series while preserving rainfall totals (sum) and averaging state variables (mean).
* **Topography:** Provides elevation, slope, and aspect angles for station sites.

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

If you use MasharikiWeather in your research, please cite:

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
