import os
import sys

# Ensure repository root is on sys.path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "MasharikiWeather"
copyright = "2026, Centre for Data Science and Artificial Intelligence (DSAIL), Dedan Kimathi University of Technology"
author = "Teofilo Ligawa, Samuel Mbugua, James Muiru, Hilda Muthoni, and Ciira wa Maina"
version = "0.1.9"
release = "0.1.9"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Napoleon extension settings for Google and NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3

# Mock optional imports that may lack system C-libraries in standard environments
autodoc_mock_imports = [
    "cfgrib",
    "eccodes",
    "cartopy",
    "torch",
    "torch_geometric",
    "earthaccess",
]

# HTML output configuration
html_theme = "sphinx_rtd_theme"
html_static_path = []
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
