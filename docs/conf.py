"""Sphinx configuration for the Seismic Workflow Engine documentation."""

from __future__ import annotations

import swe

project = "Seismic Workflow Engine"
author = "Seismic Workflow Engine contributors"
release = swe.__version__
copyright = "2020-2026, Seismic Workflow Engine contributors"  # noqa: A001

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
    "numpydoc",
    "nbsphinx",
    "sphinxcontrib.mermaid",
]

# nbsphinx settings - render pre-executed notebooks only
nbsphinx_execute = "never"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "SWE"
html_static_path = ["_static"]

# Napoleon / numpydoc settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
numpydoc_show_class_members = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "obspy": ("https://docs.obspy.org/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# MyST settings
myst_enable_extensions = ["colon_fence"]
