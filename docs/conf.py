"""Sphinx configuration for the Seismic Workflow Engine documentation."""

from __future__ import annotations

import swe

project = "Seismic Workflow Engine"
author = "Seismic Workflow Engine contributors"
release = swe.__version__
copyright = "2020-2026, Seismic Workflow Engine contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
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

# Napoleon settings (NumPy-style docstrings). numpydoc is intentionally not
# enabled alongside napoleon: both render the docstring ``Attributes`` section
# into ``.. attribute::`` directives, which double-documents NamedTuple/dataclass
# fields and breaks the strict (-W) build.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
# Render ``Attributes`` as inline :ivar: fields rather than standalone
# ``.. attribute::`` directives, so they don't duplicate the attributes autodoc
# already emits for NamedTuple/dataclass fields (which breaks the -W build).
napoleon_use_ivar = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# NDArray / Origin / UTCDateTime are imported only under ``if TYPE_CHECKING:``
# (with ``from __future__ import annotations``), so sphinx-autodoc-typehints
# cannot resolve them at doc-build time. Those forward references are benign;
# suppress just that warning category so the strict (-W) build stays clean.
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference"]

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
