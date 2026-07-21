Installation
============

Conda + pip (recommended)
-------------------------

SWE depends on ObsPy and EQcorrscan, which are best installed via conda::

    conda env create -f environment.yml
    conda activate swe
    pip install -e . --no-deps

This installs SWE in editable mode with all core dependencies from conda.

Optional extras
---------------

SWE supports optional features via pip extras:

- **picking**: SeisBench deep-learning pickers::

      pip install "seismic-workflow-engine[picking]"

- **association**: PyOcto and GaMMA associators::

      pip install "seismic-workflow-engine[association]"

- **topo**: Point-cloud and raster topography::

      pip install "seismic-workflow-engine[topo]"

- **velocity**: Travel-time computation with ttcrpy::

      pip install "seismic-workflow-engine[velocity]"

- **docs**: Documentation build tools::

      pip install "seismic-workflow-engine[docs]"

- **all**: All optional dependencies::

      pip install "seismic-workflow-engine[all]"

Development
-----------

For development, also install test dependencies::

    pip install -e ".[dev]" --no-deps
    pytest tests/ -v
