# Seismic Workflow Engine (SWE)

A modular Python toolkit for seismic analysis: from raw waveforms through noise classification, phase picking, association, matched-filter detection, hypocenter location, magnitude estimation, focal mechanisms, moment tensors, and stress inversion.

## Overview

SWE provides a complete seismic analysis pipeline as composable Python modules.
Each module accepts standard ObsPy objects (`Stream`, `Catalog`, `Inventory`) and integrates naturally with existing seismological workflows.
A YAML-driven pipeline system ties everything together with DAG execution, caching, checkpointing, and provenance tracking.

The library was migrated from a prior internal research codebase into a clean, installable package with no hardcoded paths or site-specific configuration.

## Features

### Analysis Modules

| Subpackage | Purpose |
|---|---|
| `swe.noise` | Ambient-noise classification and quality metrics |
| `swe.picking` | Phase picking (STA/LTA, SeisBench deep-learning models) |
| `swe.association` | Phase-to-event association (DBSCAN, PyOcto, GaMMA) |
| `swe.detection` | Matched-filter (template) detection via EQcorrscan |
| `swe.location` | Hypocenter location (NonLinLoc, grid search) |
| `swe.magnitude` | Local (ML) and duration (Md) magnitude estimation |
| `swe.focal` | P-wave polarity, takeoff angles, focal mechanism solving |
| `swe.moment_tensor` | Moment tensor decomposition and inversion |
| `swe.stress` | Stress-tensor inversion from focal mechanisms |
| `swe.velocity` | 1-D/3-D velocity model building and travel-time computation |
| `swe.viz` | Waveform, spectrogram, and spectral plots |
| `swe.io` | Readers/writers for picks, catalogs, and station metadata |
| `swe.spectral` | Spectral analysis (coda-Q, spectral ratios, fitting) |
| `swe.uncertainty` | Uncertainty propagation (pick -> location -> magnitude) |
| `swe.topo` | Topography and point-cloud utilities (LAS/raster) |
| `swe.roughness` | Roughness-frequency scaling, surface PSD, spatial correlation |
| `swe.utils` | Shared helpers (amplitude, SNR, coordinate transforms) |

### Pipeline Orchestration

`swe.pipeline` provides YAML-based workflow orchestration:

- **DAG execution**: steps run in dependency order with automatic resolution
- **Output caching**: skip already-completed steps on re-runs
- **Checkpointing**: resume interrupted pipelines from the last successful step
- **Provenance tracking**: record what ran, when, and with which parameters
- **Parallel execution**: independent steps run concurrently with `--parallel`
- **Uncertainty propagation**: propagate pick-time uncertainties through the pipeline with `--propagate-uncertainty`

### Command-Line Interface

```bash
swe run pipeline.yml                  # Execute a YAML pipeline
swe download -n CI -s WBS -c BHZ \
    --start 2019-07-04T17:30:00 \
    --end 2019-07-04T18:30:00         # Fetch waveforms from FDSN
swe noise -i data/ --start ... \
    --end ...                         # Classify noise windows
swe pick -i data/ --method stalta     # Run phase picking
swe associate -p picks.csv \
    -s stations.csv                   # Associate picks into events
swe detect --marker-file picks.mrkr \
    --template-dir templates/ \
    --data-dir data/ --start ... \
    --end ...                         # Matched-filter detection
swe locate -o nll_picks.obs \
    -c nll_control.txt                # Locate hypocenters
swe plot -i data.ms -t waveform       # Generate plots
```

## Installation

```bash
# 1. Create the conda environment (includes ObsPy, EQcorrscan, etc.)
conda env create -f environment.yml
conda activate swe

# 2. Install the package in editable mode
pip install -e . --no-deps
```

Optional dependency groups:

```bash
pip install -e ".[picking]"       # SeisBench deep-learning pickers
pip install -e ".[association]"   # PyOcto and GaMMA associators
pip install -e ".[topo]"          # laspy + rasterio for topography
pip install -e ".[velocity]"      # ttcrpy travel-time solver
pip install -e ".[all]"           # everything above
pip install -e ".[dev]"           # pytest + coverage
pip install -e ".[docs]"          # Sphinx documentation build
```

## Quick Start

```python
from obspy import read

from swe.io import parse_phase_picks
from swe.location import extract_nll_locations
from swe.viz import plot_stream

# Load waveforms and plot
st = read("example.mseed")
fig = plot_stream(st)
fig.savefig("waveforms.png", dpi=150)

# Parse picks from a NonLinLoc hyp file
picks = parse_phase_picks("picks.mrkr")
locations = extract_nll_locations("nlloc.hyp")
```

### Running a Pipeline

```yaml
# pipeline.yml
steps:
  - name: download_waveforms
    function: swe.io.fdsn.get_waveforms
    params:
      network: "CI"
      stations: ["WBS", "WRC2"]
      start_time: "2019-07-04T17:30:00"
      end_time: "2019-07-04T18:30:00"

  - name: pick_phases
    function: swe.picking.find_picking_windows
    params:
      filepaths: ${download_waveforms}
      method: "sta_lta"
```

```bash
swe run pipeline.yml --parallel
```

## Development

```bash
conda activate swe
pytest tests/ -v                                    # Run the full test suite
pytest tests/ --cov=swe --cov-report=term-missing   # Run with coverage
```

The project enforces 85% code coverage (tested on Python 3.10, 3.11, and 3.12).

**Conventions**: `from __future__ import annotations` in every module, snake_case throughout, NumPy-style docstrings, keyword-only public API arguments, `__all__` exports in every `__init__.py`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

## Documentation

Full documentation is available at [Read the Docs](https://seismic-workflow-engine.readthedocs.io/) or can be built locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/ docs/_build/html
```

## License

MIT
