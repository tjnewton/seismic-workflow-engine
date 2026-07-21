# Roadmap

This document outlines future development directions for the Seismic Workflow Engine.
For the history of the initial migration, see the git log and CHANGELOG.md.

## Completed (v0.2.0)

The core seismic pipeline is fully implemented:

- 18 subpackages covering noise -> picking -> association -> detection -> location -> magnitude -> focal -> moment tensor -> stress -> spectral -> uncertainty -> roughness
- YAML pipeline orchestration with DAG execution, caching, checkpointing, provenance
- CLI with 8 subcommands covering the full workflow, including `--propagate-uncertainty`
- 900 test functions, 85% coverage enforcement, tested on Python 3.10-3.12
- Sphinx docs with tutorials, API reference, and architecture diagrams
- Strategy pattern for swappable backends (3 associators, multiple pickers, multiple locators)
- `swe.spectral`: coda-Q estimation, spectral ratios (SSR/HVSR), curve fitting
- `swe.uncertainty`: Monte Carlo pick->location propagation, analytical location->magnitude propagation, pipeline integration

## Completed (v0.3.0)

### User Experience
- [x] `swe pipeline template <workflow>` CLI command - scaffolds starter YAML configs for 7 workflow types (full, detection, stress, spectral, magnitude, uncertainty, roughness)
- [x] `swe pipeline list`: lists available templates
- [x] `swe run --dry-run`: validates pipeline YAML without executing (checks imports, references, cycles)
- [x] Configuration management (`~/.swe/config.yml`) - user defaults for FDSN client, backends, output dirs

### Scientific Capabilities
- [x] Amplitude-based moment tensor inversion (`swe.moment_tensor.amplitude_inversion`) - fits P/SV/SH amplitudes and amplitude ratios; critical for LFE characterization
- [x] Advanced stacking methods (`swe.detection.stacking`) - nth-root, median, and SNR-selective stacking for weak-event detection
- [x] Uncertainty visualization (`swe.uncertainty.plotting`) - pick error bars, location ellipsoids, magnitude PDFs, summary dashboards

### Infrastructure
- [x] Batch processing (`swe.pipeline.batch`) - ProcessPoolExecutor for CPU-bound, ThreadPoolExecutor for I/O-bound, with progress tracking
- [x] ~~PyPI publish workflow~~ - removed to reduce GitHub Actions usage; to be re-added before first PyPI release
- [x] 1018 tests passing, 85% coverage maintained

## Near-Term

### Documentation
- [ ] Fill LFE catalog notebook
- [ ] Comparison notebooks: STA/LTA vs. SeisBench pickers, DBSCAN vs. PyOcto vs. GaMMA
- [ ] Expand API docs with usage examples for focal, magnitude, moment_tensor modules
- [ ] Integrate CHANGELOG into Sphinx docs
- [ ] Troubleshooting guide

### Community
- [ ] Publish to PyPI (`pip install seismic-workflow-engine`) - CI/CD workflows removed for now; re-add publish workflow before first release
- [ ] Conda-forge recipe
- [ ] JOSS / SoftwareX software paper

## Medium-Term

### `swe init` Project Scaffolding
- Create a project directory with recommended structure, example YAML, and `.swe_runs/` setup

### Site Effects (`swe.site_effects`)
- Spectral ratios and transfer functions
- Site amplification estimation
- Reference site methods

## Long-Term

### 3-D Tomography
Major undertaking - may be better as a separate package that integrates with SWE's velocity module.

### Real-Time Processing
Would require architecture redesign for streaming data. Defer unless a specific research need arises.

## Non-Goals
- Replacing ObsPy's core I/O or signal processing
- Full FDSN web service implementation (use ObsPy clients directly)
- GUI application (CLI + notebooks are the primary interfaces)
