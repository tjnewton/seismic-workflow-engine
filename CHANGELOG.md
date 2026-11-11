# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **`swe.topo` morphometry**: terrain shape on gridded elevation, using signed grid spacings so north-up rasters work.
  - `slope_aspect`: slope and downslope aspect (clockwise from north).
  - `local_plane_fit`: moving-window least-squares planes with slope, aspect and residual standard deviation; NaN nodata stays local.
  - `curvature`: Laplacian by second central differences, with optional isotropic Gaussian pre-smoothing in length units (NaN spreads over the smoothing footprint rather than biasing hole edges).
  - `faceted_template`: a regular n-sided apex landform; `base` is twice the apothem, one edge faces row 0 at zero `rotation` (a square is grid-aligned), and `dy` matches non-square cells.
  - `template_correlation`: local normalised cross-correlation by FFT; windows containing NaN are NaN.
  - `grid_spacing`: signed `(dx, dy)` of an `ElevationGrid`, rejecting non-uniform, zero-spaced, non-square transposed or geographic grids (CRS checked with pyproj or rasterio when installed; non-metre units warn).
  - Acceptance tests: with 2 m noise, curvature correlation finds a planted faceted landform and rejects a dome and a ridge of the same size; a landform turned 30° is found at the right rotation by a rotation search.
- **`swe.scattering`** (new): Born single-scattering by a compact elastic inclusion in a homogeneous full space.
  - `born_radiation_pattern` gives the P→P and P→S patterns in density and P/S velocity perturbations (Wu & Aki 1985).
  - `born_point_scattering` gives the far-field scattered/incident amplitude, with validity flags: Rayleigh regime, weak contrast, and far field (`kr > 10` and beyond the body's Fraunhofer distance). Physically impossible hosts are rejected.
  - Form factors: `sphere_form_factor` (analytic) and `voxel_form_factor` (any body given by an inside-test).
  - Tested against the exact small-fluid-sphere (Rayleigh) result in the fluid limit; a faceted polyhedral inclusion matches its volume-equivalent sphere at ka < 0.3.
  - `born_point_scattering` keeps the sign and phase of the form factor. `voxel_form_factor` uses the `exp(-i q·x)` convention of `scattering_wavenumber` and evaluates many `q` at once, and rejects a body that would be clipped by its bounds.
- **`swe.detection` injection-recovery**: `injection_recovery` estimates a detector's detection probability over a parameter grid, with Clopper–Pearson intervals (`clopper_pearson`). It is seedable through `random_state` and accepts any iterable grid.
  - `inject` plants a signal in an array or raster; `inject_trace` does the same in an ObsPy `Trace`.
  - Masked gaps stay masked, and masked signal samples are not injected. An unmasked injected trace can be written back to MiniSEED.
  - Each grid entry's random stream is seeded from `random_state` and its own canonicalised parameters (whole numbers as exact ints; every container type-tagged, so different structures never share a stream), so a row's result does not depend on the other entries, their order or numeric types. The seed key is stable across NumPy versions; NumPy does not promise identical draws across versions (NEP 19).
- **`swe.io`**: station inventories by region and epoch, and event catalogs.
  - `get_stations_in_region`: FDSN station query for a lat/lon box and epoch, at any detail level. It drops excluded networks; by default that is the synthetic `SY` network (`DEFAULT_EXCLUDED_NETWORKS`). A bare string such as `"SY"` counts as one code, and a network named explicitly (`network="SY"`) is never excluded.
  - `inventory_to_dataframe`: one row per station or per channel, with ISO-8601 epochs.
  - `get_events` returns an empty `Catalog` when nothing matches, and passes FDSN query parameters through unchanged. `catalog_to_dataframe` flattens a catalog to one row per event.
  - `load_response`: one shared reader for response files.
  - Every `swe.io.fdsn` function now accepts an ObsPy data-centre key, a base URL (e.g. `https://service.earthscope.org`), or a ready `Client` instance.
- **Leak guard**: commit messages are now scanned, `#` lines included (`git commit -m` keeps them). The `git commit -v` diff below the scissors line is ignored, whatever `core.commentChar` is, with LF or CRLF line endings.
  - `scripts/leak_guard.py --commit-msg` runs as a commit-msg hook, and `--push-log` checks every local commit not yet on the target remote, which covers multi-branch pushes and commits already on another remote.
  - In CI, `--log` covers the commits of a pull request and of every push. Push runs get their own concurrency group, so a newer run never cancels them, running or pending, and no pushed range goes unscanned.
  - Re-run `pre-commit install` in existing clones to get the commit-msg hook.
  - `pre-commit install` now installs the pre-commit, commit-msg and pre-push hooks together.
- **`swe.roughness`**: roughness-frequency scaling analysis: instantaneous-frequency time series, 1-D/2-D surface roughness PSD, spatial frequency-roughness correlation, forward models (asperity contact + structural resonance), and 6 plotting functions.
  - `roughness` pipeline template added (`swe pipeline template roughness`).
  - 58 new test functions.

### Fixed
- **Pipeline checkpoints**: a cached or resumed step output now loads back with the type the step returned. Before, any value JSON could encode was stored as JSON, so a NamedTuple result (such as `HVSRResult`), a tuple or a numpy array came back as a list, and int keys as strings; code using `result.f0` then failed on a cache hit or resume.
  - JSON is now used only for values built purely from JSON-native types (dict with str keys, list, str, int, float, bool, None, no NaN); everything else is pickled, HMAC-signed as before.
  - Re-serializing a step removes its earlier output in the other format, so a stale file cannot shadow the new one.
  - A value pickle cannot handle (such as an instance of a class defined inside a function) is stored with the previous lossy JSON encoder, with a warning, as before this change. A disk error still propagates.
  - A cached checkpoint that cannot be loaded (for example a pickled class renamed since) is a cache miss, not an error.
  - Pickled checkpoints carry the existing per-install HMAC and SWE version check, so more outputs than before are tied to the install that wrote them: a resume after an upgrade or from another install rejects them, with an error naming the step. Start a new run.
  - The checkpoint format is versioned (`serialization.OUTPUT_FORMAT`): runs record it in their manifest, and the cache key includes it, so caches written before this fix are recomputed once rather than served lossy. Resuming a run written before this change is refused with a clear error; start a new run instead.
- **Step references**: `${step}` (an earlier step's whole output) now resolves at run time and creates a DAG dependency. Before, only `${step.key}` did, while validation already accepted `${step}`, so a dry run passed and the run received the literal string. A bare `${NAME}` where NAME is no pipeline step stays literal, as before (for example `${HOME}`). A Catalog or Stream passed by `${step}` has no exact cache key, so its step is uncacheable (see Pipeline cache keys).
- **Pipeline cache keys** are exact, with memoization semantics (as for `functools.lru_cache`): two values that are not equal, compared with their exact types (arrays by dtype, shape and bytes; floats by bit pattern), never share a key. Memory layout, aliasing, writeability and set iteration order are not part of that contract (a few are keyed anyway, which only causes misses). Only exact types that cannot carry unkeyed instance state are keyed, each with its type: `None`, `bool`, `int`, `float`, `str`; `list`, `tuple`, `dict` (in insertion order), `set`, `frozenset` of keyable values (int and str dict keys differ); plain numpy arrays (no subclass, no object dtype) by dtype, shape, memory order, writeability and a SHA-256 of their bytes; numpy scalars without object fields, with their type; `UTCDateTime` to the nanosecond when no attribute is set on the instance; concrete `pathlib` paths and `Decimal` by text; dates, times and datetimes, naive or with a fixed `datetime.timezone` offset. Anything else, including dataclasses, NamedTuples, masked arrays, enums, IANA zones, ObsPy objects and subclasses of these types, makes the step uncacheable, and its cache lookup is skipped. Before, lossy JSON and a `repr()` fallback (which elides large arrays and omits an Event's origins) let different inputs share a key and be served the wrong cached output.
- **Templates**: the `spectral` template's HVSR step passed `stream_dir`, which `compute_hvsr` does not accept; it now reads one station with `obspy.read` and passes the stream as `${load_hvsr_station}`. File-reading steps in every template (`read_quakeml`, `obspy.read`) set `cache: false`, since the cache key holds the path, not the file; so does `compute_hvsr`, whose Stream param has no stable key.
- **`swe.spectral`**:
  - `compute_hvsr` rounds the window length to samples instead of truncating (40.3 s at 50 Hz is 2015 samples, not 2014). Sample counts are Python ints even for np.float64 inputs under numpy 1.x.
  - The anti-trigger's record-length check includes the band-pass settle zones, and the all-windows-rejected message names them, instead of blaming the LTA alone.
  - HVSR and SSR site labels carry the location code when it is set (`NET.STA.LOC`), so co-located sensors are told apart; with an empty location code the label is `NET.STA` as before.
  - `compute_ssr` documents `zero_gap=None` for synthetics that are exactly zero before the first arrival.
- **`swe.pipeline.provenance.save_provenance`**: concurrent saves could lose records. The file lock was released before the buffered JSON was flushed, so the write landed at `close()`, after unlock, and the next writer read stale content. The data is now flushed before unlocking, and records are serialized before the file is truncated, so a record JSON cannot encode no longer wipes the saved ones. The existing concurrency test, `TestBug3ProvenanceFileLocking`, failed intermittently in CI because of this.
- **`swe.io.get_response_files` / response removal**: downloaded responses could never be loaded. `get_response_files` wrote StationXML under a `RESP.{net}.{sta}..{chan}` name, which `swe.noise` and `swe.picking` then parsed with `format="RESP"`.
  - Responses are now one StationXML file per channel returned, named with the real codes: `{net}.{sta}.{loc}.{chan}.xml`. Pattern requests (`*`, `BH?`) and separate per-channel downloads no longer collide or overwrite each other. A channel with several epochs keeps them all in one file. A later download of the same channel replaces its file.
  - Both modules read through the shared `swe.io.load_response`. It returns the first file that covers the trace (network, station, location, channel and epoch), trying the exact channel first, then other station files, then legacy `RESP.*`.
  - Formats are detected from content; a legacy file that detection rejects or misreads is retried as RESP.
  - Unreadable or unrelated files are skipped with a warning.
  - Parsed files are cached by inode, size, mtime and ctime, and the directory is listed only when the exact file misses.
  - Regression tests: a download → load → sensitivity-removal round trip, per-channel downloads, legacy files not hidden by a non-covering new file, and the cache.
- **`swe.detection.matched_filter`**: `make_templates`/`markers_to_template` no longer force pick channel codes to `"...Z"`. This produced non-existent channels (e.g. `SP1`->`SPZ`) for borehole networks whose components are numbered `1/2/3` (such as the Parkfield HRSN), yielding empty templates. Channel codes are now preserved and 3-component expansion honours the instrument's convention (Z/N/E or 1/2/3). Fixes `detect_signals` for borehole data. Regression tests added.
- **`swe.spectral.fit_brune_model`**: the corner-frequency fit is now bounded to the fitted `[freq_min, freq_max]` band, preventing divergence to physically meaningless values when a spectrum lacks a clear corner (e.g. a weak or narrow-band stack).

## [0.3.0] -- 2026-02-27

Feature expansion adding amplitude moment-tensor inversion, advanced stacking, uncertainty visualization, batch processing, and pipeline templates.

### Added
- **`swe.moment_tensor.amplitude_inversion`**: amplitude-based moment tensor inversion using P/SV/SH amplitudes and amplitude ratios. Critical for LFE characterization where first-motion polarities are ambiguous.
  - `moment_tensor_from_amplitudes()`: L2 fit of normalised amplitude patterns.
  - `moment_tensor_from_amplitude_ratios()`: SV/P and SH/P ratio-based inversion (Hardebeck & Shearer, 2003).
  - `AmplitudeMTResult`: result type with variance reduction and component ranges.
- **Advanced stacking methods** in `swe.detection.stacking`:
  - `nth_root_stack()`: sign-preserving nth-root stack for aggressive noise suppression (Muirhead, 1968).
  - `median_stack()`: outlier-robust sample-by-sample median stack.
  - `selective_stack()`: SNR-threshold filtering before stacking with selectable method.
- **`swe.uncertainty.plotting`**: publication-quality uncertainty visualisation:
  - `plot_pick_uncertainty()`: pick time error bars.
  - `plot_location_uncertainty()`: map and depth-section views with covariance ellipses.
  - `plot_magnitude_pdf()`: individual and combined magnitude PDFs.
  - `plot_uncertainty_summary()`: multi-stage uncertainty dashboard.
- **`swe.pipeline.batch`**: parallel batch processing:
  - `batch_process()`: ProcessPoolExecutor (CPU-bound) or ThreadPoolExecutor (I/O-bound) with tqdm progress.
  - `BatchResult`: container with success/failure tracking.
- **`swe.pipeline.templates`**: built-in YAML pipeline templates for 6 workflow types:
  - `full`, `detection`, `stress`, `spectral`, `magnitude`, `uncertainty`.
  - `get_template()` / `list_templates()` public API.
- **`swe.pipeline.validation`**: dry-run mode for pipeline validation:
  - `validate_pipeline()`: checks structure, imports, references, and cycles without executing.
- **`swe.config`**: user-level configuration management:
  - `SWEConfig` dataclass with defaults for FDSN client, backends, output dirs.
  - `load_config()` / `save_config()` for `~/.swe/config.yml`.
- **CLI enhancements**:
  - `swe pipeline template <workflow>`: scaffold starter YAML configs.
  - `swe pipeline list`: list available templates.
  - `swe run --dry-run`: validate pipeline without executing.
- **PyPI publish workflow** (`.github/workflows/publish.yml`) - trusted publisher via GitHub Actions with post-publish install test.
- 1018 test functions across 70+ test files (up from 900/60+), 85% coverage maintained.
- **`swe.uncertainty`**: uncertainty propagation module with Monte Carlo and analytical methods.
- **`swe.spectral`**: spectral analysis: coda-Q estimation, spectral ratios, curve fitting.
- `SWEPick.time_uncertainty_s` field for pick-time uncertainty in seconds.
- `LocationResult.uncertainty` and `LocalMagnitudeResult.magnitude_uncertainty` fields.
- `--propagate-uncertainty` CLI flag for `swe run`.
- `locate_by_travel_time_with_uncertainty()` wrapper for Monte Carlo relocation.
- `propagate_uncertainty` YAML pipeline step option.

## [0.2.0] -- 2026-02-27

Major scope expansion from 11 to 17 subpackages, with a full seismic pipeline from noise classification through moment tensor inversion.

### Added
- **`swe.focal`**: focal mechanism determination: P-wave first-motion polarity measurement, takeoff angle computation, grid-search focal mechanism solver.
- **`swe.magnitude`**: magnitude estimation: local magnitude (ML), duration magnitude (Md), relative magnitude, and catalog integration with ObsPy Event objects.
- **`swe.moment_tensor`**: moment tensor analysis: tensor decomposition, radiation pattern computation, and polarity-based inversion.
- **`swe.pipeline`**: YAML pipeline orchestration: DAG-based step execution, output caching, run checkpointing, provenance tracking, parallel execution of independent steps.
- **`swe.cli`**: command-line interface with 8 subcommands: `run`, `download`, `noise`, `pick`, `associate`, `detect`, `locate`, `plot`.
- Pipeline completion: `party_to_markers`, `elevations_from_raster`, `compute_travel_times`, Vavrycuk stress inversion, QuakeML I/O.
- Sphinx documentation site with Furo theme, nbsphinx tutorials, Mermaid architecture diagrams, and full API reference.
- 6 tutorial notebooks: Ridgecrest pipeline, magnitude estimation, noise classification, stress inversion, velocity model, pipeline YAML demo.
- 724 test functions across 55 test files with 85% coverage enforcement in CI.
- GitHub Actions CI: test matrix (Python 3.10-3.12), Ruff linting, mypy type checking, Sphinx docs build.

### Fixed
- Removed `plt.show()` calls from `swe.noise` plotting functions; they now return `Figure` objects.
- Added missing `__all__` exports to `swe/__init__.py` and `swe/utils/__init__.py`.
- Fixed incorrect function names in README Quick Start.
- Added type annotations for `pivot` parameter in `nearest()` and `nearest_ind()`.

### Changed
- Version bumped to 0.2.0 reflecting scope expansion beyond original 11 subpackages.
- Added `keywords` and `[project.urls]` to `pyproject.toml`.
- Updated pyproject.toml description to reflect full pipeline capabilities.

## [0.1.0] -- 2025-06-01

Initial release migrated from a prior internal research codebase.

### Added
- 11 subpackages: noise, detection, io, location, picking, association, viz, stress, topo, velocity, utils.
- 126 passing tests with synthetic fixtures (no real data required).
- Installable package via `pip install -e .` with optional dependency groups (picking, topo, velocity).
- PEP 561 `py.typed` marker for downstream type-checking.

### Changed
- All modules use `from __future__ import annotations`.
- Replaced hardcoded paths with configurable parameters.
- Replaced `plt.show()` calls with returned `Figure` objects in viz module.
- Replaced deprecated `ax.plot_date()` with `ax.plot()` throughout viz module.

### Removed
- Site-specific configuration and hardcoded file paths from the prior codebase.
- Pickle-based serialization.
