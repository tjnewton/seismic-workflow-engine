"""Built-in pipeline YAML templates for common seismic workflows.

Each template is a minimal, well-commented YAML configuration that users can
scaffold via ``swe pipeline template <workflow>`` and then customise for their
data.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["TEMPLATES", "get_template", "list_templates"]

TEMPLATES: dict[str, str] = {
    "full": """\
# Full end-to-end seismic analysis pipeline
#
# Covers: download -> noise -> pick -> associate -> detect -> locate -> magnitude -> focal -> stress
#
# Customise the parameters below for your network and time window, then run:
#   swe run pipeline.yml --parallel

steps:
  - name: download_waveforms
    function: swe.io.fdsn.get_waveforms
    params:
      network: "XX"          # <-- your network code
      stations: ["STA1", "STA2", "STA3"]
      channels: ["BHZ", "BHN", "BHE"]
      start_time: "2024-01-01T00:00:00"
      end_time: "2024-01-01T02:00:00"
      output_dir: "data/waveforms"
      client: "IRIS"

  - name: classify_noise
    function: swe.noise.find_noise_windows
    params:
      filepaths: "data/waveforms/"
      detection_parameters: [15, 250, 3.0, 240]
      time_periods:
        - ["2024-01-01T00:00:00", "2024-01-01T02:00:00"]
      bandpass_filter: true
      freqmin: 1.0
      freqmax: 20.0

  - name: pick_phases
    function: swe.picking.predict_phase_arrivals
    params:
      stream_dir: "data/waveforms/"
      model_name: "PhaseNet"
      pretrained: "original"
      p_threshold: 0.3
      s_threshold: 0.3
      batch_size: 256

  - name: associate_events
    function: swe.association.associate_picks
    params:
      station_picks: ${pick_phases}
      stations: ["STA1", "STA2", "STA3"]
      location_dict:
        STA1: [-117.0, 35.0, 0.5]
        STA2: [-117.1, 35.1, 0.6]
        STA3: [-117.2, 35.2, 0.7]
      dbscan_eps: 0.5
      dbscan_min_samples: 5
      min_picks_per_cluster: 4

  - name: locate_events
    function: swe.location.run_nonlinloc
    params:
      control_file: "config/nlloc.conf"

  - name: estimate_magnitudes
    function: swe.magnitude.attach_magnitudes_to_catalog
    params:
      catalog: ${locate_events}
      stream_dir: "data/waveforms/"

  - name: solve_focal
    function: swe.focal.focal_mechanisms_batch
    params:
      catalog: ${estimate_magnitudes}
      stream_dir: "data/waveforms/"
      min_observations: 6
      grid_spacing: 5.0

  - name: invert_stress
    function: swe.stress.stress_inversion
    params:
      focal_data: ${solve_focal.mechanisms}
      n_iterations: 2000
""",
    "detection": """\
# Matched-filter detection pipeline
#
# Builds templates from marker files and runs EQcorrscan matched-filter
# detection to find repeating seismicity.
#
# Customise the parameters below, then run:
#   swe run pipeline.yml

steps:
  - name: download_waveforms
    function: swe.io.fdsn.get_waveforms
    params:
      network: "XX"          # <-- your network code
      stations: ["STA1", "STA2"]
      channels: ["BHZ"]
      start_time: "2024-01-01T00:00:00"
      end_time: "2024-01-02T00:00:00"
      output_dir: "data/waveforms"
      client: "IRIS"

  - name: pick_phases
    function: swe.picking.predict_phase_arrivals
    params:
      stream_dir: "data/waveforms/"
      model_name: "PhaseNet"
      p_threshold: 0.3
      s_threshold: 0.3

  - name: detect_events
    function: swe.detection.detect_signals
    params:
      templates: ${pick_phases.templates}
      template_files: "data/templates/"
      station_dict: ${pick_phases.station_dict}
      template_length: 16.0
      prepick: 2.0
      data_dir: "data/waveforms"
      start_date: "2024-01-01"
      end_date: "2024-01-02"
      thresh_type: "MAD"
      detect_thresh: 8.0
      trig_int: 8.0
      lowcut: 1.0
      highcut: 15.0
      samp_rate: 100.0

  - name: locate_detections
    function: swe.location.locate_by_travel_time
    params:
      catalog: ${detect_events}
      velocity: 5.0
      grid_spacing: 0.01
""",
    "stress": """\
# Stress analysis pipeline
#
# Computes focal mechanisms from polarity observations and inverts for
# the regional stress tensor.
#
# Customise the parameters below, then run:
#   swe run pipeline.yml

steps:
  - name: load_catalog
    function: swe.io.read_quakeml
    cache: false
    params:
      path: "data/catalog.xml"

  - name: solve_focal
    function: swe.focal.focal_mechanisms_batch
    params:
      catalog: ${load_catalog}
      stream_dir: "data/waveforms/"
      min_observations: 6
      grid_spacing: 5.0

  - name: moment_tensors
    function: swe.moment_tensor.moment_tensor_batch
    params:
      events: ${solve_focal.events}
      constraint: "deviatoric"
      max_misfit: 0.2
      min_polarities: 8

  - name: invert_stress
    function: swe.stress.stress_inversion
    params:
      focal_data: ${solve_focal.mechanisms}
      n_iterations: 2000
      n_bootstrap: 500
""",
    "spectral": """\
# Spectral analysis pipeline
#
# Computes HVSR for site characterisation, fits Brune source models,
# and estimates coda-Q for attenuation studies.
#
# Customise the parameters below, then run:
#   swe run pipeline.yml

steps:
  - name: load_catalog
    function: swe.io.read_quakeml
    cache: false  # file readers re-read: the cache key holds the path, not the file
    params:
      path: "data/catalog.xml"

  - name: load_hvsr_station
    function: obspy.read
    cache: false
    params:
      pathname_or_url: "data/waveforms/XX.STA1.*"  # one station, one trace per component

  - name: compute_hvsr
    function: swe.spectral.compute_hvsr
    cache: false  # its Stream param has no stable cache key
    params:
      stream: ${load_hvsr_station}
      window_length: 40.0
      smoothing_bandwidth: 40.0
      freq_min: 0.1
      freq_max: 25.0

  - name: estimate_coda_q
    function: swe.spectral.estimate_coda_q
    params:
      stream_dir: "data/waveforms/"
      catalog: ${load_catalog}
      freq_bands:
        - [1.0, 2.0]
        - [2.0, 4.0]
        - [4.0, 8.0]
        - [8.0, 16.0]
      window_start: 2.0
      window_length: 10.0

  - name: fit_spectra
    function: swe.spectral.spectral_fit_batch
    params:
      traces: "data/waveforms/"
      magnitudes: ${load_catalog.magnitudes}
      freq_min: 0.5
      freq_max: 40.0
      velocity_m_s: 3500.0
      source_model: "madariaga"
""",
    "magnitude": """\
# Magnitude estimation pipeline
#
# Loads an existing catalog and computes local (ML) and duration (Md)
# magnitudes from waveform data.
#
# Customise the parameters below, then run:
#   swe run pipeline.yml

steps:
  - name: load_catalog
    function: swe.io.read_quakeml
    cache: false
    params:
      path: "data/catalog.xml"

  - name: compute_ml
    function: swe.magnitude.attach_local_magnitudes
    params:
      catalog: ${load_catalog}
      stream_dir: "data/waveforms/"
      response_dir: "data/responses/"

  - name: compute_md
    function: swe.magnitude.attach_duration_magnitudes
    params:
      catalog: ${compute_ml}
      stream_dir: "data/waveforms/"

  - name: save_catalog
    function: swe.io.write_quakeml
    params:
      catalog: ${compute_md}
      path: "output/catalog_with_mags.xml"
""",
    "uncertainty": """\
# Uncertainty propagation pipeline
#
# Demonstrates Monte Carlo uncertainty propagation from picks through
# location to magnitude estimation.
#
# Run with:
#   swe run pipeline.yml --propagate-uncertainty

steps:
  - name: pick_phases
    function: swe.picking.predict_phase_arrivals
    params:
      stream_dir: "data/waveforms/"
      model_name: "PhaseNet"
      p_threshold: 0.3
      s_threshold: 0.3
    propagate_uncertainty: true

  - name: locate_events
    function: swe.location.locate_by_travel_time
    params:
      picks: ${pick_phases}
      velocity: 5.0
      grid_spacing: 0.01
    propagate_uncertainty: true

  - name: estimate_magnitudes
    function: swe.magnitude.attach_local_magnitudes
    params:
      catalog: ${locate_events}
      stream_dir: "data/waveforms/"
    propagate_uncertainty: true
""",
    "roughness": """\
# Roughness-frequency scaling analysis pipeline
#
# Computes instantaneous-frequency time series from arrivals, surface
# roughness PSD from elevation data, and tests the tribological scaling
# law f = V_slip / lambda_dominant.
#
# Customise the parameters below, then run:
#   swe run pipeline.yml

steps:
  - name: load_catalog
    function: swe.io.read_quakeml
    cache: false
    params:
      path: "data/catalog.xml"

  - name: frequency_time_series
    function: swe.roughness.compute_frequency_time_series
    params:
      traces: "data/waveforms/"
      pick_times: ${load_catalog.picks}
      halfwin: 0.06
      bin_width_days: 1.0
      min_frequency: 1.0
      max_frequency: 100.0

  - name: roughness_psd
    function: swe.roughness.compute_roughness_psd_1d
    params:
      profile_path: "data/roughness_profile.csv"  # <-- x, z columns
      detrend: true
      window: "hann"

  - name: fit_scaling
    function: swe.roughness.fit_scaling_law
    params:
      frequency_ts: ${frequency_time_series}
      slip_velocity_path: "data/slip_velocity.csv"  # <-- time, velocity columns
      velocity_unit: "m/day"

  - name: spatial_correlation
    function: swe.roughness.correlate_frequency_roughness
    params:
      event_locations: ${load_catalog.locations}
      event_frequencies: ${frequency_time_series.all_frequencies}
      elevation_raster: "data/dem.tif"
      roughness_metric: "rms_height"
      neighborhood_radius: 5.0
""",
}


def list_templates() -> list[str]:
    """Return sorted list of available template names.

    Returns
    -------
    list of str
        Template names that can be passed to :func:`get_template`.
    """
    return sorted(TEMPLATES)


def get_template(name: str) -> str:
    """Retrieve a pipeline template by name.

    Parameters
    ----------
    name : str
        Template name (see :func:`list_templates`).

    Returns
    -------
    str
        YAML content for the requested template.

    Raises
    ------
    KeyError
        If *name* is not a known template.
    """
    if name not in TEMPLATES:
        available = ", ".join(list_templates())
        raise KeyError(f"Unknown template {name!r}. Available: {available}")
    return TEMPLATES[name]
