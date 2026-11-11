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
# Covers: download -> pick -> associate
#
# Steps pass data through files (the downloaded waveforms), which the dependency
# graph cannot see, so run them in order:
#   swe run pipeline.yml
#
# Not included: location (swe.location.run_nonlinloc reads a phase file, and no
# step writes the association result to one). Nor, as one step cannot express them: noise windows
# (find_noise_windows needs (start, end) UTCDateTime pairs, which params cannot
# build), magnitudes, focal mechanisms and stress inversion (they need per-event
# waveforms and polarity observations; see swe.magnitude and swe.focal).

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

  - name: load_waveforms
    function: obspy.read
    cache: false
    params:
      pathname_or_url: "data/waveforms/*.ms"

  - name: pick_phases
    function: swe.picking.predict_phase_arrivals
    params:
      stream: ${load_waveforms}
      model_name: "PhaseNet"
      pretrained: "original"
      p_threshold: 0.3
      s_threshold: 0.3
      batch_size: 256

  - name: associate_events
    function: swe.association.associate_picks
    params:
      station_picks: ${pick_phases}
      station_list: ["STA1", "STA2", "STA3"]
      location_dict:
        STA1: [-117.0, 35.0, 0.5]
        STA2: [-117.1, 35.1, 0.6]
        STA3: [-117.2, 35.2, 0.7]
      dbscan_eps: 0.5
      dbscan_min_samples: 5
      min_picks_per_cluster: 4

""",
    "detection": """\
# Matched-filter detection pipeline
#
# Builds templates from a marker file and runs EQcorrscan matched-filter
# detection to find repeating seismicity.
#
# Customise the parameters below, then run:
#   swe run pipeline.yml
#
# Not included: locating the detections (locate_by_travel_time locates one
# event from per-station arrival times, which a detection Party does not hold).

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

  - name: build_templates
    function: swe.detection.markers_to_template
    cache: false
    params:
      marker_file_path: "data/templates/picks.mrkr"  # <-- S picks of the template event
      prepick_offset: 2.0
      template_duration: 16.0

  - name: start_date
    function: obspy.UTCDateTime
    params: {year: 2024, month: 1, day: 1}

  - name: end_date
    function: obspy.UTCDateTime
    params: {year: 2024, month: 1, day: 2}

  - name: detect_events
    function: swe.detection.detect_signals
    params:
      templates: ${build_templates.templates}
      template_files: ["data/templates/*.ms"]  # <-- day-long data holding the template event
      station_dict: ${build_templates.station_dict}
      template_length: 16.0
      template_prepick: 2.0
      detection_files_path: "data/waveforms"
      start_date: ${start_date}
      end_date: ${end_date}
      thresh_type: "MAD"
      detect_thresh: 8.0
      trig_int: 8.0
      lowcut: 1.0
      highcut: 15.0
      samp_rate: 100.0
""",
    "stress": """\
# Stress analysis pipeline
#
# Groups events by location and inverts their focal mechanisms for the
# stress tensor of each group, with bootstrap uncertainties.
#
# Customise the parameters below, then run:
#   swe run pipeline.yml
#
# Focal mechanisms are read from a table: solving them from polarities
# (swe.focal.focal_mechanisms_batch) needs per-event azimuth, takeoff-angle
# and polarity arrays, which one step cannot build from a catalog.

steps:
  - name: load_hypocenters
    function: numpy.loadtxt
    cache: false
    params:
      fname: "data/focal_mechanisms.csv"  # <-- header, then x, y, depth (km), strike, dip, rake
      delimiter: ","
      skiprows: 1
      usecols: [0, 1, 2]  # cluster_events uses Euclidean distance: give x, y in km, not degrees
      ndmin: 2

  - name: load_mechanisms
    function: numpy.loadtxt
    cache: false
    params:
      fname: "data/focal_mechanisms.csv"
      delimiter: ","
      skiprows: 1
      usecols: [3, 4, 5]
      ndmin: 2

  - name: cluster_events
    function: swe.stress.cluster_focal_mechanisms
    params:
      data: ${load_hypocenters}
      n_clusters: 1          # <-- number of groups to invert separately

  - name: invert_stress
    function: swe.stress.stress_inversion_with_bootstrap
    params:
      data: ${load_mechanisms}
      labels: ${cluster_events.labels}
      n_bootstrap: 500
""",
    "spectral": """\
# Spectral analysis pipeline
#
# Computes HVSR for site characterisation and fits source spectra
# for stress drops.
#
# Customise the parameters below, then run:
#   swe run pipeline.yml
#
# Not included: coda-Q (estimate_coda_q takes one trace and one origin
# time per call).

steps:
  - name: load_hvsr_station
    function: obspy.read
    cache: false  # file readers re-read: the cache key holds the path, not the file
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

  - name: load_event_waveforms
    function: obspy.read
    cache: false
    params:
      pathname_or_url: "data/event_waveforms/*"  # <-- one S-wave trace per event

  - name: fit_spectra
    function: swe.spectral.spectral_fit_batch
    params:
      traces: ${load_event_waveforms.traces}
      magnitudes: [2.0, 2.5]  # <-- moment magnitude of each trace, in trace order
      freq_min: 0.5
      freq_max: 40.0
      velocity_m_s: 3500.0
      source_model: "madariaga"
""",
    "magnitude": """\
# Magnitude estimation pipeline
#
# Loads an existing catalog.
#
# Local (ML) and duration (Md) magnitudes are computed per event:
# swe.magnitude.attach_local_magnitudes and attach_duration_magnitudes take one
# Event with its response-corrected Stream, which a step cannot pick out of a
# catalog, so those steps are not included. Call them from Python per event.
#
# Run with:
#   swe run pipeline.yml

steps:
  - name: load_catalog
    function: swe.io.read_quakeml
    cache: false
    params:
      path: "data/catalog.xml"
""",
    "uncertainty": """\
# Uncertainty pipeline (picking stage)
#
# Picks phases. The propagate_uncertainty flag marks the step for the
# --propagate-uncertainty option, but no step here produces uncertainty fields
# yet, so the picks carry none.
#
# Run with:
#   swe run pipeline.yml
#
# Not included: location and magnitude (locate_by_travel_time and
# attach_local_magnitudes work on one event per call).

steps:
  - name: load_waveforms
    function: obspy.read
    cache: false
    params:
      pathname_or_url: "data/waveforms/*.ms"

  - name: pick_phases
    function: swe.picking.predict_phase_arrivals
    params:
      stream: ${load_waveforms}
      model_name: "PhaseNet"
      p_threshold: 0.3
      s_threshold: 0.3
    propagate_uncertainty: true
""",
    "roughness": """\
# Roughness-frequency scaling analysis pipeline
#
# Computes instantaneous-frequency time series from arrivals, surface
# roughness PSD from a profile, and tests the tribological scaling
# law f = V_slip / lambda_dominant.
#
# Customise the parameters below, then run:
#   swe run pipeline.yml
#
# Not included: the spatial frequency-roughness correlation. It needs one
# frequency per event, aligned with the event locations; the frequency series
# keeps only the measurements that pass its filters, so they do not align.

steps:
  - name: load_arrivals
    function: obspy.read
    cache: false
    params:
      pathname_or_url: "data/arrivals/*"  # <-- one trace per arrival

  - name: load_pick_times
    function: numpy.loadtxt
    cache: false
    params:
      fname: "data/pick_times.txt"  # <-- one matplotlib date number per trace, in trace order
      ndmin: 1

  - name: frequency_time_series
    function: swe.roughness.compute_frequency_time_series
    params:
      traces: ${load_arrivals.traces}
      pick_times: ${load_pick_times}
      halfwin: 0.06
      bin_width_days: 1.0
      min_frequency: 1.0
      max_frequency: 100.0

  - name: load_profile_x
    function: numpy.loadtxt
    cache: false
    params:
      fname: "data/roughness_profile.csv"  # <-- x, z columns (m), no header row
      ndmin: 1
      delimiter: ","
      usecols: 0

  - name: load_profile_z
    function: numpy.loadtxt
    cache: false
    params:
      fname: "data/roughness_profile.csv"
      delimiter: ","
      usecols: 1
      ndmin: 1

  - name: roughness_psd
    function: swe.roughness.compute_roughness_psd_1d
    params:
      x: ${load_profile_x}
      z: ${load_profile_z}
      detrend: true
      window: "hann"

  - name: load_slip_times
    function: numpy.loadtxt
    cache: false
    params:
      fname: "data/slip_velocity.csv"  # <-- time (matplotlib date number), velocity; no header
      ndmin: 1
      delimiter: ","
      usecols: 0

  - name: load_slip_velocity
    function: numpy.loadtxt
    cache: false
    params:
      fname: "data/slip_velocity.csv"
      delimiter: ","
      usecols: 1
      ndmin: 1

  - name: fit_scaling
    function: swe.roughness.fit_scaling_law
    params:
      frequency_ts: ${frequency_time_series}
      slip_velocity_times: ${load_slip_times}
      slip_velocity_values: ${load_slip_velocity}
      velocity_unit: "m/day"
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
