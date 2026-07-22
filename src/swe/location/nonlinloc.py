"""NonLinLoc earthquake location format utilities.

Converts between ObsPy/Snuffler picks and NonLinLoc phase format, parses
NonLinLoc ``.hyp`` output files into structured arrays, and provides a
wrapper for running NonLinLoc as a subprocess.
"""

from __future__ import annotations

import glob
import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from obspy import Catalog, UTCDateTime, read

if TYPE_CHECKING:
    from obspy.core.event import Origin

logger = logging.getLogger(__name__)

# Approximate metres per degree of latitude at the equator (WGS-84).
# Used for converting horizontal uncertainty from metres to degrees.
_METERS_PER_DEG_LAT = 111_320.0


class NLLLocationResult(NamedTuple):
    """Result from parsing a NonLinLoc summary file."""

    pdfs: pd.DataFrame
    """Scatter-point locations and PDF weights."""
    hypocenters: np.ndarray
    """Shape ``(n_events, 4)``: ``[x_km, y_km, z_m, rms]``."""
    uncertainty_summary: pd.DataFrame | None = None
    """Per-event uncertainty summary (columns: horiz_m, depth_m, rms)."""


class HypocenterResult(NamedTuple):
    """A single hypocenter from NonLinLoc output."""

    x_km: float
    """Easting in km."""
    y_km: float
    """Northing in km."""
    z_m: float
    """Depth in metres."""
    horiz_uncertainty_m: float
    """Horizontal uncertainty in metres."""
    depth_uncertainty_m: float = 0.0
    """Vertical (depth) uncertainty in metres from ``sqrt(Cov_zz)``."""
    rms: float = 0.0
    """RMS residual from the NLL QUALITY line."""


def picks_to_nonlinloc(
    marker_file_path: str | Path,
    waveform_files_path: str | Path,
    *,
    output_path: str | Path = "nll_picks.obs",
    sampling_rate: float = 250.0,
    lowcut: float = 20.0,
    highcut: float = 60.0,
) -> Path:
    """Convert Snuffler uncertainty picks to NonLinLoc phase format.

    Reads a Snuffler marker file containing paired uncertainty lines
    (start/end time ranges) and writes a ``.obs`` file suitable for
    NonLinLoc location.

    Parameters
    ----------
    marker_file_path : str or Path
        Snuffler ``.mrkr`` file with P-phase uncertainty picks.
    waveform_files_path : str or Path
        Directory of day-long MiniSEED files (for amplitude extraction).
    output_path : str or Path
        Path for the output NLL phase file.
    sampling_rate : float
        Target sampling rate for amplitude measurement (Hz).
    lowcut, highcut : float
        Bandpass corners for amplitude measurement (Hz).

    Returns
    -------
    Path
        The written ``.obs`` file.
    """
    output_path = Path(output_path)
    event_ids: list[str] = []
    phases_dict: dict[str, list[str]] = {}
    first_event = True

    with open(output_path, "w") as wf, open(marker_file_path) as rf:
        for line in rf:
            fields = line.split()
            if len(fields) < 10:
                continue
            if fields[0] != "phase:":
                continue

            # Phase hint is at field index 8
            phase_hint = fields[8]
            if phase_hint != "P":
                continue

            # NSLC is at field index 4
            nslc_str = fields[4]
            nslc_parts = nslc_str.split(".")
            if len(nslc_parts) < 4:
                continue

            # Event ID is at field index 5
            event_id = fields[5]

            try:
                start_time = UTCDateTime(f"{fields[1]} {fields[2]}")
                end_time = UTCDateTime(
                    f"{fields[3]} {fields[4]}"
                    if fields[3].startswith("20")
                    else f"{fields[1]} {fields[2]}"
                )
            except (TypeError, ValueError):
                continue

            # Detect uncertainty markers by checking if field 3 is a date.
            if fields[3].startswith("20"):
                if len(fields) < 12:
                    continue
                try:
                    start_time = UTCDateTime(f"{fields[1]} {fields[2]}")
                    end_time = UTCDateTime(f"{fields[3]} {fields[4]}")
                except (TypeError, ValueError):
                    continue
                nslc_str = fields[6]
                nslc_parts = nslc_str.split(".")
                if len(nslc_parts) < 4:
                    continue
                event_id = fields[7]
                phase_hint = fields[10]
                if phase_hint != "P":
                    continue
            else:
                start_time = UTCDateTime(f"{fields[1]} {fields[2]}")
                end_time = start_time

            pick_station = nslc_parts[1]
            pick_channel = nslc_parts[3]

            if event_id not in event_ids:
                event_ids.append(event_id)
                phases_dict[event_id] = []
                if first_event:
                    wf.write("# EQEVENT:  Label: EQ001  Loc:  X 10.0  Y 100.0  Z 0.07  OT 0.00\n")
                    first_event = False
                else:
                    wf.write("#\n# EQEVENT:\n")

            sta_chan = f"{pick_station}.{pick_channel}"
            if sta_chan in phases_dict[event_id]:
                continue
            phases_dict[event_id].append(sta_chan)

            if start_time > end_time:
                start_time, end_time = end_time, start_time

            one_sigma = (end_time - start_time) / 2
            pick_time = start_time + one_sigma

            ptp_amp, ptp_period = _measure_amplitude(
                waveform_files_path,
                pick_station,
                pick_channel,
                start_time,
                end_time,
                pick_time,
                sampling_rate=sampling_rate,
                lowcut=lowcut,
                highcut=highcut,
            )

            if np.isnan(ptp_amp):
                logger.debug(
                    "Amplitude measurement returned NaN for %s.%s, using 0.0",
                    pick_station,
                    pick_channel,
                )
                ptp_amp = 0.0
            if np.isnan(ptp_period):
                ptp_period = 0.0

            sec_frac = pick_time.microsecond / 1e6
            sec_frac_str = _format_sec_fraction(pick_time.second, sec_frac)
            nll_line = (
                f"{pick_station:<6} ?    N    ? P      ? "
                f"{pick_time.year}{pick_time.month:02}"
                f"{pick_time.day:02} {pick_time.hour:02}"
                f"{pick_time.minute:02} "
                f"{sec_frac_str}"
                f" GAU {one_sigma:1.2e}  0.00e+00  "
                f"{ptp_amp:1.2e}  {ptp_period:1.2e}    {ptp_amp:1.2e}\n"
            )
            wf.write(nll_line)

    logger.info("Wrote %d events to %s", len(event_ids), output_path)
    return output_path


def extract_nll_locations(
    file_path: str | Path,
) -> NLLLocationResult:
    """Parse a NonLinLoc ``.hyp`` summary file.

    Extracts scatter-point PDFs and hypocenter solutions from a NonLinLoc
    LocSum output file.

    Parameters
    ----------
    file_path : str or Path
        Path to a ``.hyp`` file containing ``SCATTER`` blocks.

    Returns
    -------
    pdfs : pandas.DataFrame
        Columns ``['x', 'y', 'z', 'weight']``: scatter-point locations
        (x/y in km, z in metres) and PDF weights.
    hypocenters : numpy.ndarray
        Shape ``(n_events, 4)``: columns are ``[x_km, y_km, z_m, rms]``.
    """
    file_path = Path(file_path)
    hypocenters: list[list[float]] = []
    pdfs: list[list[float]] = []

    save_flag = False
    scatter_flag = False
    hypocenter: list[float] = []

    with open(file_path) as fh:
        for line in fh:
            if line[:5] == "NLLOC" and "Location completed" in line:
                save_flag = True

            elif line[:10] == "HYPOCENTER":
                parts = line.split()
                hypocenter = [
                    float(parts[2]),
                    float(parts[4]),
                    round(float(parts[6]) * 1000, 4),
                ]

            elif line[:7] == "QUALITY":
                parts = line.split()
                hypocenter.append(float(parts[8]))

            elif line[:7] == "SCATTER":
                scatter_flag = True

            elif line.startswith(" ") and scatter_flag and save_flag:
                parts = line.split()
                pdfs.append(
                    [
                        float(parts[0]),
                        float(parts[1]),
                        round(float(parts[2]) * 1000, 4),
                        float(parts[3]),
                    ]
                )

            elif line[:9] == "END_NLLOC":
                if save_flag and scatter_flag:
                    hypocenters.append(hypocenter)
                save_flag = False
                scatter_flag = False

    pdfs_df = pd.DataFrame(pdfs, columns=["x", "y", "z", "weight"])
    hypocenters_arr = np.asarray(hypocenters) if hypocenters else np.empty((0, 4))

    logger.info(
        "Extracted %d events, %d scatter points from %s",
        len(hypocenters),
        len(pdfs),
        file_path,
    )
    return NLLLocationResult(pdfs_df, hypocenters_arr)


def process_nll_hypocenters(
    file_path: str | Path,
) -> list[HypocenterResult]:
    """Parse accepted hypocenters from a NonLinLoc ``.hyp`` file.

    Parameters
    ----------
    file_path : str or Path
        Path to a ``.hyp`` file.

    Returns
    -------
    list of list of float
        Each entry is ``[x_km, y_km, z_m, horiz_uncertainty_m]``.
    """
    file_path = Path(file_path)
    hypocenters: list[HypocenterResult] = []
    save_flag = False
    _x_km = 0.0
    _y_km = 0.0
    _z_m = 0.0
    _horiz = 0.0
    _depth_unc = 0.0
    _rms = 0.0

    with open(file_path) as fh:
        for line in fh:
            if line[:5] == "NLLOC" and "Location completed" in line:
                save_flag = True
                _depth_unc = 0.0
                _rms = 0.0

            elif line[:10] == "HYPOCENTER" and save_flag:
                parts = line.split()
                _x_km = float(parts[2])
                _y_km = float(parts[4])
                _z_m = round(float(parts[6]) * 1000, 4)

            elif line[:7] == "QUALITY" and save_flag:
                parts = line.split()
                _rms = float(parts[8])

            elif line[:10] == "STATISTICS" and save_flag:
                parts = line.split()
                # Parse Cov_zz from STATISTICS line
                for j, token in enumerate(parts):
                    if token == "CovZZ" and j + 1 < len(parts):
                        cov_zz = float(parts[j + 1])
                        _depth_unc = round(np.sqrt(max(cov_zz, 0.0)) * 1000, 4)
                        break

            elif line[:21] == "QML_OriginUncertainty" and save_flag:
                parts = line.split()
                _horiz = round(float(parts[6]) * 1000, 4)

            elif line[:9] == "END_NLLOC":
                if save_flag:
                    hypocenters.append(
                        HypocenterResult(
                            _x_km,
                            _y_km,
                            _z_m,
                            _horiz,
                            depth_uncertainty_m=_depth_unc,
                            rms=_rms,
                        )
                    )
                save_flag = False

    logger.info("Parsed %d hypocenters from %s", len(hypocenters), file_path)
    return hypocenters


def catalog_to_obs(
    catalog: Catalog,
    *,
    output_path: str | Path = "nll_picks.obs",
    default_uncertainty: float = 0.1,
) -> Path:
    """Write an ObsPy Catalog's picks to a NonLinLoc ``.obs`` phase file.

    Supports both P and S phases.  Each event in the catalog is written
    as a separate ``EQEVENT`` block.

    Parameters
    ----------
    catalog : obspy.Catalog
        Catalog containing events with picks.
    output_path : str or Path
        Path for the output NLL phase file.
    default_uncertainty : float
        Gaussian 1-sigma pick uncertainty in seconds, used when a pick
        has no ``time_errors`` set.

    Returns
    -------
    Path
        The written ``.obs`` file.
    """
    output_path = Path(output_path)
    n_events = 0

    with open(output_path, "w") as fh:
        for event in catalog:
            if not event.picks:
                continue

            if n_events == 0:
                fh.write("# EQEVENT:  Label: EQ001  Loc:  X 0.0  Y 0.0  Z 0.0  OT 0.00\n")
            else:
                fh.write("#\n# EQEVENT:\n")
            n_events += 1

            seen: set[str] = set()
            for pick in event.picks:
                wid = pick.waveform_id
                station = wid.station_code or "UNK"
                phase = pick.phase_hint or "P"
                if phase not in {"P", "S"}:
                    continue

                key = f"{station}.{phase}"
                if key in seen:
                    continue
                seen.add(key)

                t = pick.time
                if pick.time_errors is not None and pick.time_errors.uncertainty is not None:
                    sigma = pick.time_errors.uncertainty
                else:
                    sigma = default_uncertainty

                fh.write(_format_nll_phase_line(station, phase, t, sigma))

    logger.info("Wrote %d events to %s", n_events, output_path)
    return output_path


def run_nonlinloc(
    control_file: str | Path,
    *,
    nll_binary: str = "NLLoc",
    timeout: float | None = 300.0,
) -> subprocess.CompletedProcess:
    """Run the NonLinLoc location program as a subprocess.

    Parameters
    ----------
    control_file : str or Path
        Path to the NLL control file (``*.nll_control`` or similar).
    nll_binary : str
        Name or path of the NonLinLoc executable. Must be on ``$PATH``
        or be an absolute path.
    timeout : float or None
        Maximum seconds to wait for the process. ``None`` means no limit.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process result with ``stdout`` and ``stderr``.

    Raises
    ------
    FileNotFoundError
        If *control_file* does not exist.
    subprocess.CalledProcessError
        If NonLinLoc exits with a non-zero status.
    subprocess.TimeoutExpired
        If *timeout* is exceeded.
    """
    control_file = Path(control_file)
    if not control_file.exists():
        msg = f"Control file not found: {control_file}"
        raise FileNotFoundError(msg)

    if shutil.which(nll_binary) is None and not Path(nll_binary).is_file():
        raise FileNotFoundError(
            f"NonLinLoc binary not found: {nll_binary!r}. "
            "Ensure it is installed and on $PATH, or provide an absolute path."
        )

    logger.info("Running NonLinLoc: %s %s", nll_binary, control_file)
    result = subprocess.run(
        [nll_binary, str(control_file)],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )
    logger.info("NonLinLoc completed successfully")
    return result


def hypocenter_to_obspy_origin(
    hyp: HypocenterResult,
    *,
    reference_lat: float,
    reference_lon: float,
) -> Origin:
    """Convert a :class:`HypocenterResult` to an ObsPy Origin with uncertainties.

    Parameters
    ----------
    hyp : HypocenterResult
        Parsed hypocenter from NonLinLoc.
    reference_lat : float
        Latitude of the NLL coordinate origin (degrees).
    reference_lon : float
        Longitude of the NLL coordinate origin (degrees).

    Returns
    -------
    obspy.core.event.Origin
        Origin with ``origin_uncertainty``, ``depth_errors``,
        ``latitude_errors``, and ``longitude_errors`` populated.
    """
    from obspy.core.event import (
        Origin,
        OriginQuality,
        OriginUncertainty,
        QuantityError,
    )

    origin = Origin(
        latitude=reference_lat,
        longitude=reference_lon,
        depth=hyp.z_m,
    )

    origin.depth_errors = QuantityError(uncertainty=hyp.depth_uncertainty_m)

    horiz_deg = hyp.horiz_uncertainty_m / _METERS_PER_DEG_LAT
    origin.latitude_errors = QuantityError(uncertainty=horiz_deg)
    origin.longitude_errors = QuantityError(uncertainty=horiz_deg)

    origin.origin_uncertainty = OriginUncertainty(
        horizontal_uncertainty=hyp.horiz_uncertainty_m,
    )
    origin.quality = OriginQuality(standard_error=hyp.rms)

    return origin


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_sec_fraction(second: int, sec_frac: float) -> str:
    """Format seconds and fractional seconds for NLL phase lines.

    Handles the edge case where rounding the fractional part to 4 decimal
    places causes an overflow (e.g. 0.999999 -> 1.0000), properly carrying
    over to the next second.

    Parameters
    ----------
    second : int
        Whole seconds value (0-59).
    sec_frac : float
        Fractional seconds in ``[0, 1)``.

    Returns
    -------
    str
        Formatted string like ``"SS.FFFF"``.
    """
    frac_rounded = round(sec_frac, 4)
    if frac_rounded >= 1.0:
        second += 1
        frac_rounded -= 1.0
    frac_digits = f"{frac_rounded:.4f}"[2:]  # "0.0500" -> "0500"
    return f"{second:02}.{frac_digits}"


def _format_nll_phase_line(
    station: str,
    phase: str,
    pick_time: UTCDateTime,
    sigma: float,
) -> str:
    """Format a single NLL phase observation line.

    Parameters
    ----------
    station : str
        Station code (max 6 chars).
    phase : str
        Phase label (``"P"`` or ``"S"``).
    pick_time : UTCDateTime
        Pick time.
    sigma : float
        Gaussian 1-sigma uncertainty in seconds.

    Returns
    -------
    str
        Formatted NLL phase line with trailing newline.
    """
    sec_frac = pick_time.microsecond / 1e6
    sec_str = _format_sec_fraction(pick_time.second, sec_frac)
    return (
        f"{station:<6} ?    ?    ? {phase:<6} ? "
        f"{pick_time.year}{pick_time.month:02}"
        f"{pick_time.day:02} {pick_time.hour:02}"
        f"{pick_time.minute:02} "
        f"{sec_str}"
        f" GAU {sigma:1.2e}  0.00e+00  0.00e+00  0.00e+00    0.00e+00\n"
    )


def _measure_amplitude(
    waveform_dir: str | Path,
    station: str,
    channel: str,
    start_time: UTCDateTime,
    end_time: UTCDateTime,
    pick_time: UTCDateTime,
    *,
    sampling_rate: float = 250.0,
    lowcut: float = 20.0,
    highcut: float = 60.0,
) -> tuple[float, float]:
    """Measure peak-to-peak amplitude around a pick."""
    waveform_dir = Path(waveform_dir)
    try:
        day_files = sorted(
            glob.glob(
                str(
                    waveform_dir / f"*.{station}..{channel}"
                    f".{pick_time.year}-{pick_time.month:02}"
                    f"-{pick_time.day:02}*.ms"
                )
            )
        )
        if not day_files:
            return np.nan, np.nan

        st = read(day_files[0])
        st.trim(start_time - 1, end_time + 2, pad=True, fill_value=0)
        st.interpolate(sampling_rate=sampling_rate)
        st.filter("bandpass", freqmin=lowcut, freqmax=highcut, corners=4)
        st.trim(start_time - 0.01, pick_time + 0.1, pad=True, fill_value=0)

        data = st[0].data
        max_idx = np.nanargmax(data)
        min_idx = np.nanargmin(data)
        ptp_amp = data[max_idx] - data[min_idx]
        ptp_period = abs(max_idx - min_idx) / sampling_rate
        ptp_amp = float(np.log(max(ptp_amp, 1e-30)))
        return ptp_amp, ptp_period
    except (OSError, ValueError, IndexError, TypeError) as exc:
        logger.debug("Amplitude measurement failed: %s: %s", type(exc).__name__, exc)
        return np.nan, np.nan
