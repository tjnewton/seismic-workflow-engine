"""CLI subcommands for individual SWE workflow stages."""

from __future__ import annotations

import logging
from pathlib import Path

import click
from obspy import UTCDateTime

logger = logging.getLogger(__name__)


@click.command()
@click.option("--network", "-n", required=True, help="FDSN network code.")
@click.option("--station", "-s", required=True, multiple=True, help="Station code(s).")
@click.option("--channel", "-c", required=True, multiple=True, help="Channel code(s).")
@click.option("--start", required=True, help="Start time (ISO-8601).")
@click.option("--end", required=True, help="End time (ISO-8601).")
@click.option("--output-dir", "-o", default=".", help="Output directory.")
@click.option("--client", default="IRIS", help="FDSN data centre.")
def download(
    network: str,
    station: tuple[str, ...],
    channel: tuple[str, ...],
    start: str,
    end: str,
    output_dir: str,
    client: str,
) -> None:
    """Download waveforms and station metadata from FDSN."""
    from swe.io.fdsn import get_waveforms

    paths = get_waveforms(
        network,
        list(station),
        list(channel),
        UTCDateTime(start),
        UTCDateTime(end),
        output_dir=output_dir,
        client=client,
    )
    click.echo(f"Downloaded {len(paths)} file(s)")


@click.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Directory with MiniSEED files.",
)
@click.option("--start", required=True, help="Start time (ISO-8601).")
@click.option("--end", required=True, help="End time (ISO-8601).")
@click.option("--sta", default=15, type=int, help="STA window length (samples).")
@click.option("--lta", default=250, type=int, help="LTA window length (samples).")
@click.option("--threshold", default=3.0, type=float, help="STA/LTA threshold.")
@click.option("--window-length", default=240, type=int, help="Noise window length (samples).")
@click.option("--bandpass/--no-bandpass", default=False, help="Apply bandpass filter.")
@click.option("--freqmin", default=20.0, type=float, help="Bandpass low corner (Hz).")
@click.option("--freqmax", default=60.0, type=float, help="Bandpass high corner (Hz).")
@click.option(
    "--remove-response/--no-remove-response", default=False, help="Remove instrument response."
)
@click.option(
    "--response-dir", default=None, type=click.Path(exists=True), help="RESP file directory."
)
@click.option("--output", "-o", default="noise_windows.npz", help="Output .npz file.")
def noise(
    input_dir: str,
    start: str,
    end: str,
    sta: int,
    lta: int,
    threshold: float,
    window_length: int,
    bandpass: bool,
    freqmin: float,
    freqmax: float,
    remove_response: bool,
    response_dir: str | None,
    output: str,
) -> None:
    """Classify ambient noise in continuous waveforms."""
    import json

    import numpy as np

    from swe.noise import find_noise_windows

    input_path = Path(input_dir)
    filepaths = sorted(list(input_path.glob("*.ms")) + list(input_path.glob("*.mseed")))
    if not filepaths:
        raise click.ClickException(f"No MiniSEED files found in {input_dir}")

    detection_parameters = (sta, lta, threshold, window_length)
    time_periods = [(UTCDateTime(start), UTCDateTime(end))]

    try:
        result = find_noise_windows(
            filepaths,
            detection_parameters,
            time_periods,
            bandpass_filter=bandpass,
            freqmin=freqmin,
            freqmax=freqmax,
            remove_response=remove_response,
            response_dir=response_dir,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    out_path = Path(output)
    np.savez(out_path, windows=result.windows)

    meta_path = out_path.with_suffix(".json")
    meta = [
        {
            "start": str(m.start),
            "end": str(m.end),
            "station": m.station,
            "channel": m.channel,
        }
        for m in result.times
    ]
    meta_path.write_text(json.dumps(meta, indent=2))

    click.echo(f"Found {len(result.times)} noise window(s). Saved to {out_path} and {meta_path}")


@click.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Directory with MiniSEED files.",
)
@click.option(
    "--method",
    "-m",
    default="seisbench",
    type=click.Choice(["stalta", "seisbench"]),
    help="Picking method.",
)
@click.option(
    "--model",
    default="PhaseNet",
    type=click.Choice(["PhaseNet", "EQTransformer", "GPD", "BasicPhaseAE"]),
    help="SeisBench model (seisbench method only).",
)
@click.option("--pretrained", default="original", help="Pretrained weight set name.")
@click.option("--p-threshold", default=0.3, type=float, help="P-phase probability threshold.")
@click.option("--s-threshold", default=0.3, type=float, help="S-phase probability threshold.")
@click.option("--batch-size", default=256, type=int, help="Inference batch size.")
@click.option("--start", default=None, help="Trim start time (ISO-8601).")
@click.option("--end", default=None, help="Trim end time (ISO-8601).")
@click.option("--output", "-o", default="picks.csv", help="Output picks CSV file.")
def pick(
    input_dir: str,
    method: str,
    model: str,
    pretrained: str,
    p_threshold: float,
    s_threshold: float,
    batch_size: int,
    start: str | None,
    end: str | None,
    output: str,
) -> None:
    """Run automated phase picking on waveforms."""
    from obspy import read

    input_path = Path(input_dir)
    files = sorted(list(input_path.glob("*.ms")) + list(input_path.glob("*.mseed")))
    if not files:
        raise click.ClickException(f"No MiniSEED files found in {input_dir}")

    from obspy import Stream

    st = Stream()
    for f in files:
        st += read(str(f))

    if start is not None or end is not None:
        t1 = UTCDateTime(start) if start else None
        t2 = UTCDateTime(end) if end else None
        st.trim(starttime=t1, endtime=t2)

    if method == "seisbench":
        from swe.picking import predict_phase_arrivals

        try:
            arrivals = predict_phase_arrivals(
                st,
                model_name=model,
                pretrained=pretrained,
                p_threshold=p_threshold,
                s_threshold=s_threshold,
                batch_size=batch_size,
            )
        except ImportError as exc:
            raise click.ClickException(str(exc)) from exc

        out_path = Path(output)
        with open(out_path, "w") as fh:
            fh.write("station,phase,time,confidence\n")
            for a in arrivals:
                fh.write(f"{a.station},{a.phase},{a.time},{a.confidence:.6f}\n")

        click.echo(f"Wrote {len(arrivals)} pick(s) to {out_path}")

    else:
        # STA/LTA method: window extraction only (operates on filepaths)
        from swe.picking import find_picking_windows

        t1 = UTCDateTime(start) if start else st[0].stats.starttime
        t2 = UTCDateTime(end) if end else st[0].stats.endtime
        try:
            result = find_picking_windows(
                [str(f) for f in files],
                window_length=240,
                window_offset=120,
                time_periods=[(t1, t2)],
            )
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo(
            f"STA/LTA extracted {len(result.windows)} picking window(s). "
            "Note: STA/LTA does not produce scored phase picks; "
            "use --method seisbench for phase arrivals with confidence scores."
        )


@click.command()
@click.option(
    "--picks",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Input picks CSV (from 'swe pick').",
)
@click.option(
    "--stations",
    "-s",
    required=True,
    type=click.Path(exists=True),
    help="Stations CSV (station,lon,lat,elev).",
)
@click.option("--eps", default=0.5, type=float, help="DBSCAN eps parameter.")
@click.option("--min-samples", default=5, type=int, help="DBSCAN min_samples.")
@click.option("--min-picks", default=4, type=int, help="Minimum picks per cluster.")
@click.option("--temporal-threshold", default=0.5, type=float, help="Temporal threshold (s).")
@click.option("--spatial-threshold", default=5.0, type=float, help="Spatial threshold.")
@click.option("--output", "-o", default="catalog.xml", help="Output file.")
@click.option(
    "--format",
    "output_format",
    default="quakeml",
    type=click.Choice(["quakeml", "markers"]),
    help="Output format.",
)
def associate(
    picks: str,
    stations: str,
    eps: float,
    min_samples: int,
    min_picks: int,
    temporal_threshold: float,
    spatial_threshold: float,
    output: str,
    output_format: str,
) -> None:
    """Associate phase picks into events."""
    import csv

    from swe.association import associate_picks, association_to_catalog
    from swe.association.clustering import save_associations

    # Parse stations CSV: station,lon,lat,elev
    station_list: list[str] = []
    location_dict: dict[str, list[float]] = {}
    with open(stations) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sta = row["station"]
            station_list.append(sta)
            location_dict[sta] = [
                float(row["lon"]),
                float(row["lat"]),
                float(row["elev"]),
            ]

    if not station_list:
        raise click.ClickException("No stations found in stations CSV")

    # Parse picks CSV: station,phase,time,confidence
    station_picks: dict[str, list] = {s: [] for s in station_list}
    n_picks = 0
    with open(picks) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sta = row["station"]
            if sta not in station_picks:
                station_picks[sta] = []
            station_picks[sta].append(UTCDateTime(row["time"]))
            n_picks += 1

    if n_picks == 0:
        raise click.ClickException("No picks found in picks CSV")

    try:
        result = associate_picks(
            station_picks,
            station_list,
            location_dict,
            dbscan_eps=eps,
            dbscan_min_samples=min_samples,
            min_picks_per_cluster=min_picks,
            temporal_threshold=temporal_threshold,
            spatial_threshold=spatial_threshold,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    out_path = Path(output)
    n_events = len(set(result.labels) - {-1})

    if output_format == "quakeml":
        catalog = association_to_catalog(result)
        catalog.write(str(out_path), format="QUAKEML")
    else:
        save_associations(
            str(out_path),
            result.labels,
            result.pick_times,
            result.pick_station_order,
            result.reverse_station_distance,
        )

    click.echo(f"Associated {n_picks} pick(s) into {n_events} event(s). Saved to {out_path}")


@click.command()
@click.option(
    "--marker-file",
    required=True,
    type=click.Path(exists=True),
    help="Snuffler marker file (.mrkr).",
)
@click.option(
    "--template-dir",
    "-t",
    required=True,
    type=click.Path(exists=True),
    help="Directory with template MiniSEED files.",
)
@click.option(
    "--data-dir",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Directory with continuous MiniSEED files.",
)
@click.option("--start", required=True, help="Detection start date (ISO-8601).")
@click.option("--end", required=True, help="Detection end date (ISO-8601).")
@click.option("--prepick", default=2.0, type=float, help="Pre-pick offset (s).")
@click.option("--template-length", default=16.0, type=float, help="Template duration (s).")
@click.option("--threshold", default=8.0, type=float, help="Detection threshold.")
@click.option(
    "--thresh-type",
    default="MAD",
    type=click.Choice(["MAD", "abs", "av_chan_corr"]),
    help="Threshold type.",
)
@click.option("--trig-int", default=8.0, type=float, help="Minimum trigger interval (s).")
@click.option("--lowcut", default=1.0, type=float, help="Bandpass low corner (Hz).")
@click.option("--highcut", default=15.0, type=float, help="Bandpass high corner (Hz).")
@click.option("--samp-rate", default=100.0, type=float, help="Target sampling rate (Hz).")
@click.option("--output", "-o", default=None, help="Output file path for detection results.")
def detect(
    marker_file: str,
    template_dir: str,
    data_dir: str,
    start: str,
    end: str,
    prepick: float,
    template_length: float,
    threshold: float,
    thresh_type: str,
    trig_int: float,
    lowcut: float,
    highcut: float,
    samp_rate: float,
    output: str | None,
) -> None:
    """Run matched-filter detection."""
    from swe.detection import detect_signals, markers_to_template

    # Step 1: Build templates from markers
    click.echo("Building templates from marker file...")
    try:
        tmpl_result = markers_to_template(marker_file, prepick)
    except Exception as exc:
        raise click.ClickException(f"Failed to parse marker file: {exc}") from exc

    if not tmpl_result.templates:
        raise click.ClickException("No templates extracted from marker file")

    template_path = Path(template_dir)
    template_files = sorted(list(template_path.glob("*.ms")) + list(template_path.glob("*.mseed")))
    if not template_files:
        raise click.ClickException(f"No MiniSEED files found in {template_dir}")

    # Step 2: Run matched-filter detection
    click.echo("Running matched-filter detection...")
    try:
        party = detect_signals(
            tmpl_result.templates,
            template_files,
            tmpl_result.station_dict,
            template_length,
            prepick,
            data_dir,
            UTCDateTime(start),
            UTCDateTime(end),
            thresh_type=thresh_type,
            detect_thresh=threshold,
            trig_int=trig_int,
            lowcut=lowcut,
            highcut=highcut,
            samp_rate=samp_rate,
            output_path=output,
        )
    except Exception as exc:
        raise click.ClickException(f"Detection failed: {exc}") from exc

    if party is None:
        click.echo("No detections found.")
    else:
        n_det = sum(len(fam) for fam in party.families)
        click.echo(f"Found {n_det} detection(s)")
        if output:
            click.echo(f"Saved to {output}")


@click.command()
@click.option("--obs-file", "-o", required=True, help="NLL phase observation file.")
@click.option("--control-file", "-c", required=True, help="NLL control file.")
@click.option("--binary", default="NLLoc", help="Path to NLLoc binary.")
def locate(obs_file: str, control_file: str, binary: str) -> None:
    """Locate events using NonLinLoc."""
    from swe.location.nonlinloc import run_nonlinloc

    result = run_nonlinloc(control_file, nll_binary=binary)
    click.echo(f"NonLinLoc completed (return code {result.returncode})")


@click.command()
@click.option(
    "--input",
    "-i",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input MiniSEED file or directory.",
)
@click.option(
    "--plot-type",
    "-t",
    default="waveform",
    type=click.Choice(["waveform", "spectrogram", "spectrum"]),
    help="Type of plot.",
)
@click.option(
    "--output", "-o", default=None, help="Output image path (shows interactively if omitted)."
)
@click.option("--figsize", default=None, help="Figure size as WxH in inches (e.g. '12x8').")
@click.option("--dpi", default=150, type=int, help="Output DPI.")
@click.option("--channel", default=None, help="Filter to specific channel code.")
@click.option(
    "--freq-max", default=None, type=float, help="Maximum frequency for spectrogram/spectrum."
)
def plot(
    input_file: str,
    plot_type: str,
    output: str | None,
    figsize: str | None,
    dpi: int,
    channel: str | None,
    freq_max: float | None,
) -> None:
    """Generate waveform, spectrogram, or spectrum plots."""
    import matplotlib

    if output is not None:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from obspy import read

    from swe.viz import plot_spectra, plot_spectrogram, plot_stream

    input_path = Path(input_file)
    if input_path.is_dir():
        files = sorted(list(input_path.glob("*.ms")) + list(input_path.glob("*.mseed")))
        if not files:
            raise click.ClickException(f"No MiniSEED files found in {input_file}")
        from obspy import Stream

        st = Stream()
        for f in files:
            st += read(str(f))
    else:
        st = read(str(input_path))

    if channel is not None:
        st = st.select(channel=channel)
        if len(st) == 0:
            raise click.ClickException(f"No traces matching channel={channel}")

    parsed_figsize = None
    if figsize is not None:
        try:
            w, h = figsize.split("x")
            parsed_figsize = (float(w), float(h))
        except ValueError as exc:
            raise click.ClickException(
                f"Invalid figsize format: {figsize!r}. Use WxH (e.g. '12x8')."
            ) from exc

    if plot_type == "waveform":
        fig = plot_stream(st, figsize=parsed_figsize)
    elif plot_type == "spectrogram":
        if len(st) == 0:
            raise click.ClickException("No traces to plot")
        fig = plot_spectrogram(
            st[0],
            figsize=parsed_figsize,
            freq_max=freq_max,
        )
    elif plot_type == "spectrum":
        if len(st) == 0:
            raise click.ClickException("No traces to plot")
        fig = plot_spectra(
            st[0],
            figsize=parsed_figsize,
            freq_max=freq_max,
        )

    if output is not None:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        click.echo(f"Saved {plot_type} plot to {output}")
    else:
        plt.show()
        click.echo(f"Displayed {plot_type} plot")
