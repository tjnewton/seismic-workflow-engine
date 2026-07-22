"""Deep-learning phase pickers via SeisBench.

Provides access to pre-trained models (PhaseNet, EQTransformer, etc.) for
automated P- and S-wave picking on ObsPy streams.

This module replaces an earlier custom U-Net picker with SeisBench's unified
model interface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from swe.picks import SWEPick, from_phase_arrivals

if TYPE_CHECKING:
    from obspy import Stream, UTCDateTime

logger = logging.getLogger(__name__)


class PhaseArrival(NamedTuple):
    """A single phase arrival pick from a deep-learning model."""

    station: str
    """Station code."""
    phase: str
    """Phase type (``"P"`` or ``"S"``)."""
    time: UTCDateTime
    """Pick time."""
    confidence: float
    """Model probability for the pick."""


_SUPPORTED_MODELS = frozenset(
    {
        "PhaseNet",
        "EQTransformer",
        "GPD",
        "BasicPhaseAE",
    }
)


def predict_phase_arrivals(
    stream: Stream,
    *,
    model_name: str = "PhaseNet",
    pretrained: str = "original",
    p_threshold: float = 0.3,
    s_threshold: float = 0.3,
    batch_size: int = 256,
) -> list[PhaseArrival]:
    """Predict P- and S-wave arrivals using a SeisBench deep-learning model.

    Parameters
    ----------
    stream : obspy.Stream
        Waveform data to pick.  Should contain three-component traces for
        best results, though single-component input is accepted.
    model_name : str
        Name of the SeisBench model class (e.g. ``"PhaseNet"``,
        ``"EQTransformer"``).
    pretrained : str
        Name of the pretrained weight set (e.g. ``"original"``,
        ``"stead"``, ``"instance"``).
    p_threshold : float
        Minimum probability for a P-phase pick to be retained.
    s_threshold : float
        Minimum probability for an S-phase pick to be retained.
    batch_size : int
        Batch size for model inference.

    Returns
    -------
    list of PhaseArrival
        Each :class:`PhaseArrival` has fields:

        - ``station`` (str): station code (e.g. ``"STA1"``)
        - ``phase`` (str): ``"P"`` or ``"S"``
        - ``time`` (:class:`~obspy.UTCDateTime`): pick time
        - ``confidence`` (float): model probability for the pick

    Raises
    ------
    ImportError
        If SeisBench is not installed.
    ValueError
        If *model_name* is not a supported SeisBench model.
    """
    try:
        import seisbench.models as sbm
    except ImportError as exc:
        raise ImportError(
            "seisbench is required for deep-learning picking. "
            "Install it with: pip install seismic-workflow-engine[picking]"
        ) from exc

    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name!r}. Supported models: {sorted(_SUPPORTED_MODELS)}"
        )

    model_cls = getattr(sbm, model_name)
    model = model_cls.from_pretrained(pretrained)

    logger.info(
        "Running %s (pretrained=%r) on %d traces",
        model_name,
        pretrained,
        len(stream),
    )

    model.annotate(stream, batch_size=batch_size)
    picks = model.classify(
        stream,
        batch_size=batch_size,
        P_threshold=p_threshold,
        S_threshold=s_threshold,
    )

    results: list[PhaseArrival] = []
    for pick in picks:
        results.append(
            PhaseArrival(
                station=pick.trace_id.split(".")[1] if "." in pick.trace_id else pick.trace_id,
                phase=pick.phase,
                time=pick.peak_time,
                confidence=pick.peak_value,
            )
        )

    logger.info("Found %d picks (%s)", len(results), model_name)
    return results


def pick_stream(
    stream: Stream,
    *,
    model_name: str = "PhaseNet",
    pretrained: str = "original",
    p_threshold: float = 0.3,
    s_threshold: float = 0.3,
    batch_size: int = 256,
) -> list[SWEPick]:
    """Pick phases and return a list of :class:`~swe.picks.SWEPick`.

    This is a convenience wrapper around :func:`predict_phase_arrivals` that
    converts the raw ``PhaseArrival`` results into unified :class:`SWEPick`
    objects.  To obtain the legacy ``{station: [UTCDateTime, ...]}`` format,
    pass the result to :func:`swe.picks.to_station_picks`.

    Parameters
    ----------
    stream : obspy.Stream
        Waveform data to pick.
    model_name : str
        SeisBench model name.
    pretrained : str
        Pretrained weight set name.
    p_threshold : float
        Minimum P-phase probability.
    s_threshold : float
        Minimum S-phase probability.
    batch_size : int
        Inference batch size.

    Returns
    -------
    list of SWEPick
        All picks across all stations, preserving phase and confidence.
    """
    raw_picks = predict_phase_arrivals(
        stream,
        model_name=model_name,
        pretrained=pretrained,
        p_threshold=p_threshold,
        s_threshold=s_threshold,
        batch_size=batch_size,
    )

    return from_phase_arrivals(raw_picks)
