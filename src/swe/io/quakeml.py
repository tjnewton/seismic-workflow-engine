"""QuakeML read/write wrappers.

Thin wrappers around ObsPy's QuakeML I/O with SWE-specific defaults
and validation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from obspy import Catalog, read_events
from obspy.core.event import CreationInfo

logger = logging.getLogger(__name__)


def write_quakeml(
    catalog: Catalog,
    output_path: str | Path,
    *,
    agency_id: str = "SWE",
) -> Path:
    """Write an ObsPy Catalog to QuakeML format.

    Parameters
    ----------
    catalog : obspy.Catalog
        Catalog of seismic events.
    output_path : str or Path
        Destination file path.
    agency_id : str
        Agency identifier written into ``creation_info`` for events
        that lack one.

    Returns
    -------
    Path
        The written file path.
    """
    output_path = Path(output_path)

    for event in catalog:
        if event.creation_info is None:
            event.creation_info = CreationInfo(agency_id=agency_id)
        elif not event.creation_info.agency_id:
            event.creation_info.agency_id = agency_id

    catalog.write(str(output_path), format="QUAKEML")
    logger.info("Wrote %d events to %s", len(catalog), output_path)
    return output_path


def read_quakeml(path: str | Path) -> Catalog:
    """Read a QuakeML file into an ObsPy Catalog.

    Parameters
    ----------
    path : str or Path
        Path to a QuakeML file.

    Returns
    -------
    obspy.Catalog
        Catalog of seismic events.
    """
    path = Path(path)
    catalog = read_events(str(path), format="QUAKEML")
    logger.info("Read %d events from %s", len(catalog), path)
    return catalog
