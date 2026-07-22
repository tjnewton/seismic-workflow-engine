"""Provenance tracking for pipeline step execution.

Records per-step metadata: function, params, timing, versions, cache status.
Stored as ``provenance.json`` in the run directory.
"""

from __future__ import annotations

import fcntl
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import swe

logger = logging.getLogger(__name__)

__all__ = ["StepProvenance", "load_provenance", "save_provenance"]


@dataclass
class StepProvenance:
    """Provenance record for a single pipeline step."""

    step_name: str
    function_path: str
    resolved_params: dict[str, Any]
    input_hash: str
    started_at: float
    completed_at: float
    duration_seconds: float
    return_type: str
    cache_hit: bool
    python_version: str = field(default_factory=lambda: sys.version)
    swe_version: str = field(default_factory=lambda: swe.__version__)
    uncertainty_summary: dict[str, Any] | None = field(default=None)


def save_provenance(
    run_dir: Path,
    records: list[StepProvenance],
) -> Path:
    """Write provenance records to ``provenance.json``.

    Parameters
    ----------
    run_dir : Path
        Root directory of the pipeline run.
    records : list of StepProvenance
        Provenance records to persist.

    Returns
    -------
    Path
        Path to the written file.
    """
    out_path = run_dir / "provenance.json"

    # Use file locking to prevent data loss from concurrent writes
    with open(out_path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            content = fh.read()
            existing: list[dict[str, Any]] = json.loads(content) if content.strip() else []
            existing.extend(asdict(r) for r in records)
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(existing, indent=2))
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

    logger.debug("Wrote %d provenance record(s) to %s", len(records), out_path)
    return out_path


def load_provenance(run_dir: Path) -> list[dict[str, Any]]:
    """Load provenance records from a run directory.

    Parameters
    ----------
    run_dir : Path
        Root directory of the pipeline run.

    Returns
    -------
    list of dict
        Provenance records as plain dicts.
    """
    prov_path = run_dir / "provenance.json"
    if not prov_path.exists():
        return []
    records: list[dict[str, Any]] = json.loads(prov_path.read_text())
    return records


def format_provenance(records: list[dict[str, Any]]) -> str:
    """Format provenance records for display.

    Parameters
    ----------
    records : list of dict
        Provenance records.

    Returns
    -------
    str
        Human-readable table.
    """
    if not records:
        return "No provenance records found."

    lines = [
        f"{'Step':<25} {'Duration':>10} {'Cache':>6} {'Return Type':<20} {'Function'}",
        "-" * 90,
    ]
    for r in records:
        dur = f"{r.get('duration_seconds', 0):.3f}s"
        cache = "HIT" if r.get("cache_hit") else "MISS"
        lines.append(
            f"{r['step_name']:<25} {dur:>10} {cache:>6} "
            f"{r.get('return_type', '?'):<20} {r['function_path']}"
        )
    return "\n".join(lines)
