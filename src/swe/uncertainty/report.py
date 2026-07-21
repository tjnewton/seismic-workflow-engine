"""Human-readable uncertainty summary for pipeline runs."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["summarize_pipeline_uncertainty"]


def summarize_pipeline_uncertainty(
    provenance_records: list[dict[str, Any]],
) -> str:
    """Generate a human-readable summary of uncertainty across pipeline steps.

    Parameters
    ----------
    provenance_records : list of dict
        Provenance records from a pipeline run, each containing an
        optional ``uncertainty_summary`` dict.

    Returns
    -------
    str
        Formatted multi-line summary.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Uncertainty Propagation Summary")
    lines.append("=" * 60)

    has_any = False
    for record in provenance_records:
        step_name = record.get("step_name", "?")
        unc = record.get("uncertainty_summary")
        if not unc:
            lines.append(f"\n  {step_name}: no uncertainty data")
            continue

        has_any = True
        lines.append(f"\n  {step_name}:")

        for key, val in sorted(unc.items()):
            if isinstance(val, dict):
                lines.append(f"    {key}:")
                for k2, v2 in sorted(val.items()):
                    lines.append(f"      {k2}: {_format_value(v2)}")
            else:
                lines.append(f"    {key}: {_format_value(val)}")

    if not has_any:
        lines.append("\n  No uncertainty data found in any step.")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def _format_value(val: Any) -> str:
    """Format a value for display."""
    if isinstance(val, float):
        if abs(val) < 0.001 and val != 0.0:
            return f"{val:.2e}"
        return f"{val:.4f}"
    if isinstance(val, list) and len(val) <= 6:
        formatted = [_format_value(v) for v in val]
        return f"[{', '.join(formatted)}]"
    return str(val)
