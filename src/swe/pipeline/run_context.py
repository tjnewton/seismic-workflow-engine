"""Run context management for pipeline checkpointing.

Each pipeline run creates a directory under a configurable base path::

    <base_dir>/<run_id>/
      manifest.json
      provenance.json
      steps/
        <step_name>.output.json   (or .pkl)
        <step_name>.hash
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from swe.pipeline.serialization import OUTPUT_FORMAT

logger = logging.getLogger(__name__)

__all__ = ["RunContext"]


@dataclass
class RunContext:
    """Manages a single pipeline run directory."""

    run_id: str
    base_dir: Path
    run_dir: Path
    steps_dir: Path
    _manifest: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def create(cls, base_dir: Path) -> RunContext:
        """Create a new run directory with a unique ID.

        Parameters
        ----------
        base_dir : Path
            Base directory for all runs (e.g. ``.swe_runs``).

        Returns
        -------
        RunContext
        """
        run_id = _generate_run_id()
        run_dir = base_dir / run_id
        steps_dir = run_dir / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "output_format": OUTPUT_FORMAT,
            "steps": {},
        }

        ctx = cls(
            run_id=run_id,
            base_dir=base_dir,
            run_dir=run_dir,
            steps_dir=steps_dir,
            _manifest=manifest,
        )
        ctx._save_manifest()
        logger.info("Created run: %s", run_id)
        return ctx

    @classmethod
    def load(cls, base_dir: Path, run_id: str) -> RunContext:
        """Load an existing run context.

        Parameters
        ----------
        base_dir : Path
            Base directory for all runs.
        run_id : str
            Specific run ID or ``"latest"`` to pick the most recent.

        Returns
        -------
        RunContext
        """
        if run_id == "latest":
            run_id = _find_latest_run(base_dir)

        run_dir = base_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        ctx = cls(
            run_id=run_id,
            base_dir=base_dir,
            run_dir=run_dir,
            steps_dir=run_dir / "steps",
            _manifest=manifest,
        )
        logger.info("Loaded run: %s", run_id)
        return ctx

    @property
    def output_format(self) -> int:
        """Checkpoint format the run was written with (1: before the field existed)."""
        return int(self._manifest.get("output_format", 1))

    def is_step_completed(self, step_name: str) -> bool:
        """Check whether a step has been marked as completed."""
        step_info = self._manifest.get("steps", {}).get(step_name)
        return step_info is not None and step_info.get("status") == "completed"

    def get_step_output(self, step_name: str) -> Any:
        """Deserialize and return a completed step's output."""
        from swe.pipeline.serialization import deserialize_output

        return deserialize_output(self.steps_dir, step_name)

    def mark_step_completed(self, step_name: str, step_hash: str) -> None:
        """Record a step as successfully completed."""
        steps = self._manifest.setdefault("steps", {})
        steps[step_name] = {
            "status": "completed",
            "hash": step_hash,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_manifest()

    def mark_step_failed(self, step_name: str) -> None:
        """Record a step as failed."""
        steps = self._manifest.setdefault("steps", {})
        steps[step_name] = {
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._manifest["status"] = "failed"
        self._save_manifest()

    def _save_manifest(self) -> None:
        """Write the manifest to disk."""
        manifest_path = self.run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(self._manifest, indent=2))


def _generate_run_id() -> str:
    """Create a human-readable + unique run ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{ts}_{short_uuid}"


def _find_latest_run(base_dir: Path) -> str:
    """Find the most recent run by sorting directory names."""
    if not base_dir.exists():
        raise FileNotFoundError(f"No runs directory found: {base_dir}")

    runs = sorted(
        [d.name for d in base_dir.iterdir() if d.is_dir()],
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No runs found in {base_dir}")
    return runs[0]
