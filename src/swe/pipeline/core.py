"""YAML pipeline orchestration for multi-step seismic workflows.

A pipeline configuration is a YAML file listing sequential *steps*.
Each step names a function to call and provides keyword arguments.
Parameters may reference outputs from prior steps using ``${step_name.key}``
syntax.
"""

from __future__ import annotations

import importlib
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from swe.pipeline.provenance import StepProvenance
    from swe.pipeline.run_context import RunContext

logger = logging.getLogger(__name__)

_REF_PATTERN = re.compile(r"\$\{(\w+)\.(\w+)\}")

__all__ = ["Pipeline", "PipelineStep"]


@dataclass
class PipelineStep:
    """A single step in a pipeline."""

    name: str
    """Unique identifier for this step."""
    function: str
    """Dotted import path of the callable (e.g. ``swe.io.fdsn.get_waveforms``)."""
    params: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to pass to the function."""
    cache: bool = True
    """Whether this step's output may be served from cache."""
    propagate_uncertainty: bool = False
    """Whether this step participates in uncertainty propagation."""


@dataclass
class Pipeline:
    """Ordered collection of :class:`PipelineStep` instances."""

    steps: list[PipelineStep] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Pipeline:
        """Load a pipeline from a YAML configuration file.

        Parameters
        ----------
        path : str or Path
            Path to a ``.yml`` / ``.yaml`` file.

        Returns
        -------
        Pipeline
            The loaded pipeline.
        """
        path = Path(path)
        with open(path) as fh:
            raw = yaml.safe_load(fh)

        if not isinstance(raw, dict) or "steps" not in raw:
            raise ValueError(f"Pipeline YAML must contain a 'steps' key: {path}")

        steps: list[PipelineStep] = []
        for step_dict in raw["steps"]:
            steps.append(
                PipelineStep(
                    name=step_dict["name"],
                    function=step_dict["function"],
                    params=step_dict.get("params", {}),
                    cache=step_dict.get("cache", True),
                    propagate_uncertainty=step_dict.get("propagate_uncertainty", False),
                )
            )

        logger.info("Loaded pipeline with %d steps from %s", len(steps), path)
        return cls(steps=steps)

    def execute(
        self,
        *,
        skip: set[str] | None = None,
        run_dir: str | Path | None = None,
        resume: str | None = None,
        no_cache: bool = False,
        parallel: bool = False,
    ) -> dict[str, Any]:
        """Run all steps, optionally with checkpointing, caching, and parallelism.

        Parameters
        ----------
        skip : set of str, optional
            Step names to skip.
        run_dir : str or Path, optional
            Base directory for run state (default ``.swe_runs``).
        resume : str, optional
            Run ID to resume, or ``"latest"`` for the most recent run.
        no_cache : bool
            If True, disable output caching entirely.
        parallel : bool
            If True, run independent steps concurrently.

        Returns
        -------
        dict
            Mapping of ``step_name -> return_value`` for every executed step.
        """
        skip = skip or set()
        outputs: dict[str, Any] = {}

        # If no persistence features requested, run in-memory (backward compat)
        if run_dir is None and resume is None and not parallel:
            return self._execute_simple(skip=skip, no_cache=no_cache)

        from swe.pipeline.provenance import save_provenance
        from swe.pipeline.run_context import RunContext

        provenance_records: list[StepProvenance] = []

        # Set up run context
        base_dir = Path(run_dir) if run_dir else Path(".swe_runs")
        if resume:
            run_ctx = RunContext.load(base_dir, resume)
            # Restore completed step outputs
            for step in self.steps:
                if run_ctx.is_step_completed(step.name):
                    outputs[step.name] = run_ctx.get_step_output(step.name)
                    logger.info("Restored step from checkpoint: %s", step.name)
        else:
            run_ctx = RunContext.create(base_dir)

        logger.info("Run ID: %s", run_ctx.run_id)

        # Determine execution order
        if parallel:
            self._execute_parallel(
                skip=skip,
                outputs=outputs,
                run_ctx=run_ctx,
                no_cache=no_cache,
                provenance_records=provenance_records,
            )
        else:
            self._execute_sequential(
                skip=skip,
                outputs=outputs,
                run_ctx=run_ctx,
                no_cache=no_cache,
                provenance_records=provenance_records,
            )

        # Save provenance
        if provenance_records:
            save_provenance(run_ctx.run_dir, provenance_records)

        return outputs

    def _execute_simple(
        self,
        *,
        skip: set[str],
        no_cache: bool = False,
    ) -> dict[str, Any]:
        """Run all steps sequentially in-memory, no persistence.

        This preserves the original execution behavior for callers
        that don't need checkpointing, caching, or parallelism.
        """
        outputs: dict[str, Any] = {}
        for step in self.steps:
            if step.name in skip:
                logger.info("Skipping step: %s", step.name)
                continue

            logger.info("Running step: %s (%s)", step.name, step.function)
            resolved = _resolve_params(step.params, outputs)
            func = _import_callable(step.function)
            result = func(**resolved)
            outputs[step.name] = result
            logger.info("Completed step: %s", step.name)

        return outputs

    def _execute_sequential(
        self,
        *,
        skip: set[str],
        outputs: dict[str, Any],
        run_ctx: RunContext,
        no_cache: bool,
        provenance_records: list[StepProvenance],
    ) -> None:
        """Run steps one at a time in declaration order."""

        for step in self.steps:
            if step.name in skip:
                logger.info("Skipping step: %s", step.name)
                continue
            if run_ctx.is_step_completed(step.name):
                logger.info("Step already completed (checkpoint): %s", step.name)
                continue

            self._run_step(
                step,
                outputs=outputs,
                run_ctx=run_ctx,
                no_cache=no_cache,
                provenance_records=provenance_records,
            )

    def _execute_parallel(
        self,
        *,
        skip: set[str],
        outputs: dict[str, Any],
        run_ctx: RunContext,
        no_cache: bool,
        provenance_records: list[StepProvenance],
    ) -> None:
        """Run independent steps concurrently using DAG tiers."""
        from swe.pipeline.dag import build_dependency_graph, topological_tiers

        remaining = [
            s for s in self.steps if s.name not in skip and not run_ctx.is_step_completed(s.name)
        ]

        graph = build_dependency_graph(remaining)
        tiers = topological_tiers(remaining, graph)

        lock = threading.Lock()

        for tier in tiers:
            if len(tier) == 1:
                self._run_step(
                    tier[0],
                    outputs=outputs,
                    run_ctx=run_ctx,
                    no_cache=no_cache,
                    provenance_records=provenance_records,
                    lock=lock,
                )
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor(max_workers=len(tier)) as pool:
                    futures = {
                        pool.submit(
                            self._run_step,
                            step,
                            outputs=outputs,
                            run_ctx=run_ctx,
                            no_cache=no_cache,
                            provenance_records=provenance_records,
                            lock=lock,
                        ): step
                        for step in tier
                    }
                    for future in as_completed(futures):
                        future.result()  # propagate exceptions

    def _run_step(
        self,
        step: PipelineStep,
        *,
        outputs: dict[str, Any],
        run_ctx: RunContext,
        no_cache: bool,
        provenance_records: list[StepProvenance],
        lock: threading.Lock | None = None,
    ) -> None:
        """Execute a single pipeline step with caching and provenance."""
        import time

        from swe.pipeline.cache import compute_step_hash, find_cached_result
        from swe.pipeline.provenance import StepProvenance
        from swe.pipeline.serialization import serialize_output

        resolved = _resolve_params(step.params, outputs)
        func = _import_callable(step.function)
        step_hash = compute_step_hash(step.function, resolved, func)

        cache_hit = False

        # Try cache
        if not no_cache and step.cache:
            cached = find_cached_result(step.name, step_hash, run_ctx.base_dir)
            if cached is not None:
                logger.info("Cache hit for step: %s (hash %s)", step.name, step_hash[:12])
                if lock is not None:
                    with lock:
                        outputs[step.name] = cached
                else:
                    outputs[step.name] = cached
                cache_hit = True

        started_at = time.time()

        if not cache_hit:
            logger.info("Running step: %s (%s)", step.name, step.function)
            try:
                result = func(**resolved)
            except Exception:
                run_ctx.mark_step_failed(step.name)
                logger.error(
                    "Pipeline failed at step '%s'. Resume with: "
                    "swe run <pipeline.yml> --resume latest",
                    step.name,
                )
                raise
            if lock is not None:
                with lock:
                    outputs[step.name] = result
            else:
                outputs[step.name] = result
            serialize_output(run_ctx.steps_dir, step.name, result)
            logger.info("Completed step: %s", step.name)

        completed_at = time.time()

        run_ctx.mark_step_completed(step.name, step_hash)

        prov = StepProvenance(
            step_name=step.name,
            function_path=step.function,
            resolved_params=_params_summary(resolved),
            input_hash=step_hash,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=completed_at - started_at,
            return_type=type(outputs[step.name]).__name__,
            cache_hit=cache_hit,
            uncertainty_summary=_extract_uncertainty(outputs[step.name]),
        )
        if lock is not None:
            with lock:
                provenance_records.append(prov)
        else:
            provenance_records.append(prov)


def _resolve_params(
    params: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    """Replace ``${step.key}`` references with values from prior outputs."""
    resolved: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str):
            resolved[key] = _resolve_string(value, outputs)
        elif isinstance(value, list):
            resolved[key] = [
                _resolve_string(v, outputs) if isinstance(v, str) else v for v in value
            ]
        elif isinstance(value, dict):
            resolved[key] = _resolve_params(value, outputs)
        else:
            resolved[key] = value
    return resolved


def _resolve_string(value: str, outputs: dict[str, Any]) -> Any:
    """Resolve a single string that may contain ``${step.key}`` references."""

    def _lookup(step_name: str, attr: str) -> Any:
        if step_name not in outputs:
            ref = f"${{{step_name}.{attr}}}"
            raise KeyError(
                f"Parameter reference '{ref}' failed: "
                f"step '{step_name}' not found in outputs "
                f"(available: {list(outputs)})"
            )
        step_output = outputs[step_name]
        if isinstance(step_output, dict):
            if attr not in step_output:
                ref = f"${{{step_name}.{attr}}}"
                raise KeyError(
                    f"Parameter reference '{ref}' failed: "
                    f"key '{attr}' not found in output of step '{step_name}' "
                    f"(available keys: {list(step_output)})"
                )
            return step_output[attr]
        if not hasattr(step_output, attr):
            ref = f"${{{step_name}.{attr}}}"
            raise AttributeError(
                f"Parameter reference '{ref}' failed: "
                f"output of step '{step_name}' (type {type(step_output).__name__}) "
                f"has no attribute '{attr}'"
            )
        return getattr(step_output, attr)

    match = _REF_PATTERN.fullmatch(value)
    if match:
        step_name, attr = match.groups()
        return _lookup(step_name, attr)

    def _replacer(m: re.Match) -> str:
        step_name, attr = m.groups()
        return str(_lookup(step_name, attr))

    return _REF_PATTERN.sub(_replacer, value)


def _import_callable(dotted_path: str) -> Any:
    """Import a callable from a dotted module path."""
    module_path, _, func_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid function path: {dotted_path!r}")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


_UNCERTAINTY_FIELDS = frozenset(
    {
        "ml_std",
        "ml_stderr",
        "md_std",
        "md_stderr",
        "depth_uncertainty_m",
        "rms",
        "strike_std",
        "dip_std",
        "rake_std",
        "iso_pct_range",
        "dc_pct_range",
        "clvd_pct_range",
        "bootstrap_axes",
        "cluster_quality",
        "uncertainty_summary",
        "corner_frequency_std",
        "long_period_level_std",
        "residual_rms",
        "qc_std",
        "fit_r_squared",
        # uncertainty propagation fields
        "time_uncertainty_s",
        "covariance_matrix",
        "distance_contribution",
        "horiz_uncertainty_m",
        "mag_uncertainty",
        "magnitude_uncertainty",
        "uncertainty",
    }
)


def _extract_uncertainty(result: Any) -> dict[str, Any] | None:
    """Extract recognised uncertainty fields from a step result."""
    if result is None:
        return None

    # Only inspect NamedTuples, dataclasses, and dicts - not arbitrary objects.
    if isinstance(result, dict):
        getter = lambda name: result.get(name)  # noqa: E731
    elif (isinstance(result, tuple) and hasattr(type(result), "_fields")) or hasattr(
        type(result), "__dataclass_fields__"
    ):
        getter = lambda name: getattr(result, name, None)  # noqa: E731
    else:
        return None

    found: dict[str, Any] = {}
    for attr_name in _UNCERTAINTY_FIELDS:
        val = getter(attr_name)
        if val is None:
            continue
        # Skip zero-valued scalars (default/unset)
        if isinstance(val, (int, float)) and val == 0:
            continue
        # Skip empty containers
        if isinstance(val, (dict, list)) and len(val) == 0:
            continue
        # Convert uncertainty dataclasses to dicts for JSON serialisation
        if hasattr(type(val), "__dataclass_fields__"):
            val = _dataclass_to_json_safe(val)
        # Convert tuples to lists for JSON serialisation
        elif isinstance(val, tuple):
            val = list(val)
        found[attr_name] = val

    return found if found else None


def _dataclass_to_json_safe(obj: Any) -> dict[str, Any]:
    """Convert a dataclass to a JSON-safe dict, handling numpy arrays."""
    from dataclasses import asdict

    import numpy as np

    d = asdict(obj)
    for key, val in d.items():
        if isinstance(val, np.ndarray):
            d[key] = val.tolist()
    return d


def _params_summary(params: dict[str, Any]) -> dict[str, Any]:
    """Create a JSON-safe summary of resolved params for provenance."""
    summary: dict[str, Any] = {}
    for key, value in params.items():
        try:
            import json

            json.dumps(value)
            summary[key] = value
        except (TypeError, ValueError):
            summary[key] = f"<{type(value).__name__}>"
    return summary
