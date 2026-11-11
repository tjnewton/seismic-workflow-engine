"""Pipeline validation for dry-run mode.

Checks that a pipeline YAML is structurally correct, all referenced
functions are importable, and step references resolve without cycles.
"""

from __future__ import annotations

import importlib
import inspect
import logging

from swe.pipeline.core import _REF_PATTERN, Pipeline

logger = logging.getLogger(__name__)

__all__ = ["validate_pipeline"]


def validate_pipeline(
    pipeline: Pipeline,
) -> tuple[list[str], list[str]]:
    """Validate a pipeline without executing it.

    Checks performed:

    1. Each step has a non-empty name and function path.
    2. No duplicate step names.
    3. All ``function`` paths are importable, and each step's ``params`` bind to the
       function's signature (no unknown names, no missing required ones), as the
       ``function(**params)`` call at run time needs.
    4. All ``${step.key}`` references point to earlier steps.
    5. No dependency cycles exist.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to validate.

    Returns
    -------
    errors : list of str
        Fatal problems that would prevent execution.
    warnings : list of str
        Non-fatal issues worth noting.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not pipeline.steps:
        errors.append("Pipeline has no steps.")
        return errors, warnings

    # --- Check 1 & 2: names and duplicates ---
    seen_names: set[str] = set()
    for i, step in enumerate(pipeline.steps):
        if not step.name:
            errors.append(f"Step {i} has an empty name.")
        elif step.name in seen_names:
            errors.append(f"Duplicate step name: {step.name!r}")
        seen_names.add(step.name)

        if not step.function:
            errors.append(f"Step {step.name!r} has an empty function path.")

    # --- Check 3: function imports ---
    for step in pipeline.steps:
        if not step.function:
            continue
        module_path, _, func_name = step.function.rpartition(".")
        if not module_path:
            errors.append(
                f"Step {step.name!r}: invalid function path {step.function!r} (no module)."
            )
            continue
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            errors.append(f"Step {step.name!r}: cannot import module {module_path!r}: {exc}")
            continue

        if not hasattr(module, func_name):
            errors.append(
                f"Step {step.name!r}: module {module_path!r} has no attribute {func_name!r}."
            )
            continue

        obj = getattr(module, func_name)
        if not callable(obj):
            warnings.append(
                f"Step {step.name!r}: {step.function!r} is not callable "
                f"(type: {type(obj).__name__})."
            )
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):  # some builtins expose no signature
            continue
        try:
            signature.bind(**dict.fromkeys(step.params))
        except TypeError as exc:
            errors.append(f"Step {step.name!r}: params do not fit {step.function}: {exc}.")

    # --- Check 4: reference resolution ---
    step_names_ordered: list[str] = []
    for step in pipeline.steps:
        refs = _extract_step_refs(step.params)
        for ref_name in refs:
            if ref_name not in seen_names:
                errors.append(f"Step {step.name!r}: references unknown step {ref_name!r}.")
            elif ref_name not in step_names_ordered:
                errors.append(
                    f"Step {step.name!r}: references step "
                    f"{ref_name!r} which has not been declared yet "
                    f"(forward reference)."
                )
        step_names_ordered.append(step.name)

    # --- Check 5: cycle detection ---
    try:
        from swe.pipeline.dag import build_dependency_graph, topological_tiers

        graph = build_dependency_graph(pipeline.steps)
        topological_tiers(pipeline.steps, graph)
    except ValueError as exc:
        errors.append(f"Dependency cycle detected: {exc}")

    # --- Summary output ---
    if not errors:
        _log_execution_plan(pipeline, warnings)

    return errors, warnings


def _extract_step_refs(params: dict) -> set[str]:
    """Extract step names from ``${step.key}`` references recursively."""
    refs: set[str] = set()
    for value in params.values():
        if isinstance(value, str):
            for m in _REF_PATTERN.finditer(value):
                refs.add(m.group(1))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    for m in _REF_PATTERN.finditer(item):
                        refs.add(m.group(1))
        elif isinstance(value, dict):
            refs.update(_extract_step_refs(value))
    return refs


def _log_execution_plan(
    pipeline: Pipeline,
    warnings: list[str],
) -> None:
    """Log a human-readable execution plan."""
    from swe.pipeline.dag import build_dependency_graph, topological_tiers

    graph = build_dependency_graph(pipeline.steps)
    tiers = topological_tiers(pipeline.steps, graph)

    logger.info("Execution plan (%d steps, %d tiers):", len(pipeline.steps), len(tiers))
    for tier_idx, tier in enumerate(tiers):
        names = ", ".join(s.name for s in tier)
        parallel_note = " (parallel)" if len(tier) > 1 else ""
        logger.info("  Tier %d%s: %s", tier_idx + 1, parallel_note, names)
