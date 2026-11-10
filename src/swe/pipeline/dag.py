"""DAG construction and parallel tier execution for pipeline steps.

Dependencies are inferred from ``${step.key}`` references in step parameters.
Steps with no interdependencies are grouped into tiers that can execute
concurrently.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from swe.pipeline.core import _REF_PATTERN, PipelineStep

logger = logging.getLogger(__name__)

__all__ = ["build_dependency_graph", "topological_tiers"]


def build_dependency_graph(
    steps: list[PipelineStep],
) -> dict[str, set[str]]:
    """Infer a dependency graph from ``${step.key}`` references.

    Parameters
    ----------
    steps : list of PipelineStep
        Pipeline steps (order does not matter).

    Returns
    -------
    dict
        Mapping of ``step_name -> set of step names it depends on``.
    """
    step_names = {s.name for s in steps}
    graph: dict[str, set[str]] = {s.name: set() for s in steps}

    for step in steps:
        refs = _extract_references(step.params)
        for ref_name in refs:
            if ref_name in step_names:
                graph[step.name].add(ref_name)

    return graph


def topological_tiers(
    steps: list[PipelineStep],
    graph: dict[str, set[str]],
) -> list[list[PipelineStep]]:
    """Group steps into tiers using Kahn's algorithm.

    Each tier contains steps whose dependencies are all in earlier tiers.
    Steps within a tier have no mutual dependencies and can run in parallel.

    Parameters
    ----------
    steps : list of PipelineStep
        All pipeline steps to schedule.
    graph : dict
        Dependency graph from :func:`build_dependency_graph`.

    Returns
    -------
    list of list of PipelineStep
        Tiers of steps, ordered from first to last.

    Raises
    ------
    ValueError
        If the graph contains a cycle.
    """
    step_map = {s.name: s for s in steps}
    in_degree: dict[str, int] = {name: 0 for name in graph}

    for name, deps in graph.items():
        in_degree[name] = len(deps)

    # Reverse adjacency: who depends on me?
    reverse: dict[str, set[str]] = defaultdict(set)
    for name, deps in graph.items():
        for dep in deps:
            reverse[dep].add(name)

    tiers: list[list[PipelineStep]] = []
    remaining = set(graph)

    while remaining:
        # Find all nodes with in_degree 0
        ready = [n for n in remaining if in_degree[n] == 0]
        if not ready:
            raise ValueError(f"Dependency cycle detected among steps: {remaining}")

        tier = [step_map[n] for n in sorted(ready)]
        tiers.append(tier)

        for n in ready:
            remaining.discard(n)
            for dependent in reverse.get(n, set()):
                in_degree[dependent] -= 1

    return tiers


def _extract_references(params: dict[str, Any]) -> set[str]:
    """Recursively extract step names referenced via ``${step}`` or ``${step.key}``."""
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
            refs.update(_extract_references(value))

    return refs
