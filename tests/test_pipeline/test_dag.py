"""Tests for swe.pipeline.dag."""

from __future__ import annotations

import pytest

from swe.pipeline.core import PipelineStep
from swe.pipeline.dag import (
    _extract_references,
    build_dependency_graph,
    topological_tiers,
)


class TestBuildDependencyGraph:
    """Tests for build_dependency_graph."""

    def test_no_dependencies(self):
        """Steps with no references should have empty dep sets."""
        steps = [
            PipelineStep(name="a", function="mod.fn", params={"x": 1}),
            PipelineStep(name="b", function="mod.fn", params={"y": 2}),
        ]
        graph = build_dependency_graph(steps)
        assert graph == {"a": set(), "b": set()}

    def test_simple_dependency(self):
        """A reference ${a.val} should create a -> b dependency."""
        steps = [
            PipelineStep(name="a", function="mod.fn", params={}),
            PipelineStep(
                name="b",
                function="mod.fn",
                params={"input": "${a.val}"},
            ),
        ]
        graph = build_dependency_graph(steps)
        assert graph["a"] == set()
        assert graph["b"] == {"a"}

    def test_chained_dependencies(self):
        """a -> b -> c chain."""
        steps = [
            PipelineStep(name="a", function="mod.fn", params={}),
            PipelineStep(
                name="b",
                function="mod.fn",
                params={"x": "${a.out}"},
            ),
            PipelineStep(
                name="c",
                function="mod.fn",
                params={"y": "${b.out}"},
            ),
        ]
        graph = build_dependency_graph(steps)
        assert graph["a"] == set()
        assert graph["b"] == {"a"}
        assert graph["c"] == {"b"}

    def test_multiple_dependencies(self):
        """A step referencing two other steps."""
        steps = [
            PipelineStep(name="a", function="mod.fn", params={}),
            PipelineStep(name="b", function="mod.fn", params={}),
            PipelineStep(
                name="c",
                function="mod.fn",
                params={"x": "${a.out}", "y": "${b.out}"},
            ),
        ]
        graph = build_dependency_graph(steps)
        assert graph["c"] == {"a", "b"}

    def test_ignores_unknown_references(self):
        """References to steps not in the list should be ignored."""
        steps = [
            PipelineStep(
                name="a",
                function="mod.fn",
                params={"x": "${unknown.val}"},
            ),
        ]
        graph = build_dependency_graph(steps)
        assert graph["a"] == set()


class TestTopologicalTiers:
    """Tests for topological_tiers."""

    def test_independent_steps_single_tier(self):
        """All independent steps should be in one tier."""
        steps = [
            PipelineStep(name="a", function="mod.fn", params={}),
            PipelineStep(name="b", function="mod.fn", params={}),
            PipelineStep(name="c", function="mod.fn", params={}),
        ]
        graph = build_dependency_graph(steps)
        tiers = topological_tiers(steps, graph)
        assert len(tiers) == 1
        assert len(tiers[0]) == 3

    def test_linear_chain(self):
        """a -> b -> c should produce 3 tiers."""
        steps = [
            PipelineStep(name="a", function="mod.fn", params={}),
            PipelineStep(name="b", function="mod.fn", params={"x": "${a.out}"}),
            PipelineStep(name="c", function="mod.fn", params={"x": "${b.out}"}),
        ]
        graph = build_dependency_graph(steps)
        tiers = topological_tiers(steps, graph)
        assert len(tiers) == 3
        assert tiers[0][0].name == "a"
        assert tiers[1][0].name == "b"
        assert tiers[2][0].name == "c"

    def test_diamond_graph(self):
        """a -> (b, c) -> d produces 3 tiers; b and c are parallel."""
        steps = [
            PipelineStep(name="a", function="mod.fn", params={}),
            PipelineStep(name="b", function="mod.fn", params={"x": "${a.out}"}),
            PipelineStep(name="c", function="mod.fn", params={"x": "${a.out}"}),
            PipelineStep(
                name="d",
                function="mod.fn",
                params={"x": "${b.out}", "y": "${c.out}"},
            ),
        ]
        graph = build_dependency_graph(steps)
        tiers = topological_tiers(steps, graph)
        assert len(tiers) == 3
        assert tiers[0][0].name == "a"
        assert {s.name for s in tiers[1]} == {"b", "c"}
        assert tiers[2][0].name == "d"

    def test_cycle_raises(self):
        """Circular dependencies should raise ValueError."""
        steps = [
            PipelineStep(name="a", function="mod.fn", params={"x": "${b.out}"}),
            PipelineStep(name="b", function="mod.fn", params={"x": "${a.out}"}),
        ]
        graph = build_dependency_graph(steps)
        with pytest.raises(ValueError, match="cycle"):
            topological_tiers(steps, graph)


class TestExtractReferences:
    """Tests for _extract_references."""

    def test_string_ref(self):
        assert _extract_references({"x": "${step1.val}"}) == {"step1"}

    def test_list_ref(self):
        assert _extract_references({"x": ["${a.val}", "${b.val}"]}) == {"a", "b"}

    def test_nested_dict_ref(self):
        assert _extract_references({"x": {"y": "${deep.val}"}}) == {"deep"}

    def test_no_refs(self):
        assert _extract_references({"x": "plain", "y": 42}) == set()
