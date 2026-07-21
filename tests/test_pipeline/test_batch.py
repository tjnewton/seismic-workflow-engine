"""Tests for swe.pipeline.batch - batch processing."""

from __future__ import annotations

import pytest

from swe.pipeline.batch import BatchResult, batch_process


def _double(x):
    """Simple test function that doubles a value."""
    return x * 2


def _fail_on_three(x):
    """Test function that raises on input 3."""
    if x == 3:
        raise ValueError("Cannot process 3")
    return x * 2


def _add_offset(x, *, offset=0):
    """Test function with kwargs."""
    return x + offset


class TestBatchResult:
    def test_basic_properties(self):
        result = BatchResult(results=[1, 2], errors=[], n_total=2)
        assert result.n_succeeded == 2
        assert result.n_failed == 0
        assert result.success_rate == 1.0

    def test_partial_failure(self):
        result = BatchResult(
            results=[1],
            errors=[(1, ValueError("test"))],
            n_total=2,
        )
        assert result.n_succeeded == 1
        assert result.n_failed == 1
        assert result.success_rate == 0.5

    def test_empty_batch(self):
        result = BatchResult(results=[], errors=[], n_total=0)
        assert result.success_rate == 0.0


class TestBatchProcess:
    def test_empty_items(self):
        result = batch_process(_double, [], progress=False)
        assert result.n_total == 0
        assert result.results == []

    def test_thread_executor(self):
        result = batch_process(
            _double,
            [1, 2, 3, 4, 5],
            executor_type="thread",
            progress=False,
            max_workers=2,
        )
        assert result.n_succeeded == 5
        assert sorted(result.results) == [2, 4, 6, 8, 10]

    def test_process_executor(self):
        result = batch_process(
            _double,
            [1, 2, 3],
            executor_type="process",
            progress=False,
            max_workers=2,
        )
        assert result.n_succeeded == 3
        assert sorted(result.results) == [2, 4, 6]

    def test_partial_failure(self):
        result = batch_process(
            _fail_on_three,
            [1, 2, 3, 4],
            executor_type="thread",
            progress=False,
            max_workers=2,
        )
        assert result.n_succeeded == 3
        assert result.n_failed == 1
        assert sorted(result.results) == [2, 4, 8]

    def test_func_kwargs(self):
        result = batch_process(
            _add_offset,
            [10, 20, 30],
            executor_type="thread",
            progress=False,
            func_kwargs={"offset": 5},
        )
        assert result.n_succeeded == 3
        assert sorted(result.results) == [15, 25, 35]

    def test_invalid_executor_type(self):
        with pytest.raises(ValueError, match="Unknown executor_type"):
            batch_process(_double, [1], executor_type="invalid", progress=False)

    def test_single_item(self):
        result = batch_process(
            _double,
            [42],
            executor_type="thread",
            progress=False,
        )
        assert result.n_succeeded == 1
        assert result.results == [84]
