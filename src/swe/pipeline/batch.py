"""Batch processing for multi-station/multi-window workflows.

Provides :func:`batch_process` which splits a workload across multiple
processes using :class:`~concurrent.futures.ProcessPoolExecutor` for
CPU-bound operations (spectral analysis, matched-filter detection) or
:class:`~concurrent.futures.ThreadPoolExecutor` for I/O-bound operations
(data download, file reading).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["BatchResult", "batch_process"]


class BatchResult:
    """Result container for batch processing.

    Parameters
    ----------
    results : list
        Successful results, one per completed work item.
    errors : list of tuple
        ``(item_index, exception)`` pairs for failed items.
    n_total : int
        Total number of work items submitted.
    """

    def __init__(
        self,
        results: list[Any],
        errors: list[tuple[int, Exception]],
        n_total: int,
    ) -> None:
        self.results = results
        self.errors = errors
        self.n_total = n_total

    @property
    def n_succeeded(self) -> int:
        """Number of successfully completed items."""
        return len(self.results)

    @property
    def n_failed(self) -> int:
        """Number of failed items."""
        return len(self.errors)

    @property
    def success_rate(self) -> float:
        """Fraction of items that succeeded (0.0 - 1.0)."""
        if self.n_total == 0:
            return 0.0
        return self.n_succeeded / self.n_total


def batch_process(
    func: Callable[..., Any],
    items: list[Any],
    *,
    max_workers: int | None = None,
    executor_type: str = "process",
    progress: bool = True,
    description: str = "Processing",
    func_kwargs: dict[str, Any] | None = None,
) -> BatchResult:
    """Process a list of items in parallel with progress tracking.

    Parameters
    ----------
    func : callable
        Function to apply to each item.  Called as
        ``func(item, **func_kwargs)``.
    items : list
        Work items to process.
    max_workers : int, optional
        Maximum number of parallel workers.  Defaults to the number of
        CPUs for ``"process"`` or 4x CPUs for ``"thread"``.
    executor_type : str
        ``"process"`` for CPU-bound work (uses ``ProcessPoolExecutor``)
        or ``"thread"`` for I/O-bound work (uses ``ThreadPoolExecutor``).
    progress : bool
        Whether to display a tqdm progress bar.
    description : str
        Label for the progress bar.
    func_kwargs : dict, optional
        Additional keyword arguments passed to *func*.

    Returns
    -------
    BatchResult
        Container with results, errors, and counts.

    Raises
    ------
    ValueError
        If *executor_type* is not ``"process"`` or ``"thread"``.
    """
    if not items:
        return BatchResult(results=[], errors=[], n_total=0)

    func_kwargs = func_kwargs or {}

    executor_cls: type[ProcessPoolExecutor] | type[ThreadPoolExecutor]
    if executor_type == "process":
        executor_cls = ProcessPoolExecutor
    elif executor_type == "thread":
        executor_cls = ThreadPoolExecutor
    else:
        raise ValueError(
            f"Unknown executor_type: {executor_type!r}. Supported: 'process', 'thread'"
        )

    logger.info(
        "Batch processing %d items with %s executor (max_workers=%s)",
        len(items),
        executor_type,
        max_workers,
    )

    results: list[Any] = []
    errors: list[tuple[int, Exception]] = []

    with executor_cls(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_worker, func, item, func_kwargs): i for i, item in enumerate(items)
        }

        iterator = as_completed(future_to_idx)
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(
                    iterator,
                    total=len(items),
                    desc=description,
                    unit="item",
                )
            except ImportError:
                pass  # tqdm not installed; run without progress bar

        for future in iterator:
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                logger.warning("Batch item %d failed: %s: %s", idx, type(exc).__name__, exc)
                errors.append((idx, exc))

    logger.info(
        "Batch complete: %d/%d succeeded, %d failed",
        len(results),
        len(items),
        len(errors),
    )

    return BatchResult(results=results, errors=errors, n_total=len(items))


def _worker(
    func: Callable[..., Any],
    item: Any,
    kwargs: dict[str, Any],
) -> Any:
    """Worker function for parallel execution."""
    return func(item, **kwargs)
