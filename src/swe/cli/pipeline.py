"""Backward-compatibility shim - all pipeline logic lives in :mod:`swe.pipeline`.

This module re-exports the public API so that existing code using
``from swe.cli.pipeline import Pipeline`` continues to work.
"""

from __future__ import annotations

from swe.pipeline.core import (
    Pipeline,
    PipelineStep,
    _import_callable,
    _resolve_params,
    _resolve_string,
)

__all__ = ["Pipeline", "PipelineStep"]
