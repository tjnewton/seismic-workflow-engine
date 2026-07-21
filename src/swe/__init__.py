"""Seismic Workflow Engine (SWE).

A unified seismic analysis pipeline: noise detection, phase picking,
matched-filter detection, hypocenter location, focal mechanism
determination, and stress inversion.
"""

from __future__ import annotations

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.3.0"

__all__ = ["__version__"]
