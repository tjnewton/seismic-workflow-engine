"""Shared test fixtures for seismic-workflow-engine."""

import numpy as np
import pytest


@pytest.fixture
def synthetic_trace_data():
    """Generate synthetic seismic trace data for testing.

    Returns a 1D numpy array simulating 60 seconds of ambient noise
    at 100 Hz sampling rate with a few embedded spikes.
    """
    rng = np.random.default_rng(42)
    sampling_rate = 100
    duration = 60
    n_samples = sampling_rate * duration

    # Ambient noise
    data = rng.normal(0, 0.01, n_samples)

    # Add a few "event" spikes
    for t in [10, 30, 50]:
        idx = t * sampling_rate
        data[idx : idx + 50] += rng.normal(0, 0.1, 50)

    return data, sampling_rate
