"""Tests for swe.utils.helpers.is_integer."""

import numpy as np

from swe.utils import is_integer


def test_is_integer():
    assert is_integer(3) and is_integer(np.int64(3))
    assert not is_integer(True) and not is_integer(np.bool_(True))
    assert not is_integer(3.0) and not is_integer("3")
