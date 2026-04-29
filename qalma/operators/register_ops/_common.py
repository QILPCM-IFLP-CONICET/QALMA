"""Shared constants for operator arithmetic registration."""

from numbers import Number

import numpy as np

NUMERIC_TYPES = tuple((Number, int, float, complex, np.float64, np.complex128))
