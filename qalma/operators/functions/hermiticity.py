"""Functions for operators."""

from numbers import Complex, Real

# from collections.abc import Iterable
# from typing import Callable, List, Optional, Tuple
from typing import Tuple

from qalma.operators.basic import (
    Operator,
)

__all__ = ["compute_dagger", "hermitian_and_antihermitian_parts"]


def compute_dagger(operator):
    """Compute the adjoint of an `operator.

    If `operator` is a number, return its complex conjugate.
    """
    if isinstance(operator, Real):
        return operator
    if isinstance(operator, Complex):
        if operator.imag == 0:
            return operator.real
        return operator.conj()
    return operator.dag()


def hermitian_and_antihermitian_parts(operator: Operator) -> Tuple[Operator, Operator]:
    """Decompose an operator Q as A + i B with A and B self-adjoint operators."""
    return operator.hermitian_part(), (operator * (-1j)).hermitian_part()
