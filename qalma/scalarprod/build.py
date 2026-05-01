"""Functions to fetch specific scalar product functions."""

# from datetime import datetime
from typing import Callable

import numpy as np

from qalma.operators import Operator
from qalma.operators.states import DensityOperatorProtocol

from .covar import CovariantScalarProductFunction

#  ### Functions that build the scalar products ###


def kubo_scalar_product(sigma: Operator, threshold=0) -> Callable:
    """
    Fetch a KMB scalar product callable. Spectral implementation.

    Build a KMB scalar product function associated to the state
    ``sigma``.
    """
    evals_evecs = sorted(zip(*sigma.eigenstates()), key=lambda x: -x[0])
    w = 1
    for i, val_vec in enumerate(evals_evecs):
        p = val_vec[0]
        w -= p
        if w < threshold or p <= 0:
            evals_evecs = evals_evecs[: i + 1]
            break

    def ksp(op1, op2):
        result = sum(
            (
                np.conj((v2.dag() * op1 * v1).tr())
                * ((v2.dag() * op2 * v1).tr())
                * (p1 if p1 == p2 else (p1 - p2) / np.log(p1 / p2))
            )
            for p1, v1 in evals_evecs
            for p2, v2 in evals_evecs
            if (p1 > 0 and p2 > 0)
        )

        #    stored[key] = result
        return result

    return ksp


def kubo_integral_representation_scalar_product(sigma: Operator) -> Callable:
    """
    Fetch a KMB scalar product callable. Integral implementation.

    Build a KMB scalar product function associated to the state ``sigma``,
    from its integral form.
    """
    evals, evecs = sigma.eigenstates()

    def return_func(op1, op2):
        return 0.01 * sum(
            (
                np.conj((v2.dag() * op1 * v1).tr())
                * ((v2.dag() * op2 * v1).tr())
                * ((p1) ** (1.0 - tau))
                * ((p1) ** (tau))
            )
            for p1, v1 in zip(evals, evecs)
            for p2, v2 in zip(evals, evecs)
            for tau in np.linspace(0.0, 1.0, 100)
            if (p1 > 0.0 and p2 > 0.0)
        )

    return return_func


def covar_scalar_product(sigma: DensityOperatorProtocol) -> Callable:
    r"""
    Fetch a Covar scalar product callable.

    Returns a scalar product function based on the covariance of a density
    operator.

    The scalar product for two operators op1 and op2 is defined as:

    .. math::

        0.5 * Tr(sigma * {op1^\dagger, op2}),

    where sigma is a density operator, ${op1^\dagger, op2}$ is the
    anticommutator of the Hermitian conjugate of ``op1`` and ``op2``,
    and $Tr$ denotes the trace.

    Parameters
    ----------
        sigma: The density operator (quantum state) used to define the scalar
        product.

    Returns
    -------
        A function that takes two operators (op1, op2) and computes their
        covariance-based scalar product.

    """
    return CovariantScalarProductFunction(sigma)


def hs_scalar_product() -> Callable:
    """Fetch a HS scalar product function."""
    return lambda op1, op2: (op1.dag() * op2).tr()
