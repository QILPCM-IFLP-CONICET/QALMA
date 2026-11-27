"""Scalar products between operators.

Functions that build scalar product functions ``sp(op1, op2)``: the
Hilbert-Schmidt scalar product, the covariance scalar product associated to a
state, and the Kubo-Mori-Bogoliubov (KMB) scalar product. All of them are
antilinear in the first argument and linear in the second one, and can be
used to build Gram matrices (:func:`~qalma.scalarprod.gram.gram_matrix`) and to
orthogonalize bases of operators
(:func:`~qalma.scalarprod.orthogonalize.orthogonalize_basis`).

"""

# from datetime import datetime
from typing import Callable

import numpy as np

from qalma.operators import Operator
from qalma.operators.states import DensityOperatorProtocol

from .covar import CovariantScalarProductFunction

#  ### Functions that build the scalar products ###

__all__ = [
    "fetch_covar_scalar_product",
    "fetch_hs_scalar_product",
    "fetch_kubo_integral_representation_scalar_product",
    "fetch_kubo_scalar_product",
]


# Gauss-Legendre nodes and weights for the integral over tau in [0, 1].
_TAU_NODES, _TAU_WEIGHTS = np.polynomial.legendre.leggauss(32)
_TAU_NODES = 0.5 * (_TAU_NODES + 1.0)
_TAU_WEIGHTS = 0.5 * _TAU_WEIGHTS


def _to_qutip(op):
    """Return the ``qutip.Qobj`` representation of ``op`` (an Operator or a Qobj)."""
    return op.to_qutip() if hasattr(op, "to_qutip") else op


def _log_mean(p1: float, p2: float) -> float:
    """Logarithmic mean ``(p1 - p2) / log(p1 / p2)`` of two positive numbers.

    Written with ``log1p`` so it stays accurate for almost degenerate values,
    where ``log(p1 / p2)`` loses all its significant digits.
    """
    if p1 == p2:
        return p1
    return (p1 - p2) / np.log1p((p1 - p2) / p2)


def fetch_kubo_scalar_product(sigma: Operator, threshold=0) -> Callable:
    r"""Fetch a KMB scalar product callable. Spectral implementation.

    Build a Kubo-Mori-Bogoliubov (KMB) scalar product function associated
    to the state ``sigma`` :cite:p:`Kubo1957,PetzToth1993`:

    .. math::

        \langle A, B\rangle_\sigma
        = \int_0^1 \mathrm{Tr}\left(\sigma^{1-\tau} A^\dagger \sigma^\tau B\right)
          d\tau
        = \sum_{i,j} \overline{\langle j|A|i\rangle}\,\langle j|B|i\rangle\,
          \frac{p_i-p_j}{\ln p_i-\ln p_j},

    where :math:`\sigma|i\rangle=p_i|i\rangle`, and the fraction is
    replaced by :math:`p_i` when :math:`p_i=p_j`. The sum runs over the
    eigenvalues :math:`p_i>0`, and it is evaluated from the spectral decomposition
    of ``sigma``. The integral form is implemented in
    :func:`~qalma.scalarprod.build.fetch_kubo_integral_representation_scalar_product`.

    Parameters
    ----------
    sigma: Operator
        The reference state: a positive semidefinite operator with unit trace.
        Its spectral decomposition is obtained from ``sigma.eigenstates()``, so
        the cost of building the function is that of a dense diagonalization.
    threshold: float, optional
        The eigenvalues of ``sigma`` are sorted from the largest to the smallest,
        and the sum is truncated as soon as the weight of the remaining ones is
        below ``threshold``. With the default, ``0``, only the eigenvalues that
        are zero (or negative by round-off) are dropped.

    Returns
    -------
    Callable
        A function ``f(op1, op2) -> complex`` that computes the scalar product of
        two operators. The operators can be qalma operators or ``qutip.Qobj``.

    See Also
    --------
    qalma.scalarprod.build.fetch_kubo_integral_representation_scalar_product
    qalma.scalarprod.build.fetch_covar_scalar_product
    qalma.scalarprod.build.fetch_hs_scalar_product

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
        """Spectral KMB scalar product of ``op1`` and ``op2``."""
        q1, q2 = _to_qutip(op1), _to_qutip(op2)
        result = sum(
            (np.conj(v2.dag() * q1 * v1) * (v2.dag() * q2 * v1) * _log_mean(p1, p2))
            for p1, v1 in evals_evecs
            for p2, v2 in evals_evecs
            if (p1 > 0 and p2 > 0)
        )

        #    stored[key] = result
        return result

    return ksp


def fetch_kubo_integral_representation_scalar_product(sigma: Operator) -> Callable:
    r"""Fetch a KMB scalar product callable. Integral implementation.

    Build a Kubo-Mori-Bogoliubov (KMB) scalar product function associated
    to the state ``sigma`` :cite:p:`Kubo1957,PetzToth1993`, from its integral
    form:

    .. math::

        \langle A, B\rangle_\sigma
        = \int_0^1 \mathrm{Tr}\left(\sigma^{1-\tau} A^\dagger \sigma^\tau B\right)
          d\tau
        = \sum_{i,j} \overline{\langle j|A|i\rangle}\,\langle j|B|i\rangle\,
          \int_0^1 p_i^{1-\tau} p_j^\tau d\tau,

    where :math:`\sigma|i\rangle=p_i|i\rangle`. The integral over :math:`\tau`
    is computed with a 32-point Gauss-Legendre rule, and the sum runs over the
    eigenvalues :math:`p_i>0`. It gives the same result as
    :func:`~qalma.scalarprod.build.fetch_kubo_scalar_product`, and is slower. It is useful to check the
    spectral implementation.

    Parameters
    ----------
    sigma: Operator
        The reference state: a positive semidefinite operator with unit trace.
        Its spectral decomposition is obtained from ``sigma.eigenstates()``.

    Returns
    -------
    Callable
        A function ``f(op1, op2) -> complex`` that computes the scalar product of
        two operators. The operators can be qalma operators or ``qutip.Qobj``.

    See Also
    --------
    qalma.scalarprod.build.fetch_kubo_scalar_product

    """
    evals, evecs = sigma.eigenstates()

    def return_func(op1, op2):
        """Integral-form KMB scalar product of ``op1`` and ``op2``."""
        q1, q2 = _to_qutip(op1), _to_qutip(op2)
        return sum(
            np.conj(v2.dag() * q1 * v1)
            * (v2.dag() * q2 * v1)
            * sum(
                w * p1 ** (1.0 - tau) * p2**tau
                for tau, w in zip(_TAU_NODES, _TAU_WEIGHTS)
            )
            for p1, v1 in zip(evals, evecs)
            for p2, v2 in zip(evals, evecs)
            if (p1 > 0.0 and p2 > 0.0)
        )

    return return_func


def fetch_covar_scalar_product(sigma: DensityOperatorProtocol) -> Callable:
    r"""Fetch a covariance scalar product callable.

    Returns a scalar product function based on the covariance of a density
    operator.

    The scalar product for two operators ``op1`` and ``op2`` is defined as:

    .. math::

        \frac{1}{2} \mathrm{Tr}\!\left(\sigma\,\{op_1^\dagger, op_2\}\right),

    where :math:`\sigma` is a density operator, :math:`\{A, B\}` is the
    anticommutator, and :math:`\mathrm{Tr}` the trace.

    Parameters
    ----------
    sigma : DensityOperatorProtocol
        The density operator (quantum state) used to define the scalar product.

    Returns
    -------
    Callable
        A function ``f(op1, op2) -> complex`` that computes the
        covariance-based scalar product of two operators.

    """
    return CovariantScalarProductFunction(sigma)


def fetch_hs_scalar_product() -> Callable:
    r"""Fetch a HS scalar product function.

    The Hilbert-Schmidt scalar product of two operators is
    :math:`\langle A, B\rangle=\mathrm{Tr}(A^\dagger B)`. It is not normalized by
    the dimension of the Hilbert space.

    Returns
    -------
    Callable
        A function ``f(op1, op2) -> complex`` that computes the scalar product of
        two operators.

    """
    return lambda op1, op2: (op1.dag() * op2).tr()
