r"""Quadratic-form representation of two-body quantum operators.

This subpackage provides a structured representation for operators of the form

.. math::

    T = L + \sum_\alpha w_\alpha Q_\alpha^2 + \delta T,

where :math:`L` is a one-body (linear) term, each :math:`Q_\alpha` is a
Hermitian one-body operator normalised to spectral norm 1, :math:`w_\alpha`
are real weights, and :math:`\delta T` collects any remainder that cannot be
captured by the quadratic or linear parts (e.g. three-body terms).

The decomposition is the core data structure consumed by the variational
mean-field approximation implemented in :mod:`qalma.meanfield.variational`.
Given a many-body Hamiltonian :math:`K`, the workflow is:

1. Project :math:`K` onto its two-body sector relative to a reference state
   :math:`\sigma` using :func:`~qalma.projections.n_body_projection`.
2. Convert the projected operator to a
   :class:`QuadraticFormOperator` via :func:`build_quadratic_form_from_operator`,
   optionally keeping only the ``count`` modes with the most negative
   eigenvalues (the modes that lower the free energy the most).
3. Use the resulting object to build a trial state
   :math:`\sigma = e^{-\kappa}/Z` and minimise the variational free energy
   :math:`F[\sigma] = \operatorname{Tr}[\sigma(\kappa + \log\sigma)]`.

Public API
----------
QuadraticFormOperator
    Operator class storing the decomposition :math:`(L,\{w_\alpha\},\{Q_\alpha\},\delta T)`.
build_quadratic_form_from_operator
    Factory function that constructs a :class:`QuadraticFormOperator` from any
    :class:`~qalma.operators.basic.Operator` by diagonalising its two-body
    coupling matrix.
"""

from .build import build_quadratic_form_from_operator
from .quadratic import QuadraticFormOperator

__all__ = [
    "QuadraticFormOperator",
    "build_quadratic_form_from_operator",
]
