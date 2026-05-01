"""
Scalar Product on operator spaces routines.

Routines to compute scalar products of operators and related functions, like
orthogonalize sets of operators.
"""

from qalma.scalarprod.basis import HierarchicalOperatorBasis, OperatorBasis
from qalma.scalarprod.build import (
    covar_scalar_product,
    hs_scalar_product,
    kubo_integral_representation_scalar_product,
    kubo_scalar_product,
)
from qalma.scalarprod.covar import trim_terms_by_tolerance
from qalma.scalarprod.gram import gram_matrix
from qalma.scalarprod.orthogonalize import (
    build_hermitian_basis,
    operator_components,
    orthogonalize_basis,
    orthogonalize_basis_cholesky,
    orthogonalize_basis_gs,
    orthogonalize_basis_svd,
)

__all__ = [
    "OperatorBasis",
    "HierarchicalOperatorBasis",
    "build_hermitian_basis",
    "hs_scalar_product",
    "covar_scalar_product",
    "kubo_integral_representation_scalar_product",
    "kubo_scalar_product",
    "gram_matrix",
    "operator_components",
    "orthogonalize_basis",
    "orthogonalize_basis_cholesky",
    "orthogonalize_basis_gs",
    "orthogonalize_basis_svd",
    "trim_terms_by_tolerance",
]
