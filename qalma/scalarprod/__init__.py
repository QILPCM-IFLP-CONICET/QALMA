"""

Routines to compute scalar products of operators and related functions,
like orthogonalize sets of operators.

"""

from qalma.scalarprod.basis import HierarchicalOperatorBasis, OperatorBasis
from qalma.scalarprod.build import (
    fetch_covar_scalar_product,
    fetch_HS_scalar_product,
    fetch_kubo_int_scalar_product,
    fetch_kubo_scalar_product,
)
from qalma.scalarprod.gram import gram_matrix
from qalma.scalarprod.orthogonalize import (
    build_hermitician_basis,
    operator_components,
    orthogonalize_basis,
    orthogonalize_basis_cholesky,
    orthogonalize_basis_gs,
    orthogonalize_basis_svd,
)

__all__ = [
    "OperatorBasis",
    "HierarchicalOperatorBasis",
    "build_hermitician_basis",
    "fetch_HS_scalar_product",
    "fetch_covar_scalar_product",
    "fetch_kubo_int_scalar_product",
    "fetch_kubo_scalar_product",
    "gram_matrix",
    "operator_components",
    "orthogonalize_basis",
    "orthogonalize_basis_cholesky",
    "orthogonalize_basis_gs",
    "orthogonalize_basis_svd",
]
