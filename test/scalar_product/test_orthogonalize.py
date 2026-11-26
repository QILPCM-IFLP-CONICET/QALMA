"""Tests for the functions in ``qalma.scalarprod.orthogonalize``."""

from test.helper import (
    SMINUS_A,
    SPLUS_A,
    SPLUS_B,
    SX_A,
    SX_B,
    SY_A,
    SY_B,
    SYSTEM,
    SZ_A,
    SZ_C,
    TEST_CASES_STATES,
)

import numpy as np
import pytest
from numpy.linalg import LinAlgError

from qalma.operators import ScalarOperator
from qalma.scalarprod import (
    build_hermitian_basis,
    fetch_covar_scalar_product,
    fetch_hs_scalar_product,
    gram_matrix,
    operator_components,
    orthogonalize_basis,
    orthogonalize_basis_cholesky,
    orthogonalize_basis_gs,
    orthogonalize_basis_svd,
)

SCALAR_PRODUCTS = {
    "hilbert-schmidt": fetch_hs_scalar_product(),
    "covariance, x semipolarized": fetch_covar_scalar_product(
        TEST_CASES_STATES["x semipolarized"]
    ),
}

ORTHOGONALIZERS = {
    "default": orthogonalize_basis,
    "gram-schmidt": orthogonalize_basis_gs,
    "cholesky": orthogonalize_basis_cholesky,
    "svd": orthogonalize_basis_svd,
}

# A linearly independent, non-orthogonal set of operators.
BASIS = [
    SX_A,
    SX_A + SY_A,
    SZ_A + SX_B,
    SX_A * SX_B + SZ_A,
    SY_B,
]

ZERO = ScalarOperator(0.0, SYSTEM)


def operator_distance(op1, op2):
    """Largest entry of the difference between the matrices of two operators."""
    return np.abs((op1 - op2).to_qutip().full()).max()


def gram_error(basis, scalar_product):
    """Distance between the Gram matrix of ``basis`` and the identity."""
    return np.abs(gram_matrix(basis, scalar_product) - np.eye(len(basis))).max()


@pytest.mark.parametrize("sp_name", list(SCALAR_PRODUCTS))
@pytest.mark.parametrize("method_name", list(ORTHOGONALIZERS))
def test_orthogonalize_basis_is_orthonormal(method_name, sp_name):
    """The result is orthonormal and spans the same space as the input."""
    scalar_product = SCALAR_PRODUCTS[sp_name]
    orth_basis = ORTHOGONALIZERS[method_name](BASIS, scalar_product)

    assert len(orth_basis) == len(BASIS)
    assert gram_error(orth_basis, scalar_product) < 1e-8

    # Every original operator is recovered from its components.
    for op in BASIS:
        components = operator_components(op, orth_basis, scalar_product)
        rebuilt = sum(c * o for c, o in zip(components, orth_basis))
        assert operator_distance(rebuilt, op) < 1e-8


def test_orthogonalize_basis_default_is_gram_schmidt():
    """``orthogonalize_basis`` delegates to the Gram-Schmidt implementation."""
    scalar_product = SCALAR_PRODUCTS["hilbert-schmidt"]
    default = orthogonalize_basis(BASIS, scalar_product)
    gram_schmidt = orthogonalize_basis_gs(BASIS, scalar_product)
    assert len(default) == len(gram_schmidt)
    for op_1, op_2 in zip(default, gram_schmidt):
        assert operator_distance(op_1, op_2) < 1e-12


@pytest.mark.parametrize("sp_name", list(SCALAR_PRODUCTS))
def test_orthogonalize_gram_schmidt_is_triangular(sp_name):
    """Gram-Schmidt keeps the order: the first element is the normalized one."""
    scalar_product = SCALAR_PRODUCTS[sp_name]
    orth_basis = orthogonalize_basis_gs(BASIS, scalar_product)
    norm = abs(scalar_product(BASIS[0], BASIS[0])) ** 0.5
    assert operator_distance(orth_basis[0], BASIS[0] / norm) < 1e-12


@pytest.mark.parametrize("sp_name", list(SCALAR_PRODUCTS))
def test_orthogonalize_gram_schmidt_discards_dependent_and_null(sp_name):
    """Linearly dependent and null operators are dropped."""
    scalar_product = SCALAR_PRODUCTS[sp_name]
    basis = [SX_A, ZERO, SY_A, SX_A + 2 * SY_A, 3 * SX_A]
    orth_basis = orthogonalize_basis_gs(basis, scalar_product)
    assert len(orth_basis) == 2
    assert gram_error(orth_basis, scalar_product) < 1e-8


@pytest.mark.parametrize("method_name", list(ORTHOGONALIZERS))
def test_orthogonalize_empty_basis(method_name):
    """An empty basis has nothing to orthogonalize."""
    scalar_product = SCALAR_PRODUCTS["hilbert-schmidt"]
    assert not list(ORTHOGONALIZERS[method_name]([], scalar_product))


SINGULAR_BASES = {
    "collinear": [SX_A, 2 * SX_A],
    "dependent": [SX_A, SY_A, SX_A + 2 * SY_A],
    "with a null operator": [SX_A, ZERO, SY_A],
}


@pytest.mark.parametrize("case", list(SINGULAR_BASES))
@pytest.mark.parametrize(
    "method", [orthogonalize_basis_cholesky, orthogonalize_basis_svd]
)
def test_orthogonalize_singular_basis_raises(method, case):
    """Cholesky and SVD cannot orthogonalize linearly dependent operators."""
    with pytest.raises(LinAlgError):
        method(SINGULAR_BASES[case], SCALAR_PRODUCTS["hilbert-schmidt"])


@pytest.mark.parametrize("sp_name", list(SCALAR_PRODUCTS))
def test_operator_components(sp_name):
    """Components of operators in span, and of a projection."""
    scalar_product = SCALAR_PRODUCTS[sp_name]
    orth_basis = orthogonalize_basis_gs(BASIS, scalar_product)

    # An operator inside the span.
    op = 2 * orth_basis[0] - 3 * orth_basis[2] + 0.5 * orth_basis[3]
    components = operator_components(op, orth_basis, scalar_product)
    assert isinstance(components, np.ndarray)
    assert components.shape == (len(orth_basis),)
    assert np.abs(components - np.array([2.0, 0.0, -3.0, 0.5, 0.0])).max() < 1e-8

    # For an operator outside of the span, the residual of the projection
    # is orthogonal to the basis.
    outside = SX_A + SZ_C + SPLUS_B * 0.1 + SPLUS_B.dag() * 0.1
    components = operator_components(outside, orth_basis, scalar_product)
    residual = outside - sum(c * o for c, o in zip(components, orth_basis))
    assert (
        np.abs(operator_components(residual, orth_basis, scalar_product)).max() < 1e-8
    )


# Mix of hermitian and non-hermitian operators.
HERMITIAN_INPUTS = {
    "hermitian": [SX_A, SZ_A, SX_A * SX_B],
    "non hermitian": [SPLUS_A, SMINUS_A * SPLUS_B],
    "mixed": [SX_A, SPLUS_A, SMINUS_A * SPLUS_B],
    "dependent": [SX_A, 2 * SX_A, SPLUS_A, SPLUS_A.dag()],
    "with null operators": [ZERO, SX_A, ZERO, SPLUS_A],
    "anti-hermitian": [1j * SX_A, SPLUS_A - SPLUS_A.dag()],
}

HERMITIAN_BASIS_SIZES = {
    "hermitian": 3,
    "non hermitian": 4,  # sx, sy and two hermitian combinations
    "mixed": 4,  # sx is already generated by s+
    "dependent": 2,  # sx, sy
    "with null operators": 2,  # sx, sy
    "anti-hermitian": 2,  # sx, sy
}


@pytest.mark.parametrize("sp_name", [None, "covariance, x semipolarized"])
@pytest.mark.parametrize("case", list(HERMITIAN_INPUTS))
def test_build_hermitian_basis(case, sp_name):
    """The new basis is hermitian, orthonormal, and generates the input."""
    basis = HERMITIAN_INPUTS[case]
    if sp_name is None:
        scalar_product = fetch_hs_scalar_product()
        new_basis, coeffs = build_hermitian_basis(basis)
    else:
        scalar_product = SCALAR_PRODUCTS[sp_name]
        new_basis, coeffs = build_hermitian_basis(basis, scalar_product)

    assert len(new_basis) == HERMITIAN_BASIS_SIZES[case]
    assert all(op.isherm for op in new_basis)
    assert gram_error(new_basis, scalar_product) < 1e-8

    # ``coeffs[i, j]`` is the coefficient of ``new_basis[j]`` in ``basis[i]``.
    assert coeffs.shape == (len(basis), len(new_basis))
    for op, row in zip(basis, coeffs):
        rebuilt = sum(c * new_op for c, new_op in zip(row, new_basis))
        assert operator_distance(rebuilt, op) < 1e-8


@pytest.mark.parametrize("sp_name", [None, "covariance, x semipolarized"])
def test_build_hermitian_basis_null_operators(sp_name):
    """Null operators are filtered out, and their coefficients are zero."""
    basis = [ZERO, SX_A, ZERO, SPLUS_A]
    args = () if sp_name is None else (SCALAR_PRODUCTS[sp_name],)
    _, coeffs = build_hermitian_basis(basis, *args)
    assert np.all(coeffs[0] == 0.0)
    assert np.all(coeffs[2] == 0.0)
    assert np.abs(coeffs[1]).max() > 0.0
    assert np.abs(coeffs[3]).max() > 0.0


def test_build_hermitian_basis_only_null_operators():
    """If every operator is null, the new basis is empty."""
    new_basis, coeffs = build_hermitian_basis([ZERO, 0 * ZERO])
    assert not new_basis
    assert coeffs.shape == (2, 0)
