"""Tests for the scalar products built by ``qalma.scalarprod.build``."""

from test.helper import (
    SMINUS_A,
    SPLUS_A,
    SX_A,
    SX_B,
    SY_A,
    SY_B,
    SYSTEM,
    SZ_A,
    TEST_CASES_STATES,
)

import numpy as np
import pytest

from qalma.operators import ScalarOperator
from qalma.scalarprod.build import (
    fetch_hs_scalar_product,
    fetch_kubo_integral_representation_scalar_product,
    fetch_kubo_scalar_product,
)

KUBO_BUILDERS = {
    "spectral": fetch_kubo_scalar_product,
    "integral": fetch_kubo_integral_representation_scalar_product,
}

# Full rank, almost degenerate (product states), and rank-deficient states.
STATE_NAMES = (
    "fully mixed",
    "z semipolarized",
    "x semipolarized",
    "first full polarized",
    "gibbs_H",
)

# Pairs of operators that do not commute with the states above.
OPERATOR_PAIRS = {
    "sx, sx": (SX_A, SX_A),
    "sx, sz": (SX_A, SZ_A),
    "s+, s-": (SPLUS_A, SMINUS_A),
    "s+, s+": (SPLUS_A, SPLUS_A),
    "sx sx, sy sy + sz": (SX_A * SX_B, SY_A * SY_B + SZ_A),
}

IDENTITY = ScalarOperator(1.0, SYSTEM)


def kmb_reference(sigma, op1, op2, n_nodes=40):
    r"""Dense reference for the KMB scalar product.

    ``\int_0^1 Tr(sigma^{1-t} op1^dagger sigma^t op2) dt``, computed with a
    Gauss-Legendre rule over the eigendecomposition of ``sigma``.
    """
    evals, evecs = np.linalg.eigh(sigma.to_qutip().full())
    evals = np.clip(evals, 0.0, None)
    mat1, mat2 = op1.to_qutip().full(), op2.to_qutip().full()
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    result = 0.0
    for tau, weight in zip(0.5 * (nodes + 1.0), 0.5 * weights):
        sigma_1 = (evecs * evals ** (1.0 - tau)) @ evecs.conj().T
        sigma_2 = (evecs * evals**tau) @ evecs.conj().T
        result += weight * np.trace(sigma_1 @ mat1.conj().T @ sigma_2 @ mat2)
    return result


@pytest.mark.parametrize("builder_name", list(KUBO_BUILDERS))
@pytest.mark.parametrize("state_name", STATE_NAMES)
def test_kubo_normalization(builder_name, state_name):
    """<1, 1> = Tr(sigma) = 1."""
    sigma = TEST_CASES_STATES[state_name]
    scalar_product = KUBO_BUILDERS[builder_name](sigma)
    assert abs(scalar_product(IDENTITY, IDENTITY) - 1.0) < 1e-10


@pytest.mark.parametrize("builder_name", list(KUBO_BUILDERS))
@pytest.mark.parametrize("pair_name", list(OPERATOR_PAIRS))
@pytest.mark.parametrize("state_name", STATE_NAMES)
def test_kubo_matches_dense_reference(builder_name, state_name, pair_name):
    """Both implementations reproduce the dense KMB scalar product."""
    sigma = TEST_CASES_STATES[state_name]
    op1, op2 = OPERATOR_PAIRS[pair_name]
    scalar_product = KUBO_BUILDERS[builder_name](sigma)
    expected = kmb_reference(sigma, op1, op2)
    assert abs(scalar_product(op1, op2) - expected) < 1e-10


@pytest.mark.parametrize("builder_name", list(KUBO_BUILDERS))
@pytest.mark.parametrize("pair_name", list(OPERATOR_PAIRS))
def test_kubo_scalar_product_properties(builder_name, pair_name):
    """Conjugate symmetry and positivity of <A, A>."""
    sigma = TEST_CASES_STATES["x semipolarized"]
    op1, op2 = OPERATOR_PAIRS[pair_name]
    scalar_product = KUBO_BUILDERS[builder_name](sigma)
    assert abs(scalar_product(op1, op2) - np.conj(scalar_product(op2, op1))) < 1e-10
    norm_sq = scalar_product(op1, op1)
    assert abs(norm_sq.imag) < 1e-10
    assert norm_sq.real > 0.0


@pytest.mark.parametrize("builder_name", list(KUBO_BUILDERS))
def test_kubo_accepts_qutip_operators(builder_name):
    """The scalar products work on ``Qobj`` as well as on qalma operators."""
    sigma = TEST_CASES_STATES["z semipolarized"]
    scalar_product = KUBO_BUILDERS[builder_name](sigma)
    expected = scalar_product(SPLUS_A, SY_A)
    result = scalar_product(SPLUS_A.to_qutip(), SY_A.to_qutip())
    assert abs(result - expected) < 1e-12


@pytest.mark.parametrize("builder_name", list(KUBO_BUILDERS))
def test_kubo_fully_mixed_is_hilbert_schmidt(builder_name):
    """For sigma = 1/d, KMB reduces to Tr(A^dag B) / d."""
    sigma = TEST_CASES_STATES["fully mixed"]
    dim = np.prod(list(SYSTEM.dimensions.values()))
    scalar_product = KUBO_BUILDERS[builder_name](sigma)
    for op1, op2 in OPERATOR_PAIRS.values():
        expected = (op1.dag() * op2).tr() / dim
        assert abs(scalar_product(op1, op2) - expected) < 1e-10


def test_kubo_commuting_operators_reduce_to_covariance():
    """If A and B commute with sigma, <A, B> = Tr(sigma A^dag B)."""
    sigma = TEST_CASES_STATES["gibbs_sz"]
    expected = (sigma * (SZ_A.dag() * SZ_A)).tr()
    for builder in KUBO_BUILDERS.values():
        assert abs(builder(sigma)(SZ_A, SZ_A) - expected) < 1e-10


def test_kubo_spectral_threshold():
    """With ``threshold`` the lowest eigenvalues of sigma are dropped."""
    sigma = TEST_CASES_STATES["z semipolarized"]
    full = fetch_kubo_scalar_product(sigma)(IDENTITY, IDENTITY).real
    truncated = fetch_kubo_scalar_product(sigma, threshold=0.2)(IDENTITY, IDENTITY).real
    # The weight that is left out is below the threshold.
    assert 0.8 < truncated < full


def test_hs_scalar_product():
    """``fetch_hs_scalar_product`` computes Tr(A^dag B)."""
    scalar_product = fetch_hs_scalar_product()
    dim = np.prod(list(SYSTEM.dimensions.values()))
    assert abs(scalar_product(SX_A, SX_A) - dim / 4) < 1e-12
    assert abs(scalar_product(SX_A, SY_A)) < 1e-12
    assert abs(scalar_product(SX_A, SX_B)) < 1e-12
    assert abs(scalar_product(SPLUS_A, SPLUS_A) - dim / 2) < 1e-12
    assert abs(scalar_product(SPLUS_A, SMINUS_A)) < 1e-12
    assert abs(scalar_product(SPLUS_A, SY_B)) < 1e-12
