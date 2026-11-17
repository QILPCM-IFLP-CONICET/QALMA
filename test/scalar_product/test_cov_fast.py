"""Tests for the fast covariance-product kernel relative to product states."""

import gc

import numpy as np
import pytest
import qutip

from qalma import build_system
from qalma.operators import ProductOperator, QutipOperator
from qalma.operators.states import GibbsProductDensityOperator
from qalma.scalarprod import _cov_fast, covar, fetch_covar_scalar_product

SIZE = 5


@pytest.fixture(scope="module")
def chain():
    """Spin chain, its sites, and a few building blocks."""
    system = build_system("open chain lattice", "spin", L=SIZE, J=1.0)
    sites = tuple(system.sites)
    ops = {
        name: [system.site_operator(f"{name}@{site}") for site in sites]
        for name in ("Sx", "Sy", "Sz", "Splus", "Sminus")
    }
    ham = system.global_operator("Hamiltonian")
    return system, sites, ops, ham


def random_product_state(system, sites, ops, seed):
    """A product state with random, non-diagonal local states."""
    rng = np.random.default_rng(seed)
    kappa = sum(
        rng.normal() * ops["Sx"][i]
        + rng.normal() * ops["Sy"][i]
        + rng.normal() * ops["Sz"][i]
        for i in range(len(sites))
    )
    return GibbsProductDensityOperator(kappa.simplify()).to_product_state()


def operator_cases(system, sites, ops, ham):
    """Operators with local, product, dense, scalar and non-Hermitian terms."""
    sx, sz, sp_, sm = ops["Sx"], ops["Sz"], ops["Splus"], ops["Sminus"]
    return {
        "local": 0.3 * sx[0] + 0.7 * sz[3],
        "hamiltonian": ham.simplify(),
        "ham_times_local": (ham * sx[1]).simplify(),
        "products_unsimplified": sx[0] * sz[1] + 2.0 * sz[2] * sx[4],
        "non_hermitian_products": sp_[1] * sz[2] + 0.5j * sm[3],
        "with_scalar": (ham + 1.5).simplify(),
        "three_body": (sx[0] * sz[1] * sx[2] + sz[2] * sz[3] * sz[4]).simplify(),
        "ham_squared": (ham * ham).simplify(),
        "dense_block": QutipOperator(
            qutip.Qobj(
                np.random.default_rng(3).normal(size=(8, 8))
                + 1j * np.random.default_rng(4).normal(size=(8, 8)),
                dims=[[2, 2, 2], [2, 2, 2]],
            ),
            system,
            names={sites[1]: 0, sites[3]: 1, sites[2]: 2},
        ),
    }


def exact_cov(rho, op_1, op_2, sites):
    """<{op_1^dagger, op_2}>/2 with dense matrices."""
    r = rho.to_qutip(sites)
    a = op_1.to_qutip(sites)
    b = op_2.to_qutip(sites)
    return 0.5 * (r * (a.dag() * b + b * a.dag())).tr()


@pytest.fixture(scope="module")
def cases(chain):
    """Reference state and operator cases."""
    system, sites, ops, ham = chain
    rho = random_product_state(system, sites, ops, seed=7)
    return rho, operator_cases(system, sites, ops, ham)


@pytest.mark.parametrize(
    "name_1",
    [
        "local",
        "hamiltonian",
        "ham_times_local",
        "products_unsimplified",
        "non_hermitian_products",
        "with_scalar",
        "three_body",
        "ham_squared",
        "dense_block",
    ],
)
@pytest.mark.parametrize(
    "name_2",
    [
        "local",
        "hamiltonian",
        "products_unsimplified",
        "non_hermitian_products",
        "three_body",
        "dense_block",
    ],
)
def test_scalar_product_matches_dense(chain, cases, name_1, name_2):
    """The covariance product agrees with the dense computation."""
    _, sites, _, _ = chain
    rho, ops = cases
    sp = fetch_covar_scalar_product(rho)
    op_1, op_2 = ops[name_1], ops[name_2]
    value = sp(op_1, op_2)
    expected = exact_cov(rho, op_1, op_2, sites)
    if op_1.isherm and op_2.isherm:
        expected = np.real(expected)
    assert value == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    "name",
    [
        "local",
        "hamiltonian",
        "non_hermitian_products",
        "three_body",
        "ham_squared",
        "dense_block",
    ],
)
def test_norm_matches_dense(chain, cases, name):
    """||A||^2 = <{A^dagger, A}>/2, also for operators with non-normal terms.

    Regression: the general implementation used <T^dagger T> for the diagonal
    contributions, which is wrong for non-normal terms.
    """
    _, sites, _, _ = chain
    rho, ops = cases
    sp = fetch_covar_scalar_product(rho)
    op = ops[name]
    assert sp(op, op) == pytest.approx(
        np.real(exact_cov(rho, op, op, sites)), abs=1e-10
    )


def test_fast_kernel_used_for_supported_terms(chain, cases):
    """All the terms of the cases are supported by the fast kernel."""
    rho, ops = cases
    for name, op in ops.items():
        flat = op.simplify().flat()
        assert _cov_fast.fast_cov_prod_sp(rho, flat, flat, op.isherm) is not None, name


def test_fallback_for_unsupported_terms(chain, cases, monkeypatch):
    """Dense terms larger than MAX_DENSE_SITES use the general implementation."""
    _, sites, _, _ = chain
    rho, ops = cases
    op = ops["hamiltonian"]  # dense two-site terms after simplify
    flat = op.simplify().flat()
    monkeypatch.setattr(_cov_fast, "MAX_DENSE_SITES", 1)
    assert _cov_fast.fast_cov_prod_sp(rho, flat, flat, True) is None
    sp = fetch_covar_scalar_product(rho)
    assert sp(op, ops["local"]) == pytest.approx(
        np.real(exact_cov(rho, op, ops["local"], sites)), abs=1e-8
    )


def test_switch_off_fast_kernel(chain, cases, monkeypatch):
    """With USE_FAST_KERNEL=False, Hermitian products agree with the fast kernel."""
    rho, ops = cases
    sp = fetch_covar_scalar_product(rho)
    pairs = [("hamiltonian", "ham_times_local"), ("three_body", "local")]
    fast = [sp(ops[a], ops[b]) for a, b in pairs]
    monkeypatch.setattr(covar, "USE_FAST_KERNEL", False)
    general = [sp(ops[a], ops[b]) for a, b in pairs]
    assert np.allclose(fast, general, atol=1e-8)


def test_gram_matrix_matches_dense(chain, cases):
    """compute_gram_matrix uses the fast kernel and agrees with dense matrices."""
    _, sites, _, _ = chain
    rho, ops = cases
    basis = tuple(
        ops[name] for name in ("local", "hamiltonian", "three_body", "with_scalar")
    )
    gram = fetch_covar_scalar_product(rho).compute_gram_matrix(basis)
    expected = np.array(
        [[np.real(exact_cov(rho, a, b, sites)) for b in basis] for a in basis]
    )
    assert np.allclose(gram, expected, atol=1e-10)


def test_cache_survives_temporary_operators(chain):
    """Reusing the cache with short-lived operators never mixes their data."""
    system, sites, ops, ham = chain
    rho = random_product_state(system, sites, ops, seed=11)
    sp = fetch_covar_scalar_product(rho)
    rng = np.random.default_rng(5)
    for _ in range(20):
        coeffs = rng.normal(size=3)
        i, j = rng.choice(SIZE, size=2, replace=False)
        op = (
            coeffs[0] * ops["Sx"][i] * ops["Sz"][j]
            + coeffs[1] * ops["Sy"][j]
            + coeffs[2] * ham
        ).simplify()
        assert sp(op, op) == pytest.approx(
            np.real(exact_cov(rho, op, op, sites)), abs=1e-10
        )
        del op
        gc.collect()


def test_cache_bound_to_state(chain, cases):
    """A cache built for another state is not used."""
    system, sites, ops, ham = chain
    rho, cases_ops = cases
    other = random_product_state(system, sites, ops, seed=99)
    cache = _cov_fast.DenseTermCache(other)
    op = cases_ops["hamiltonian"].simplify().flat()
    value = _cov_fast.fast_cov_prod_sp(rho, op, op, True, cache)
    assert value == pytest.approx(np.real(exact_cov(rho, op, op, sites)), abs=1e-10)


def test_product_terms_are_not_densified(chain):
    """Pairs of product terms are evaluated site by site."""
    system, sites, ops, ham = chain
    rho = random_product_state(system, sites, ops, seed=2)
    long_product = (
        ops["Sx"][0] * ops["Sz"][1] * ops["Sx"][2] * ops["Sz"][3] * ops["Sx"][4]
    )
    assert isinstance(long_product, ProductOperator)
    cache = _cov_fast.DenseTermCache(rho)
    value = _cov_fast.fast_cov_prod_sp(rho, long_product, long_product, True, cache)
    assert value == pytest.approx(
        np.real(exact_cov(rho, long_product, long_product, sites)), abs=1e-12
    )
    assert not cache._reduced  # no reduction to dense matrices was needed
