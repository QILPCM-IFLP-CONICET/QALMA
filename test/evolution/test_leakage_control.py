"""Tests for the leakage-based error control of the Max-Ent evolution."""

import itertools

import numpy as np
import pytest
from scipy.linalg import eigh, expm
from scipy.optimize import minimize

from qalma import build_system
from qalma.evolution import (
    leakage_controlled_evolution,
    leakage_gram,
    leakage_integrals,
    leakage_rate,
    local_mean_field_drift,
    qutip_me_solve,
    update_basis,
)
from qalma.operators.functions import commutator
from qalma.operators.states import GibbsProductDensityOperator
from qalma.scalarprod import fetch_covar_scalar_product, trim_terms_by_tolerance

# ─────────────────────────── helpers ─────────────────────────────────────────


def spin_chain(size):
    """Return system, site names, Heisenberg+field Hamiltonian and Sx operators."""
    system = build_system("open chain lattice", "spin", L=size, J=1.0)
    sites = tuple(system.sites)
    ham = (
        system.global_operator("Hamiltonian") + system.global_operator("Sz")
    ).simplify()
    sx = [system.site_operator(f"Sx@{site}") for site in sites]
    return system, sites, ham, sx


def dense(op, sites):
    """Return the dense matrix of ``op`` in the site order ``sites``."""
    return op.to_qutip(sites).full()


def gibbs(kmat):
    """Return (log rho, rho) for rho = exp(-K)/Tr exp(-K)."""
    evals, evecs = eigh(kmat)
    evals = evals - evals.min()
    log_z = np.log(np.sum(np.exp(-evals)))
    log_rho = (evecs * (-evals - log_z)) @ evecs.conj().T
    rho = (evecs * np.exp(-evals - log_z)) @ evecs.conj().T
    return log_rho, rho


def centered_norm(op_mat, rho):
    """Return sqrt(<A^2> - <A>^2), the covariance norm of the centered operator."""
    mean = np.real(np.trace(rho @ op_mat))
    second = np.real(np.trace(rho @ op_mat @ op_mat))
    return np.sqrt(max(second - mean**2, 0.0))


def two_body_maxent_projection(rho, size):
    """Max-Ent projection onto generators with at most two-body Pauli strings."""
    paulis = [
        np.array([[0, 1], [1, 0]], complex),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]], complex),
    ]
    ops = []
    for n_sites in (1, 2):
        for sites in itertools.combinations(range(size), n_sites):
            for factors in itertools.product(paulis, repeat=n_sites):
                mats = [np.eye(2)] * size
                for site, factor in zip(sites, factors):
                    mats[site] = factor
                q = mats[0]
                for m in mats[1:]:
                    q = np.kron(q, m)
                ops.append(q)
    ops = np.array(ops)
    q_rho = np.real(np.einsum("aij,ji->a", ops, rho))

    def dual(lam):
        """Convex dual of the Max-Ent problem and its gradient."""
        evals, evecs = eigh(np.einsum("a,aij->ij", lam, ops))
        shift = evals.min()
        weights = np.exp(-(evals - shift))
        z = weights.sum()
        sigma = (evecs * (weights / z)) @ evecs.conj().T
        q_sigma = np.real(np.einsum("aij,ji->a", ops, sigma))
        return np.log(z) - shift + lam @ q_rho, q_rho - q_sigma

    res = minimize(
        dual, np.zeros(len(ops)), jac=True, method="BFGS", options={"gtol": 1e-10}
    )
    return gibbs(np.einsum("a,aij->ij", res.x, ops))


# ─────────────────────────── fixes ───────────────────────────────────────────


def test_trim_keeps_terms_above_tolerance():
    """Regression: the smallest term was always dropped, whatever its norm."""
    system, sites, ham, sx = spin_chain(4)
    sigma = GibbsProductDensityOperator(0.3 * sx[1]).to_product_state()
    sp = fetch_covar_scalar_product(sigma)
    trimmed = trim_terms_by_tolerance(sigma, ham, 1e-9)
    diff = (ham - trimmed).simplify()
    assert abs(sp(diff, diff)) < 1e-12


def test_trim_removes_terms_below_tolerance():
    """Terms whose accumulated norm is below the tolerance are removed."""
    system, sites, ham, sx = spin_chain(4)
    sigma = GibbsProductDensityOperator(0.3 * sx[1]).to_product_state()
    sp = fetch_covar_scalar_product(sigma)
    op = (ham + 1e-7 * sx[2]).simplify()
    trimmed = trim_terms_by_tolerance(sigma, op, 1e-4)
    diff = (op - trimmed).simplify()
    assert abs(sp(diff, diff)) ** 0.5 < 1e-4
    rest = (ham - trimmed).simplify()
    assert abs(sp(rest, rest)) ** 0.5 < 1e-6


@pytest.mark.parametrize(
    "k0_kind", ["local", "uniform"], ids=["local one-body", "uniform one-body"]
)
def test_update_basis_skips_dependent_reference(k0_kind):
    """For a one-body K, k_ref is in the span: the Gram must stay regular."""
    system, sites, ham, sx = spin_chain(4)
    k0 = 0.5 * sx[1] if k0_kind == "local" else (0.5 * sum(sx)).simplify()
    basis, _, _ = update_basis(
        k0.simplify(), GibbsProductDensityOperator({}, system), ham, 4, 2, ()
    )
    assert np.linalg.cond(basis.gram) < 1e8


# ─────────────────────────── leakage tensors ────────────────────────────────


@pytest.fixture(scope="module")
def basis_case():
    """Hierarchical basis for a local perturbation of a 4-site chain."""
    system, sites, ham, sx = spin_chain(4)
    k0 = 0.5 * sx[1]
    basis, sigma, _ = update_basis(
        k0.simplify(), GibbsProductDensityOperator({}, system), ham, 3, 2, ()
    )
    return system, sites, ham, k0, basis, sigma


def test_leakage_gram_is_symmetric_positive(basis_case):
    """E is symmetric, positive semidefinite and cached on the basis."""
    *_, basis, _ = basis_case
    e_mat = leakage_gram(basis)
    assert np.allclose(e_mat, e_mat.T)
    assert np.linalg.eigvalsh(e_mat).min() > -1e-12
    assert leakage_gram(basis) is e_mat  # cached


def test_leakage_rate_matches_direct_projection(basis_case):
    """sqrt(phi^T E phi) equals the norm of pi_perp(-i[H, K]) computed directly."""
    *_, basis, _ = basis_case
    e_mat = leakage_gram(basis)
    rng = np.random.default_rng(1)
    for _ in range(3):
        phi = rng.normal(size=len(basis.operator_basis))
        k_op = basis.operator_from_coefficients(phi)
        lk = commutator(k_op, basis.generator).simplify()
        parallel = basis.operator_from_coefficients(basis.coefficient_expansion(lk))
        residual = (lk - parallel).simplify()
        direct = np.sqrt(abs(np.real(basis.sp(residual, residual))))
        assert leakage_rate(e_mat, phi) == pytest.approx(direct, rel=1e-6, abs=1e-9)


def test_leakage_vanishes_for_conserved_generator():
    """A conserved generator does not leak and does not evolve."""
    system = build_system("open chain lattice", "spin", L=4, J=1.0)
    ham = system.global_operator("Hamiltonian")
    k0 = 0.4 * system.global_operator("Sz")
    sim = leakage_controlled_evolution(
        ham, k0, np.linspace(0, 1, 5), 3, 2, store_states=True
    )
    assert max(sim.stats["errors"]) < 1e-10
    assert max(sim.stats["leakage rate"]) < 1e-8
    assert sim.stats["update reasons"] == ["initial"]


def test_leakage_integrals(basis_case):
    """Propagated coefficients and leakage integrals match direct quadrature."""
    *_, k0, basis, _ = basis_case
    e_mat = leakage_gram(basis)
    gen = np.real(basis.gen_matrix)
    phi_0 = np.real(basis.coefficient_expansion(k0))
    tau = 0.7
    phi, eps, psi, rate = leakage_integrals(gen, phi_0, e_mat, tau, n_quad=200)
    assert np.allclose(phi, expm(tau * gen) @ phi_0)
    assert rate == pytest.approx(leakage_rate(e_mat, phi))
    # eps is the integral of the rate; psi the integral of phi
    taus = np.linspace(0, tau, 2001)
    phis = np.array([expm(s * gen) @ phi_0 for s in taus])
    assert eps == pytest.approx(
        np.trapezoid([leakage_rate(e_mat, p) for p in phis], taus), rel=1e-3
    )
    assert np.allclose(psi, np.trapezoid(phis, taus, axis=0), atol=1e-5)
    # tau = 0
    phi, eps, psi, _ = leakage_integrals(gen, phi_0, e_mat, 0.0)
    assert eps == 0 and np.allclose(psi, 0) and np.allclose(phi, phi_0)


# ─────────────────────────── reference drift ────────────────────────────────


def test_local_drift_vanishes_at_the_reference():
    """The drift is zero for the mean-field generator of the reference itself."""
    system, sites, ham, sx = spin_chain(4)
    kappa = (0.3 * sx[0] + 0.2 * sx[2]).simplify()
    sigma = GibbsProductDensityOperator(kappa)
    max_drift, total = local_mean_field_drift(kappa, sigma)
    assert max_drift < 1e-10 and total < 1e-10
    max_drift, total = local_mean_field_drift((kappa + 0.5 * sx[1]).simplify(), sigma)
    assert max_drift > 1e-3


def test_local_drift_is_intensive():
    """For a uniform perturbation, the max drift does not depend on L."""
    results = {}
    for size in (4, 8):
        system, sites, ham, sx = spin_chain(size)
        k = (0.3 * sum(sx)).simplify()
        results[size] = local_mean_field_drift(
            k, GibbsProductDensityOperator({}, system)
        )
    assert results[4][0] == pytest.approx(results[8][0], rel=1e-8)
    assert results[8][1] == pytest.approx(2 * results[4][1], rel=1e-8)


# ─────────────────────────── evolution ──────────────────────────────────────


@pytest.fixture(scope="module")
def evolution_case():
    """Leakage-controlled evolution of a local Sx perturbation (4 sites)."""
    system, sites, ham, sx = spin_chain(4)
    k0 = 0.5 * sx[1]
    t_span = np.linspace(0, 1.6, 9)
    e_ops = {f"x{i}": op for i, op in enumerate(sx)}
    sim = leakage_controlled_evolution(
        ham, k0, t_span, 4, 2, e_ops=e_ops, store_states=True
    )
    return system, sites, ham, k0, t_span, e_ops, sim


def test_evolution_reaches_final_time(evolution_case):
    """All output times are reached, with consistent statistics."""
    *_, t_span, _, sim = evolution_case
    assert len(sim.time_span) == len(t_span)
    assert len(sim.states) == len(t_span)
    updates = sim.stats["basis update times"]
    assert sim.stats["update reasons"][0] == "initial"
    assert all(t1 < t2 for t1, t2 in zip(updates, updates[1:]))  # never twice
    assert len(sim.stats["leakage floors"]) == len(updates)
    errors = sim.stats["errors"]
    assert all(e2 >= e1 - 1e-12 for e1, e2 in zip(errors, errors[1:]))


def test_evolution_matches_exact_observables(evolution_case):
    """Local observables agree with the exact evolution."""
    system, sites, ham, k0, t_span, e_ops, sim = evolution_case
    exact = qutip_me_solve(ham, GibbsProductDensityOperator(k0), t_span, e_ops=e_ops)
    for key in e_ops:
        approx = np.real(np.array(sim.expect_ops[key]))
        reference = np.real(np.array(exact.expect_ops[key]))
        assert np.max(np.abs(approx - reference)) < 1e-2


def test_error_estimate_bounds_true_error(evolution_case):
    """The accumulated estimate stays above ||K(t) - K~(t)|| (up to a few %)."""
    system, sites, ham, k0, t_span, e_ops, sim = evolution_case
    hmat, k0mat = dense(ham, sites), dense(k0, sites)
    for t, k_t, estimate in zip(sim.time_span, sim.states, sim.stats["errors"]):
        u = expm(-1j * t * hmat)
        k_exact = u @ k0mat @ u.conj().T
        _, rho = gibbs(k_exact)
        true_error = centered_norm(k_exact - dense(k_t, sites), rho)
        assert estimate >= 0.95 * true_error - 1e-8


def test_irreducible_entropy_estimate_short_times(evolution_case):
    """1/2 ||R||^2 estimates S(rho || Pi_2 rho) = S(Pi_2 rho) - S(rho) at short times."""
    system, sites, ham, k0, t_span, e_ops, sim = evolution_case
    hmat, k0mat = dense(ham, sites), dense(k0, sites)
    checked = 0
    for t, estimate in zip(sim.time_span, sim.stats["irreducible entropy estimate"]):
        if not 0.3 < t < 0.9:
            continue
        u = expm(-1j * t * hmat)
        log_rho, rho = gibbs(u @ k0mat @ u.conj().T)
        log_pi, _ = two_body_maxent_projection(rho, len(sites))
        s_irr = np.real(np.trace(rho @ (log_rho - log_pi)))
        assert s_irr > 0
        assert 0.7 * s_irr < estimate < 2.0 * s_irr
        checked += 1
    assert checked > 0


def test_rebuild_reasons_follow_thresholds():
    """Rebuilds happen only for the criteria whose thresholds are exceeded."""
    system, sites, ham, sx = spin_chain(4)
    k0 = (0.5 * sum(sx)).simplify()
    t_span = np.linspace(0, 1.0, 6)
    frozen = leakage_controlled_evolution(
        ham, k0, t_span, 3, 2, rate_min=1e6, drift_tol=1e6
    )
    assert frozen.stats["update reasons"] == ["initial"]
    by_drift = leakage_controlled_evolution(
        ham, k0, t_span, 3, 2, rate_min=1e6, drift_tol=1e-6
    )
    assert "drift" in by_drift.stats["update reasons"]
    assert "leakage" not in by_drift.stats["update reasons"]
