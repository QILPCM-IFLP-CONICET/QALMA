"""Tests for the restricted Max-Ent evolution with a memory closure."""

import warnings

import numpy as np
import pytest
from scipy.linalg import eigh, expm

from qalma import build_system
from qalma.evolution import (
    adaptive_projected_evolution,
    estimate_memory_time,
    leakage_controlled_evolution,
    memory_projected_evolution,
    sector_leakage_fraction,
    update_basis,
)
from qalma.operators.states import GibbsProductDensityOperator

# ─────────────────────────── helpers ─────────────────────────────────────────


def chain(size):
    """Spin chain, site names and the Sx, Sy, Sz operators of each site."""
    system = build_system("open chain lattice", "spin", L=size, J=1.0)
    sites = tuple(system.sites)
    ops = {
        name: [system.site_operator(f"{name}@{site}") for site in sites]
        for name in ("Sx", "Sy", "Sz")
    }
    return system, sites, ops


def xxz_precession(size):
    """Uniformly x-polarized state in an XXZ chain with a field along z."""
    system, sites, ops = chain(size)
    sx, sy, sz = ops["Sx"], ops["Sy"], ops["Sz"]
    ham = (
        sum(
            sx[i] * sx[i + 1] + sy[i] * sy[i + 1] + 0.5 * sz[i] * sz[i + 1]
            for i in range(size - 1)
        )
        + sum(sz)
    ).simplify()
    k0 = (0.5 * sum(sx)).simplify()
    return system, sites, ops, ham, k0


def heisenberg_field(size):
    """Local Sx perturbation of a Heisenberg chain in a field."""
    system, sites, ops = chain(size)
    ham = (
        system.global_operator("Hamiltonian") + system.global_operator("Sz")
    ).simplify()
    return system, sites, ops, ham, 0.5 * ops["Sx"][1]


def gibbs(kmat):
    """rho = exp(-K) / Tr exp(-K) for a dense K."""
    evals, evecs = eigh(kmat)
    weights = np.exp(-(evals - evals.min()))
    return (evecs * (weights / weights.sum())) @ evecs.conj().T


def local_errors(states, ham, k0, ops, sites, t_span):
    """Maximum error of <Sx_i> along the evolution, against the exact one."""
    hmat = ham.to_qutip(sites).full()
    k0mat = k0.to_qutip(sites).full()
    mats = [op.to_qutip(sites).full() for op in ops["Sx"]]
    worst = 0.0
    for t, k_t in zip(t_span, states):
        u = expm(-1j * t * hmat)
        rho = gibbs(u @ k0mat @ u.conj().T)
        sigma = gibbs(k_t.to_qutip(sites).full())
        for m in mats:
            worst = max(worst, abs(np.trace((rho - sigma) @ m)))
    return worst


@pytest.fixture(scope="module")
def precession_runs():
    """First-order and memory evolutions of the XXZ precession (4 sites)."""
    system, sites, ops, ham, k0 = xxz_precession(4)
    t_span = np.linspace(0, 4, 21)
    first = leakage_controlled_evolution(ham, k0, t_span, 4, 2, store_states=True)
    memory = memory_projected_evolution(ham, k0, t_span, 4, 2, store_states=True)
    return system, sites, ops, ham, k0, t_span, first, memory


# ─────────────────────────── memory time ────────────────────────────────────


def test_memory_time_scales_as_inverse_energy():
    """tau is the inverse of an energy scale: H -> 2H gives tau / 2."""
    system, sites, ops, ham, k0 = heisenberg_field(4)
    sigma = GibbsProductDensityOperator({}, system)
    k = (k0 + 0.3 * ops["Sx"][1] * ops["Sx"][2]).simplify()
    taus = []
    for scale in (1.0, 2.0):
        basis, _, _ = update_basis(k, sigma, scale * ham, 2, 2, ())
        tau, leak, second = estimate_memory_time(basis, k)
        assert leak > 0 and second > 0
        taus.append(tau)
    assert taus[1] == pytest.approx(taus[0] / 2, rel=1e-6)


def test_memory_time_fallback_without_leakage():
    """A conserved generator does not leak: the fallback value is returned."""
    system, sites, ops = chain(4)
    ham = system.global_operator("Hamiltonian")
    k0 = 0.4 * system.global_operator("Sz")
    basis, _, _ = update_basis(
        k0.simplify(), GibbsProductDensityOperator({}, system), ham, 2, 2, ()
    )
    tau, leak, _ = estimate_memory_time(basis, k0, fallback=0.123)
    assert tau == 0.123
    assert leak < 1e-8


# ─────────────────────────── sector leakage ─────────────────────────────────


def test_sector_leakage_fraction_detects_three_body_dynamics():
    """For a Heisenberg chain and an XY generator, [H, K] is purely three-body."""
    system, sites, ops = chain(4)
    sx, sy = ops["Sx"], ops["Sy"]
    ham = system.global_operator("Hamiltonian")
    k = (0.5 * sum(sx[i] * sx[i + 1] + sy[i] * sy[i + 1] for i in range(3))).simplify()
    sigma = GibbsProductDensityOperator({}, system)
    assert sector_leakage_fraction(ham, k, sigma, 2) == pytest.approx(1.0, abs=1e-8)
    assert sector_leakage_fraction(ham, k, sigma, 3) == pytest.approx(0.0, abs=1e-8)


def test_warning_when_the_sector_is_too_small():
    """The evolution warns when the dynamics leaves the n-body sector."""
    system, sites, ops = chain(4)
    sx, sy = ops["Sx"], ops["Sy"]
    ham = system.global_operator("Hamiltonian")
    k0 = (0.5 * sum(sx[i] * sx[i + 1] + sy[i] * sy[i + 1] for i in range(3))).simplify()
    with pytest.warns(RuntimeWarning, match="outside the 2-body sector"):
        memory_projected_evolution(ham, k0, np.linspace(0, 0.4, 3), 2, 2)


# ─────────────────────────── evolution ──────────────────────────────────────


def test_without_memory_reproduces_first_order():
    """memory_time=None gives exactly leakage_controlled_evolution."""
    system, sites, ops, ham, k0 = heisenberg_field(4)
    t_span = np.linspace(0, 2, 11)
    first = leakage_controlled_evolution(ham, k0, t_span, 4, 2, store_states=True)
    no_memory = memory_projected_evolution(
        ham, k0, t_span, 4, 2, memory_time=None, store_states=True
    )
    assert first.stats["basis update times"] == no_memory.stats["basis update times"]
    for k_1, k_2 in zip(first.states, no_memory.states):
        diff = (k_1 - k_2).to_qutip(sites).full()
        assert np.abs(diff).max() < 1e-8


def test_short_memory_approaches_first_order():
    """A very short memory time gives almost the first-order evolution."""
    system, sites, ops, ham, k0 = heisenberg_field(4)
    t_span = np.linspace(0, 1, 6)
    first = memory_projected_evolution(
        ham, k0, t_span, 4, 2, memory_time=None, store_states=True
    )
    short = memory_projected_evolution(
        ham, k0, t_span, 4, 2, memory_time=1e-3, store_states=True
    )
    for k_1, k_2 in zip(first.states, short.states):
        diff = (k_1 - k_2).to_qutip(sites).full()
        assert np.abs(diff).max() < 1e-2


def test_memory_improves_local_observables(precession_runs):
    """The memory closure reduces the error of <Sx_i> against the exact one."""
    system, sites, ops, ham, k0, t_span, first, memory = precession_runs
    err_first = local_errors(first.states, ham, k0, ops, sites, t_span)
    err_memory = local_errors(memory.states, ham, k0, ops, sites, t_span)
    assert err_memory < 0.5 * err_first


def test_statistics_and_continuity(precession_runs):
    """Per-rebuild statistics are consistent, and the velocity is continuous."""
    *_, t_span, first, memory = precession_runs
    stats = memory.stats
    n_rebuilds = len(stats["basis update times"])
    assert n_rebuilds > 1
    for key in (
        "update reasons",
        "leakage floors",
        "memory times",
        "sector leakage fraction",
        "velocity jump",
        "basis time costs",
    ):
        assert len(stats[key]) == n_rebuilds, key
    assert len(memory.time_span) == len(t_span)
    assert all(tau > 0 for tau in stats["memory times"])
    assert max(stats["velocity jump"]) < 1e-6
    errors = stats["errors"]
    assert all(e_2 >= e_1 - 1e-12 for e_1, e_2 in zip(errors, errors[1:]))


def test_velocity_jumps_without_continuity():
    """Without continuous_velocity, the velocity jumps at the rebuilds."""
    system, sites, ops, ham, k0 = xxz_precession(4)
    sim = memory_projected_evolution(
        ham,
        k0,
        np.linspace(0, 2, 11),
        4,
        2,
        memory_time=0.8,
        continuous_velocity=False,
    )
    assert max(sim.stats["velocity jump"]) > 1e-3


def test_auto_memory_time_is_estimated_once_leakage_appears():
    """With a local one-body K(0) there is no initial leakage; tau is estimated later."""
    system, sites, ops, ham, k0 = heisenberg_field(4)
    sim = memory_projected_evolution(
        ham, k0, np.linspace(0, 1, 6), 4, 2, default_memory_time=0.123
    )
    assert sim.stats["memory times"][0] != 0.123
    assert sim.stats["memory times"][0] > 0


def test_invalid_memory_time():
    """Negative or zero memory times are rejected."""
    system, sites, ops, ham, k0 = heisenberg_field(4)
    with pytest.raises(ValueError):
        memory_projected_evolution(ham, k0, [0.0, 0.1], 2, 2, memory_time=0.0)


def test_adaptive_projected_evolution_is_deprecated():
    """The previous adaptive evolution still works but is deprecated."""
    system, sites, ops, ham, k0 = heisenberg_field(4)
    with pytest.warns(DeprecationWarning, match="memory_projected_evolution"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            adaptive_projected_evolution(ham, k0, [0.0, 0.1], 2, 2)
