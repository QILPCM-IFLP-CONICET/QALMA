"""Tests for the RPA reduction of Gibbs states."""

from functools import partial

import numpy as np
import pytest
import qutip

from qalma import settings
from qalma.meanfield import variational_quadratic_mfa
from qalma.meanfield.rpa import (
    RPAGibbsDensityOperator,
    build_rpa_structure,
    effective_bath_modes,
    kubo_mori_susceptibility,
    local_boson_modes,
    rpa_coupling_kernel,
    rpa_dynamic_kernel,
)
from qalma.meanfield.rpa.environment import environment_modes, static_kernel
from qalma.meanfield.rpa.rpa_structure import _split_sites
from qalma.model import build_spin_chain, build_system
from qalma.operators.states.gibbs import GibbsDensityOperator
from qalma.projections.symmetries import project_conserved_quantity

PAULI = {
    "0": qutip.qeye(2),
    "x": qutip.sigmax(),
    "y": qutip.sigmay(),
    "z": qutip.sigmaz(),
}


def _two_site_coefficient(operator: qutip.Qobj, key: str) -> float:
    """Coefficient of the Pauli string ``key`` in a two-site operator."""
    basis = qutip.tensor(PAULI[key[0]], PAULI[key[1]])
    return float(np.real((basis * operator).tr() / 4))


def _setup(system, beta):
    """RPA Gibbs state of the Hamiltonian, with a fixed mean-field state."""
    hamiltonian = system.global_operator("Hamiltonian")
    k_op = (beta * hamiltonian).simplify()
    sigma_mf = variational_quadratic_mfa(k_op.flat())
    state = RPAGibbsDensityOperator(k_op, meanfield=sigma_mf)
    return state, sigma_mf


def _reduced_generator(state, sites, rpa=False, dynamic=False):
    """Reduced generator of ``state`` over ``sites``, as a qutip operator."""
    generator = state.reduced_generator(sites, rpa=rpa, dynamic=dynamic)
    return generator.to_qutip(tuple(sorted(sites)))


def _normalized(rho):
    """Normalize a qutip density matrix."""
    return rho / rho.tr()


def _gibbs_matrix(state, sites, **kwargs):
    """Normalized ``exp(-K_A')`` of the approximate reduced generator."""
    return _normalized((-_reduced_generator(state, sites, **kwargs)).expm())


# ### Reference implementation based on explicit Matsubara sums ###


def _local_chi0_matsubara(sigma_loc, basis, n_values):
    """Local ``int_0^1 e^{i w_n tau} <de_a(tau) de_b>`` for ``w_n = 2 pi n``."""
    p, vecs = np.linalg.eigh(sigma_loc)
    p = np.clip(p, 1e-300, None)
    eps = -np.log(p)
    ops = np.array([vecs.conj().T @ e @ vecs for e in basis])
    delta = eps[:, None] - eps[None, :]
    num = p[None, :] - p[:, None]
    prod = np.einsum("amn,bnm->abmn", ops, ops)
    avg = np.array([np.sum(p * np.diag(o)).real for o in ops])
    result = []
    for n in n_values:
        den = delta + 2j * np.pi * n
        if n == 0:
            small = np.abs(delta) < 1e-12
            w = np.where(
                small, p[:, None] * np.ones_like(delta), num / np.where(small, 1, den)
            )
        else:
            w = num / den
        chi = np.einsum("mn,abmn->ab", w, prod).astype(complex)
        result.append(chi - np.outer(avg, avg) if n == 0 else chi)
    return np.array(result)


def _local_chi0_tau(sigma_loc, basis, taus):
    """Local imaginary-time correlator ``<de_a(tau) de_b>``."""
    p, vecs = np.linalg.eigh(sigma_loc)
    p = np.clip(p, 1e-300, None)
    ops = np.array([vecs.conj().T @ e @ vecs for e in basis])
    avg = np.array([np.sum(p * np.diag(o)).real for o in ops])
    weights = p[None, :, None] ** (1.0 - taus[:, None, None]) * (
        p[None, None, :] ** taus[:, None, None]
    )
    prod = np.einsum("amn,bnm->abmn", ops, ops)
    return np.einsum("tmn,abmn->tab", weights, prod) - np.outer(avg, avg)


def _block_diag(blocks):
    n_freq, size = blocks[0].shape[0], sum(b.shape[1] for b in blocks)
    result = np.zeros((n_freq, size, size), dtype=complex)
    pos = 0
    for block in blocks:
        k = block.shape[1]
        result[:, pos : pos + k, pos : pos + k] = block
        pos += k
    return result


def _matsubara_kernel(generator, sites, sigma_mf, nodes, n_max=200):
    """Sigma(tau): bare part exact, dressing summed over |n| <= n_max."""
    structure = build_rpa_structure(generator, sigma_mf)
    m_full = structure.m_full
    idx_a, idx_e, _, sites_e = _split_sites(structure, sites)
    ops = {}
    for site in sites_e:
        populations, vectors = structure.local_spectrum(site)
        ops[site] = (
            (vectors * populations) @ vectors.conj().T,
            structure.local_data(site)[1],
        )
    m_ae = m_full[np.ix_(idx_a, idx_e)]
    m_ee = m_full[np.ix_(idx_e, idx_e)]
    chi0_tau = _block_diag([_local_chi0_tau(*ops[s], nodes) for s in sites_e])
    sigma = np.einsum("ie,tef,jf->tij", m_ae, chi0_tau, m_ae)
    n_values = list(range(-n_max, n_max + 1))
    chi0_w = _block_diag([_local_chi0_matsubara(*ops[s], n_values) for s in sites_e])
    phases = np.exp(-2j * np.pi * np.outer(n_values, nodes))
    eye = np.eye(len(idx_e))
    for pos, chi0 in enumerate(chi0_w):
        chi_e = np.linalg.solve(eye + chi0 @ m_ee, chi0)
        sigma += phases[pos][:, None, None] * (m_ae @ (chi_e - chi0) @ m_ae.T)[None]
    return sigma


# ### Tests ###


def test_kubo_mori_commuting_limit():
    """For a diagonal state and a diagonal Q, chi is the classical variance."""
    p = np.array([0.7, 0.3])
    sz = np.diag([1.0, -1.0])
    chi = kubo_mori_susceptibility(np.diag(p), [sz])
    assert abs(chi[0, 0] - (1.0 - (p[0] - p[1]) ** 2)) < 1e-12


def test_boson_modes_reproduce_local_correlator():
    """The linear boson truncation plus static modes gives the exact chi0(tau)."""
    sigma = np.array([[0.6, 0.1], [0.1, 0.4]], dtype=complex)
    basis = [qutip.sigmax().full(), qutip.sigmay().full(), qutip.sigmaz().full()]
    freqs, g_mat = local_boson_modes(sigma, basis)
    taus = np.linspace(0.0, 1.0, 7)
    # <(g a + g* a^dag)(tau) (g a + g* a^dag)> for a thermal boson
    occ = 1.0 / np.expm1(freqs)
    corr = np.einsum(
        "ak,bk,tk->tab",
        g_mat,
        g_mat.conj(),
        (1 + occ)[None] * np.exp(-np.outer(taus, freqs)),
    ) + np.einsum(
        "ak,bk,tk->tab", g_mat.conj(), g_mat, occ[None] * np.exp(np.outer(taus, freqs))
    )
    assert np.allclose(corr, _local_chi0_tau(sigma, basis, taus), atol=1e-5)


def test_dynamic_kernel_matches_matsubara_sum():
    """Bosonization is the analytic sum over all Matsubara frequencies."""
    system = build_spin_chain(10, field=0.3)
    sites = list(system.sites)
    sites_a = frozenset([sites[4], sites[6]])
    state, sigma_mf = _setup(system, 1.0)
    generator = state.k.flat().simplify()
    structure = build_rpa_structure(generator, sigma_mf)
    sigma_tau, nodes, _, _, _ = rpa_dynamic_kernel(structure, sites_a, 16)
    reference = _matsubara_kernel(generator, sites_a, sigma_mf, nodes)
    assert np.max(np.abs(sigma_tau - reference)) < 1e-5 * np.max(np.abs(reference))


def test_dynamic_kernel_static_sum_rule():
    """The time integral of Sigma(tau) is the static RPA kernel."""
    system = build_spin_chain(10, field=0.3)
    sites = list(system.sites)
    sites_a = frozenset([sites[4], sites[6]])
    state, sigma_mf = _setup(system, 1.0)
    generator = state.k.flat().simplify()
    structure = build_rpa_structure(generator, sigma_mf)
    sigma_tau, _, weights, _, _ = rpa_dynamic_kernel(structure, sites_a, 32)
    static, _, _ = rpa_coupling_kernel(structure, sites_a)
    integral = np.einsum("t,tij->ij", weights, sigma_tau)
    assert np.allclose(integral, static, atol=1e-5 * np.max(np.abs(static)))


def test_rpa_improves_second_neighbors():
    """The environment-induced coupling between second neighbors."""
    system = build_spin_chain(10, field=0.3)
    sites = list(system.sites)
    sites_a = frozenset([sites[4], sites[6]])
    # block = tuple(sorted(sites_a))
    state, sigma_mf = _setup(system, 0.5)
    rho_exact = state.to_qutip_operator().partial_trace(sites_a).to_qutip()

    def trace_distance(**kwargs):
        k_a = _reduced_generator(state, sites_a, **kwargs)
        rho = (-k_a).expm()
        rho = rho / rho.tr()
        return 0.5 * (rho - rho_exact).norm("tr")

    mean_field = trace_distance()
    assert trace_distance(rpa=True) < 0.5 * mean_field
    assert trace_distance(rpa=True, dynamic=True) < 0.5 * mean_field


def test_dynamic_equals_static_for_commuting_generator():
    """For a classical Ising generator the dynamic kernel is time independent."""
    system = build_system("chain lattice", "spin", L=10, J=1, Jz=1, Jxy=0, h=0.3)
    sites = list(system.sites)
    sites_a = frozenset([sites[4], sites[6]])
    # block = tuple(sorted(sites_a))
    state, sigma_mf = _setup(system, 0.8)
    k_static = _reduced_generator(state, sites_a, rpa=True)
    k_dynamic = _reduced_generator(state, sites_a, rpa=True, dynamic=True)
    assert np.linalg.norm((k_static - k_dynamic).full()) < 1e-6


def _lipkin_state(field, temperature, names=("Sx", "Sy"), n_sites=6, **kwargs):
    """Fully connected (Lipkin) spin-1/2 model with couplings scaled as 1/n.

    ``H = b S_z - (v/n) sum_{i != j} sum_a s^a_i s^a_j`` with ``v = 1`` and
    ``a`` in ``names``. Returns the RPA state, its pair of sites and the
    exact reduced state of the pair.
    """
    system = build_system("chain lattice", "spin", L=n_sites, J=1, local_S=0.5)
    sites = list(system.sites)

    def op(name, site):
        return system.site_operator(f"{name}@{site}")

    hamiltonian = field * sum(op("Sz", site) for site in sites)
    for i, site_i in enumerate(sites):
        for site_j in sites[i + 1 :]:
            for name in names:
                hamiltonian = hamiltonian - (2.0 / n_sites) * (
                    op(name, site_i) * op(name, site_j)
                )
    k_op = (hamiltonian / temperature).simplify()
    state = RPAGibbsDensityOperator(
        k_op, system, meanfield=variational_quadratic_mfa(k_op.flat()), **kwargs
    )
    pair = frozenset(sites[:2])
    exact = state.to_qutip_operator().partial_trace(pair).to_qutip(tuple(sorted(pair)))
    return state, pair, exact / exact.tr()


def _pair_distance(state, pair, exact, **options):
    """Trace distance between the approximate and exact states of ``pair``."""
    rho = state.partial_trace(pair, max_exact_sites=0, **options)
    rho = rho.to_qutip(tuple(sorted(pair)))
    return 0.5 * (rho / rho.tr() - exact).norm("tr")


def test_dynamic_rpa_at_low_temperature():
    """In the normal phase of the x-coupled Lipkin model the RPA correlations
    do not vanish at T = 0. The dynamical correction stays finite, and close
    to the exact state, down to the ground state."""
    for temperature in (0.05, 0.005):
        state, pair, exact = _lipkin_state(1.5, temperature, names=("Sx",))
        mean_field = _pair_distance(state, pair, exact, rpa=False)
        dynamic = _pair_distance(state, pair, exact, dynamic=True)
        assert np.isfinite(dynamic)
        assert dynamic < 1e-3
        assert dynamic < 0.1 * mean_field


def test_dynamic_rpa_vanishes_in_the_saturated_xx_phase():
    """In the normal phase of the XX Lipkin model the ground state is the
    saturated mean-field state, and the RPA corrections vanish at T -> 0."""
    state, pair, exact = _lipkin_state(1.5, 0.005)
    # The residual (~1e-8) is the precision of the mean-field solver, whose
    # random initial point changes from run to run.
    assert _pair_distance(state, pair, exact, dynamic=True) < 1e-6


def test_dynamic_rpa_at_high_temperature():
    """At high temperature the dynamical correction is the best reduction."""
    for field in (0.5, 1.5):
        state, pair, exact = _lipkin_state(field, 1.0)
        mean_field = _pair_distance(state, pair, exact, rpa=False)
        static = _pair_distance(state, pair, exact)
        dynamic = _pair_distance(state, pair, exact, dynamic=True)
        assert dynamic < 0.25 * static
        assert static < 0.6 * mean_field


def test_dynamic_rpa_quadrature_is_adaptive():
    """At low temperature the quadrature is refined from any starting size."""
    results = []
    for n_tau in (4, 128):
        state, pair, exact = _lipkin_state(1.5, 0.002, names=("Sx",), n_tau=n_tau)
        rho = state.partial_trace(pair, max_exact_sites=0, dynamic=True)
        results.append(rho.to_qutip(tuple(sorted(pair))))
    assert (results[0] - results[1]).norm("tr") < 1e-4


def _lipkin_operator(system, field, couplings):
    """Generator-independent Hamiltonian of the fully connected model."""
    sites = list(system.sites)

    def op(name, site):
        return system.site_operator(f"{name}@{site}")

    n_sites = len(sites)
    hamiltonian = field * sum(op("Sz", site) for site in sites)
    for i, site_i in enumerate(sites):
        for site_j in sites[i + 1 :]:
            for name, value in couplings.items():
                hamiltonian = hamiltonian - (2.0 * value / n_sites) * (
                    op(name, site_i) * op(name, site_j)
                )
    return hamiltonian.simplify()


def _rpa_log_partition(k_op):
    """``ln Z`` of the mean field plus RPA (no static fluctuations)."""
    k_flat = k_op.flat()
    sigma = variational_quadratic_mfa(k_flat)
    structure = build_rpa_structure(k_flat, sigma)
    entropy = 0.0
    local = []
    for site in structure.positions:
        populations, _ = structure.local_spectrum(site)
        populations = populations[populations > 1e-300]
        entropy -= np.sum(populations * np.log(populations))
        freqs = structure.boson_modes(site)[0]
        local.extend(freqs[freqs > structure.static_gap * (1 + 1e-6)])
    bosons = structure.boson_system()
    omega = bosons.lam[bosons.lam > 0]
    return (
        entropy
        - np.real(sigma.expect(k_flat))
        + np.sum(np.log(np.sinh(np.array(local) / 2)))
        - np.sum(np.log(np.sinh(omega / 2)))
    )


def test_bosonized_state_reproduces_the_rpa_partition_function():
    """The bosonized pair state gives the correlations obtained deriving the
    RPA partition function (mean field plus RPA) with respect to the
    couplings (thesis of J. M. Matera, chapter 6)."""
    system = build_system("chain lattice", "spin", L=6, J=1, local_S=0.5)
    sites = list(system.sites)
    k_op = (_lipkin_operator(system, 1.5, {"Sx": 1.0, "Sy": 0.5}) / 0.05).simplify()
    state = RPAGibbsDensityOperator(k_op, system)
    pair = (sites[0], sites[1])
    rho = state.bosonized_state(frozenset(pair)).to_qutip(pair)
    step = 1e-4
    for name in ("Sx", "Sy", "Sz"):
        local = [system.site_operator(f"{name}@{site}") for site in pair]
        product = (local[0] * local[1]).simplify()
        for perturbation, observable in ((product, product), (local[0], local[0])):
            expected = -(
                _rpa_log_partition((k_op + step * perturbation).simplify())
                - _rpa_log_partition((k_op - step * perturbation).simplify())
            ) / (2 * step)
            value = (rho * observable.to_qutip(pair)).tr().real
            if name == "Sz" and observable is product:
                # the Gaussian state adds O(1/n^2) terms to <sz sz>
                assert abs(value - expected) < 5e-3
            else:
                assert abs(value - expected) < 1e-6


def test_bosonized_state_improves_mean_field():
    """Bosonized states of pairs of spin 1/2 and spin 1 in the normal phase."""
    for local_spin, field, temperature in ((0.5, 1.5, 0.05), (1.0, 4.0, 0.5)):
        system = build_system("chain lattice", "spin", L=5, J=1, local_S=local_spin)
        sites = list(system.sites)
        k_op = (_lipkin_operator(system, field, {"Sx": 1.0}) / temperature).simplify()
        state = RPAGibbsDensityOperator(k_op, system)
        pair = frozenset(sites[:2])
        block = tuple(sorted(pair))
        exact = state.to_qutip_operator().partial_trace(pair).to_qutip(block)
        exact = exact / exact.tr()
        rho = state.bosonized_state(pair).to_qutip(block)
        mean_field = state.partial_trace(pair, max_exact_sites=0, rpa=False)
        mean_field = mean_field.to_qutip(block)
        mean_field = mean_field / mean_field.tr()
        assert abs(rho.tr() - 1) < 1e-12
        assert (rho - rho.dag()).norm() < 1e-12
        assert np.linalg.eigvalsh(rho.full()).min() > -2e-3
        assert (rho - exact).norm("tr") < 0.6 * (mean_field - exact).norm("tr")
        # the state of one site is the marginal of the pair
        single = state.bosonized_state(frozenset([block[0]])).to_qutip((block[0],))
        assert (single - rho.ptrace(0)).norm() < 1e-12


def test_bosonized_state_limits():
    """More than two sites, and zero modes, are rejected."""
    state, site_a, site_b = _pair_state(0.5, 0.5, 40.0, jy=1.0)
    with pytest.raises(ValueError, match="zero mode"):
        state.bosonized_state(frozenset([site_a, site_b]))
    system = build_system("chain lattice", "spin", L=4, J=1, local_S=0.5)
    k_op = (_lipkin_operator(system, 1.5, {"Sx": 1.0}) / 0.5).simplify()
    state = RPAGibbsDensityOperator(k_op, system)
    with pytest.raises(NotImplementedError):
        state.bosonized_state(frozenset(list(system.sites)[:3]))


def test_static_kernel_of_the_isolated_environment():
    """The static kernel of the normal modes is the RPA susceptibility."""
    system = build_system("chain lattice", "spin", L=6, J=1, local_S=0.5)
    sites = list(system.sites)
    for temperature in (0.05, 0.5):
        k_op = (
            _lipkin_operator(system, 1.5, {"Sx": 1.0, "Sy": 0.5}) / temperature
        ).simplify()
        structure = RPAGibbsDensityOperator(k_op, system).rpa_structure
        pair = frozenset(sites[:2])
        expected = rpa_coupling_kernel(structure, pair)[0]
        isolated = static_kernel(environment_modes(structure, pair, "isolated"))
        assert np.abs(isolated - expected).max() < 1e-10 * np.abs(expected).max()
        # the global modes include the feedback of the subsystem
        global_ = static_kernel(environment_modes(structure, pair, "global"))
        assert np.abs(global_ - expected).max() > 1e-3 * np.abs(expected).max()


def test_global_environment():
    """Correlations of the environment from the bosonization of the whole
    system (thesis of J. M. Matera, chapter 6): the dynamical reduction stays
    close to the exact one, and the global modes are computed once."""
    for temperature in (0.05, 1.0):
        state, pair, exact = _lipkin_state(
            1.5, temperature, names=("Sx",), environment="global"
        )
        mean_field = _pair_distance(state, pair, exact, rpa=False)
        dynamic = _pair_distance(state, pair, exact, dynamic=True)
        assert dynamic < 0.2 * mean_field
        isolated = _pair_distance(
            state, pair, exact, dynamic=True, environment="isolated"
        )
        assert isolated < dynamic
    structure = state.rpa_structure
    modes = structure._global_modes  # pylint: disable=protected-access
    assert modes is not None
    other = frozenset(list(state.system.sites)[2:4])
    state.partial_trace(other, max_exact_sites=0, dynamic=True)
    assert structure._global_modes is modes  # pylint: disable=protected-access


def test_environment_options_are_checked():
    """Unknown environments, and the hybrid reduction with the global
    environment, are rejected."""
    state, pair, _ = _lipkin_state(1.5, 0.5, names=("Sx",))
    with pytest.raises(ValueError, match="environment"):
        RPAGibbsDensityOperator(state.k, state.system, environment="bath")
    with pytest.raises(ValueError, match="isolated"):
        state.partial_trace(pair, max_exact_sites=0, hybrid=True, environment="global")


def test_dynamic_corrects_adjacent_bond():
    """The dynamical term renormalizes the coupling between adjacent sites."""
    system = build_spin_chain(10, field=0.3)
    sites = list(system.sites)
    sites_a = frozenset(sites[4:6])
    # block = tuple(sorted(sites_a))
    state, sigma_mf = _setup(system, 0.5)
    k_exact = -state.to_qutip_operator().partial_trace(sites_a).to_qutip().logm()
    k_static = _reduced_generator(state, sites_a, rpa=True)
    k_dynamic = _reduced_generator(state, sites_a, rpa=True, dynamic=True)
    exact = _two_site_coefficient(k_exact, "xx")
    error_static = abs(_two_site_coefficient(k_static, "xx") - exact)
    error_dynamic = abs(_two_site_coefficient(k_dynamic, "xx") - exact)
    assert error_dynamic < 0.2 * error_static


def test_partial_trace_rpa_option():
    """``partial_trace(..., rpa=True)`` improves the mean-field reduced state."""
    system = build_spin_chain(10, field=0.3)
    sites = list(system.sites)
    sites_a = frozenset([sites[4], sites[6]])
    block = tuple(sorted(sites_a))
    state, _ = _setup(system, 0.5)
    exact = state.to_qutip_operator().partial_trace(sites_a).to_qutip(block)

    def trace_distance(**kwargs):
        rho = state.partial_trace(sites_a, **kwargs).to_qutip(block)
        return 0.5 * (rho / rho.tr() - exact).norm("tr")

    mean_field = trace_distance(rpa=False)
    assert trace_distance(rpa=True) < 0.5 * mean_field
    assert trace_distance(rpa=True, dynamic=True) < 0.5 * mean_field
    # ``reduce`` forwards the options
    rho_reduce = state.reduce(sites_a, dynamic=True).to_qutip(block)
    rho_trace = state.partial_trace(sites_a, dynamic=True).to_qutip(block)
    assert (rho_reduce - rho_trace).norm("tr") < 1e-10


# ### RPAGibbsDensityOperator ###


def _rpa_state(system, beta, **kwargs):
    """``RPAGibbsDensityOperator`` and plain Gibbs state sharing the mean field."""
    hamiltonian = system.global_operator("Hamiltonian")
    k_op = (beta * hamiltonian).simplify()
    sigma_mf = variational_quadratic_mfa(k_op.flat())
    rpa_state = RPAGibbsDensityOperator(k_op, meanfield=sigma_mf, **kwargs)
    plain_state = GibbsDensityOperator(k_op, meanfield=sigma_mf)
    return rpa_state, plain_state


def test_rpa_state_partial_trace_uses_rpa_by_default():
    """Its reductions are the Gibbs states of the RPA-corrected generator."""
    system = build_spin_chain(10, field=0.3)
    sites = list(system.sites)
    sites_a = frozenset([sites[4], sites[6]])
    block = tuple(sorted(sites_a))
    rpa_state, plain_state = _rpa_state(system, 0.5)
    reduced = _normalized(rpa_state.partial_trace(sites_a).to_qutip(block))
    expected_rpa = _gibbs_matrix(rpa_state, sites_a, rpa=True)
    mean_field = _normalized(plain_state.partial_trace(sites_a).to_qutip(block))
    assert (reduced - expected_rpa).norm("tr") < 1e-10
    assert (reduced - mean_field).norm("tr") > 1e-4
    # the options can be overridden
    no_rpa = _normalized(rpa_state.partial_trace(sites_a, rpa=False).to_qutip(block))
    assert (no_rpa - mean_field).norm("tr") < 1e-10
    dynamic = _normalized(
        rpa_state.partial_trace(sites_a, dynamic=True).to_qutip(block)
    )
    expected_dyn = _gibbs_matrix(rpa_state, sites_a, rpa=True, dynamic=True)
    assert (dynamic - expected_dyn).norm("tr") < 1e-10


def test_rpa_state_dynamic_by_default():
    """The ``dynamic`` option given at construction is the default."""
    system = build_spin_chain(10, field=0.3)
    sites = list(system.sites)
    sites_a = frozenset(sites[4:6])
    block = tuple(sorted(sites_a))
    rpa_state, _ = _rpa_state(system, 0.5, dynamic=True)
    reduced = _normalized(rpa_state.partial_trace(sites_a).to_qutip(block))
    expected = _gibbs_matrix(rpa_state, sites_a, rpa=True, dynamic=True)
    assert (reduced - expected).norm("tr") < 1e-10
    assert (
        _reduced_generator(rpa_state, sites_a, rpa=True, dynamic=True)
        - rpa_state.reduced_generator(sites_a).to_qutip(block)
    ).norm("tr") < 1e-10


def test_rpa_state_stores_structures():
    """The subsystem-independent data are computed once and reused."""
    system = build_spin_chain(10, field=0.3)
    sites = list(system.sites)
    rpa_state, _ = _rpa_state(system, 0.5)
    assert rpa_state._rpa_structure is None  # pylint: disable=protected-access
    rpa_state.partial_trace(frozenset([sites[4], sites[6]]))
    structure = rpa_state.rpa_structure
    assert structure.sigma_mf is rpa_state.meanfield
    rpa_state.partial_trace(frozenset(sites[2:4]))
    assert rpa_state.rpa_structure is structure
    assert len(structure._chi0) > 0  # pylint: disable=protected-access
    # the stored single-site susceptibilities are not recomputed
    site = next(iter(structure._chi0))  # pylint: disable=protected-access
    chi0 = structure.static_chi0(site)
    assert structure.static_chi0(site) is chi0


def test_rpa_state_arithmetic():
    """Arithmetic works, and products by scalars preserve the type."""
    system = build_spin_chain(10, field=0.3)
    sites = list(system.sites)
    rpa_state, _ = _rpa_state(system, 0.5)
    rpa_state.partial_trace(frozenset(sites[4:6]))  # builds the structure
    for scaled in (2 * rpa_state, rpa_state * 2.0, rpa_state / 2.0):
        assert isinstance(scaled, RPAGibbsDensityOperator)
        # pylint: disable=protected-access
        assert scaled._rpa_structure is rpa_state._rpa_structure
    assert abs((2 * rpa_state).prefactor - 2 * rpa_state.prefactor) < 1e-12
    assert abs((rpa_state / 2).prefactor - rpa_state.prefactor / 2) < 1e-12
    # other operations give the standard types, but work
    assert rpa_state + rpa_state is not None
    hamiltonian = system.global_operator("Hamiltonian")
    assert rpa_state + hamiltonian is not None
    # expectation values are normalized
    energy = rpa_state.expect(hamiltonian)
    assert abs((2 * rpa_state).expect(hamiltonian) - energy) < 1e-10
    assert abs((rpa_state / 2).expect(hamiltonian) - energy) < 1e-10


def test_rpa_state_dynamic_improves_energy():
    """Expectation values use the RPA reductions: the energy improves."""
    system = build_spin_chain(10, field=0.3)
    hamiltonian = system.global_operator("Hamiltonian")
    rpa_state, plain_state = _rpa_state(system, 0.5, dynamic=True)
    exact = plain_state.to_qutip_operator().expect(hamiltonian)
    error_mf = abs(plain_state.expect(hamiltonian) - exact)
    error_rpa = abs(rpa_state.expect(hamiltonian) - exact)
    assert error_rpa < 0.5 * error_mf


def test_max_exact_sites_option(monkeypatch):
    """Small reductions are exact unless ``max_exact_sites`` excludes them."""
    system = build_spin_chain(4, field=0.3)
    sites = list(system.sites)
    sites_a = frozenset(sites[:2])
    block = tuple(sorted(sites_a))
    rpa_state, plain_state = _rpa_state(system, 0.5)
    exact = plain_state.to_qutip_operator().partial_trace(sites_a).to_qutip(block)
    for state in (rpa_state, plain_state):
        reduced = state.partial_trace(sites_a).to_qutip(block)
        assert (reduced - exact).norm("tr") < 1e-10
        reduced = state.partial_trace(sites_a, max_exact_sites=4).to_qutip(block)
        assert (reduced - exact).norm("tr") < 1e-10
    approx = rpa_state.partial_trace(sites_a, max_exact_sites=3).to_qutip(block)
    mean_field = plain_state.partial_trace(sites_a, max_exact_sites=3).to_qutip(block)
    assert (approx - exact).norm("tr") > 1e-6
    assert (mean_field - exact).norm("tr") > 1e-6
    assert (approx - mean_field).norm("tr") > 1e-6
    # the default is read from the settings when the reduction is computed
    monkeypatch.setattr(settings, "MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE", 0)
    default = plain_state.partial_trace(sites_a).to_qutip(block)
    assert (default - mean_field).norm("tr") < 1e-10
    with pytest.raises(ValueError):
        plain_state.partial_trace(sites_a, max_exact_sites=-1)


def test_gibbs_state_has_no_rpa_options():
    """The plain Gibbs state only provides the mean-field reduction."""
    system = build_spin_chain(4, field=0.3)
    _, plain_state = _rpa_state(system, 0.5)
    with pytest.raises(TypeError):
        plain_state.partial_trace(frozenset(list(system.sites)[:2]), rpa=True)


# ### Full bosonization: Gaussian state of the whole system ###


def _pair_state(spin, field, beta, jx=1.0, jy=0.5):
    """Pair of spins ``spin`` of the XY model of Matera et al., PRA 82, 052332.

    The equations cited in this section are numbered as in the preprint,
    arXiv:1009.1750v2.
    """
    system = build_system("chain lattice", "spin", L=2, J=1, local_S=spin)
    site_a, site_b = list(system.sites)

    def op(name, site):
        return system.site_operator(f"{name}@{site}")

    hamiltonian = field * (op("Sz", site_a) + op("Sz", site_b))
    for name, coupling in (("Sx", jx), ("Sy", jy)):
        if coupling:
            hamiltonian = hamiltonian - (1.0 / spin) * coupling * (
                op(name, site_a) * op(name, site_b)
            )
    k_op = (beta * hamiltonian).simplify()
    sigma_mf = variational_quadratic_mfa(k_op.flat())
    state = RPAGibbsDensityOperator(k_op, system, meanfield=sigma_mf)
    return state, site_a, site_b


def _paper_occupation(field, jx=1.0, jy=0.5):
    """Average occupation ``f`` of the RPA of the pair, Eqs. (46), (51) and (62)."""
    critical = jx
    if field > critical:
        lam, jx_eff = field, jx
    else:
        lam, jx_eff = jx, jx * (field / critical) ** 2
    w_0 = np.sqrt((lam - jx_eff) * (lam - jy))
    w_1 = np.sqrt((lam + jx_eff) * (lam + jy))
    omega = 0.5 * (w_0 + w_1)
    return 0.5 * (np.sqrt(1 + (lam**2 - omega**2) / (w_0 * w_1)) - 1)


def test_local_spectrum_keeps_tiny_populations():
    """The local populations are not limited by the precision of a density matrix."""
    state, site_a, _ = _pair_state(0.5, 1.2, 60.0)
    populations, _ = state.rpa_structure.local_spectrum(site_a)
    ratio = populations.min() / populations.max()
    assert abs(np.log(ratio) + 60.0 * 1.2) < 1e-6


def test_gaussian_rpa_pair_matches_paper():
    """Entropy and negativity of the pair coincide with Eqs. (61)-(63) of the paper."""
    for spin in (0.5, 2.0):
        for field in (0.6, 1.2, 1.5):
            state, site_a, site_b = _pair_state(spin, field, 60.0)
            f = _paper_occupation(field)
            entropy = (1 + f) * np.log2(1 + f) - f * np.log2(f)
            negativity = f + np.sqrt(f * (f + 1))
            assert abs(state.rpa_entropy(frozenset([site_a])) - entropy) < 1e-4
            assert (
                abs(
                    state.rpa_negativity(frozenset([site_a]), frozenset([site_b]))
                    - negativity
                )
                < 1e-4
            )


def test_gaussian_rpa_uncoupled_spins_are_not_entangled():
    """Without coupling, the bosonic vacuum is a product state."""
    state, site_a, site_b = _pair_state(0.5, 1.2, 40.0, jx=0.0, jy=0.0)
    assert abs(state.rpa_entropy(frozenset([site_a]))) < 1e-6
    assert abs(state.rpa_negativity(frozenset([site_a]), frozenset([site_b]))) < 1e-8


# ### Hybrid reduction: the subsystem exact, RPA on the environment ###


def test_effective_bath_modes_reproduce_spectral_moments():
    """The Lanczos modes reproduce the first moments of the spectral density."""
    rng = np.random.default_rng(1234)
    omega = rng.uniform(0.5, 3.0, 25)
    coupling = rng.normal(size=(2, 25)) + 1j * rng.normal(size=(2, 25))

    def moment(w, c, k):
        return (c * w[None, :] ** k) @ c.conj().T

    for n_blocks in (1, 2, 3):
        w_eff, c_eff = effective_bath_modes(omega, coupling, n_blocks)
        assert len(w_eff) == 2 * n_blocks
        for k in range(2 * n_blocks):
            assert np.allclose(
                moment(w_eff, c_eff, k), moment(omega, coupling, k), atol=1e-9
            ), (n_blocks, k)
    # if the Krylov space is the whole bath, the compression is exact
    w_eff, c_eff = effective_bath_modes(omega, coupling, 30)
    for k in range(6):
        assert np.allclose(moment(w_eff, c_eff, k), moment(omega, coupling, k))


def _exact_pair_entropy(spin, field, beta, jx=1.0, jy=0.5):
    """Entropy of the reduced state of the Gibbs state of the pair."""
    system = build_system("chain lattice", "spin", L=2, J=1, local_S=spin)
    site_a, site_b = list(system.sites)

    def op(name, site):
        return system.site_operator(f"{name}@{site}")

    hamiltonian = field * (op("Sz", site_a) + op("Sz", site_b))
    for name, coupling in (("Sx", jx), ("Sy", jy)):
        hamiltonian = hamiltonian - (1.0 / spin) * coupling * (
            op(name, site_a) * op(name, site_b)
        )
    rho = (-beta * hamiltonian.to_qutip((site_a, site_b))).expm()
    rho = (rho / rho.tr()).ptrace(0)
    return _entropy_bits(rho.full())


def _entropy_bits(rho):
    p = np.clip(np.linalg.eigvalsh(rho), 0, None)
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log2(p)))


def test_hybrid_pair_in_normal_phase():
    """In the normal phase the hybrid reduction captures the entanglement."""
    for spin in (0.5, 2.0):
        state, site_a, _ = _pair_state(spin, 1.5, 60.0)
        exact = _exact_pair_entropy(spin, 1.5, 60.0)
        options = {"n_fock": 8}
        rho = state.partial_trace(
            frozenset([site_a]), max_exact_sites=0, hybrid=True, hybrid_options=options
        ).to_qutip((site_a,))
        rho_mf = state.partial_trace(
            frozenset([site_a]), max_exact_sites=0, rpa=False
        ).to_qutip((site_a,))
        entropy = _entropy_bits((rho / rho.tr()).full())
        assert _entropy_bits((rho_mf / rho_mf.tr()).full()) < 1e-6
        assert abs(entropy - exact) < 0.3 * exact
        # the result converges with the Fock truncation
        rho_big = state.partial_trace(
            frozenset([site_a]),
            max_exact_sites=0,
            hybrid=True,
            hybrid_options={"n_fock": 12},
        ).to_qutip((site_a,))
        assert (rho / rho.tr() - rho_big / rho_big.tr()).norm("tr") < 1e-6


def test_hybrid_uncoupled_spins_give_mean_field():
    """Without coupling, the hybrid reduction is the mean-field state."""
    state, site_a, _ = _pair_state(0.5, 1.2, 10.0, jx=0.0, jy=0.0)
    hybrid = state.partial_trace(
        frozenset([site_a]), max_exact_sites=0, hybrid=True
    ).to_qutip((site_a,))
    mean_field = state.partial_trace(
        frozenset([site_a]), max_exact_sites=0, rpa=False
    ).to_qutip((site_a,))
    assert (hybrid / hybrid.tr() - mean_field / mean_field.tr()).norm("tr") < 1e-8


def test_hybrid_chain_with_classical_modes():
    """At moderate temperature the classical static fields are included."""
    system = build_system(
        "chain lattice", "spin", L=6, J=1, Jz0=1, Jxy0=1, h=0.3, local_S=0.5
    )
    sites = list(system.sites)
    sites_a = frozenset([sites[2]])
    block = tuple(sites_a)
    hamiltonian = system.global_operator("Hamiltonian")
    k_op = (0.6 * hamiltonian).simplify()
    sigma_mf = variational_quadratic_mfa(k_op.flat())
    plain = GibbsDensityOperator(k_op, meanfield=sigma_mf)
    state = RPAGibbsDensityOperator(k_op, meanfield=sigma_mf)
    exact = plain.to_qutip_operator().partial_trace(sites_a).to_qutip(block)
    exact = exact / exact.tr()

    def distance(rho):
        return 0.5 * (rho / rho.tr() - exact).norm("tr")

    mean_field = distance(
        plain.partial_trace(sites_a, max_exact_sites=0).to_qutip(block)
    )
    hybrid = distance(
        state.partial_trace(
            sites_a,
            max_exact_sites=0,
            hybrid=True,
            hybrid_options={"n_fock": 4, "n_gauss": 5},
        ).to_qutip(block)
    )
    assert hybrid < 0.2 * mean_field


def test_hybrid_dimension_guard():
    """Too large hybrid problems are rejected."""
    state, site_a, _ = _pair_state(0.5, 1.5, 60.0)
    try:
        state.partial_trace(
            frozenset([site_a]),
            max_exact_sites=0,
            hybrid=True,
            hybrid_options={"n_fock": 100, "max_dim": 50},
        )
    except ValueError as error:
        assert "dimension" in str(error)
    else:
        raise AssertionError("A ValueError was expected")


def test_rpa_state_hybrid_by_default():
    """The ``hybrid`` option given at construction is the default."""
    system = build_spin_chain(10, field=0.3)
    sites = list(system.sites)
    sites_a = frozenset([sites[4]])
    block = tuple(sites_a)
    hamiltonian = system.global_operator("Hamiltonian")
    k_op = (0.5 * hamiltonian).simplify()
    sigma_mf = variational_quadratic_mfa(k_op.flat())
    options = {"n_fock": 4, "n_gauss": 5}
    state = RPAGibbsDensityOperator(
        k_op, meanfield=sigma_mf, hybrid=True, hybrid_options=options
    )
    plain = GibbsDensityOperator(k_op, meanfield=sigma_mf)
    reduced = state.partial_trace(sites_a).to_qutip(block)
    default = RPAGibbsDensityOperator(k_op, meanfield=sigma_mf)
    expected = default.partial_trace(
        sites_a, hybrid=True, hybrid_options=options
    ).to_qutip(block)
    assert (reduced - expected).norm("tr") < 1e-10
    mean_field = plain.partial_trace(sites_a).to_qutip(block)
    assert (reduced - mean_field).norm("tr") > 1e-6


# ### Zero modes of the RPA ###


def test_zero_modes_of_the_xxz_pair():
    """The XXZ pair has a Goldstone mode in the broken phase, and the Gaussian
    measures, which need the symmetry restoration, are rejected."""
    state, site_a, site_b = _pair_state(0.5, 0.5, 40.0, jy=1.0)
    assert state.rpa_zero_modes() == 1
    try:
        state.rpa_entropy(frozenset([site_a]))
    except ValueError as error:
        assert "zero mode" in str(error)
    else:
        raise AssertionError("A ValueError was expected")
    state, site_a, site_b = _pair_state(0.5, 1.5, 40.0, jy=1.0)
    assert state.rpa_zero_modes() == 0
    assert abs(state.rpa_entropy(frozenset([site_a]))) < 1e-6


def test_unphysical_gaussian_state_without_zero_modes():
    """Removing the zero mode leaves contractions that violate the uncertainty
    relations on a single site, which is why the Gaussian measures are rejected."""
    state, site_a, _ = _pair_state(0.5, 0.5, 40.0, jy=1.0)
    bosons = state.rpa_structure.boson_system()
    assert bosons.n_zero_modes == 1
    sigma = bosons.covariance_matrix(bosons.modes_of([site_a]))
    assert np.sqrt(np.linalg.det(sigma)) < 0.5 - 1e-3
    with pytest.raises(ValueError, match="zero mode"):
        bosons.entropy(bosons.modes_of([site_a]))
    with pytest.raises(ValueError, match="symmetry_projections"):
        state.rpa_negativity(frozenset([site_a]))


def test_hybrid_with_symmetry_projection_restores_the_xxz_pair():
    """With a broken U(1) symmetry, the entanglement follows from the hybrid
    reduced state with the symmetry projection (orientation integrated)."""
    state, site_a, _ = _pair_state(0.5, 0.5, 40.0, jy=1.0)
    projection = partial(project_conserved_quantity, op_name="Sz")
    state = RPAGibbsDensityOperator(
        state.k,
        state.system,
        meanfield=state.meanfield,
        symmetry_projections=[projection],
    )
    sites_a = frozenset([site_a])
    block = (site_a,)
    exact = state.to_qutip_operator().partial_trace(sites_a).to_qutip(block)
    exact = exact / exact.tr()

    def reduced(**options):
        rho = state.partial_trace(sites_a, max_exact_sites=0, **options)
        rho = rho.to_qutip(block)
        return rho / rho.tr()

    hybrid, mean_field = reduced(hybrid=True), reduced(rpa=False)
    assert abs(_entropy_bits(exact.full()) - 1.0) < 1e-6
    assert abs(_entropy_bits(hybrid.full()) - 1.0) < 0.1
    assert 0.5 * (hybrid - exact).norm("tr") < 0.6 * (
        0.5 * (mean_field - exact).norm("tr")
    )


def test_static_modes_are_not_zero_modes():
    """The static (classical) modes have a tiny frequency, but are not zero modes."""
    system = build_spin_chain(6, field=0.3)
    sites = list(system.sites)
    hamiltonian = system.global_operator("Hamiltonian")
    k_op = (1.0 * hamiltonian).simplify()
    state = RPAGibbsDensityOperator(
        k_op, meanfield=variational_quadratic_mfa(k_op.flat())
    )
    assert state.rpa_zero_modes(frozenset([sites[2]])) == 0


def _probe_state(n_env, field, beta, spin=0.5):
    """An XY magnet (environment, with a Goldstone mode) and a spin coupled by Sz Sz."""
    system = build_system("chain lattice", "spin", L=n_env + 1, J=1, local_S=spin)
    sites = list(system.sites)
    site_a, env = sites[0], sites[1:]

    def op(name, site):
        return system.site_operator(f"{name}@{site}")

    hamiltonian = field * sum(op("Sz", s) for s in sites) + 0.8 * op("Sx", site_a)
    for i in range(n_env):
        for j in range(i + 1, n_env):
            for name in ("Sx", "Sy"):
                hamiltonian = hamiltonian - (1.0 / (spin * (n_env - 1))) * (
                    op(name, env[i]) * op(name, env[j])
                )
    for site in env:
        hamiltonian = hamiltonian - (2.0 / (spin * n_env)) * (
            op("Sz", site_a) * op("Sz", site)
        )
    k_op = (beta * hamiltonian).simplify()
    sigma_mf = variational_quadratic_mfa(k_op.flat())
    state = RPAGibbsDensityOperator(k_op, system, meanfield=sigma_mf)
    return state, sites, hamiltonian


def test_hybrid_with_zero_modes_includes_the_charge_fluctuations():
    """The conserved charge of the zero mode correlates the subsystem with the
    environment: neglecting it gives the mean field, including it improves it."""
    beta = 60.0
    state, sites, hamiltonian = _probe_state(8, 0.3, beta)
    sites_a = frozenset([sites[0]])
    block = (sites[0],)
    assert state.rpa_zero_modes(sites_a) == 1
    everything = tuple(sorted(sites))
    exact = (-beta * hamiltonian.to_qutip(everything)).expm()
    exact = (exact / exact.tr()).ptrace([everything.index(sites[0])])

    def distance(**options):
        rho = state.partial_trace(
            sites_a,
            max_exact_sites=0,
            hybrid=True,
            hybrid_options={"n_gauss": 9, **options},
        ).to_qutip(block)
        return 0.5 * (rho / rho.tr() - exact).norm("tr")

    mean_field = state.partial_trace(sites_a, max_exact_sites=0, rpa=False).to_qutip(
        block
    )
    error_mf = 0.5 * (mean_field / mean_field.tr() - exact).norm("tr")
    assert abs(distance(charge_fluctuations=False) - error_mf) < 1e-6
    assert distance(charge_fluctuations=True) < 0.95 * error_mf


def test_hybrid_zero_modes_with_classical_modes_is_rejected():
    """The charge fluctuations are not supported with classical thermal modes."""
    state, sites, _ = _probe_state(6, 0.3, 30.0)
    try:
        state.partial_trace(
            frozenset([sites[0]]),
            max_exact_sites=0,
            hybrid=True,
            hybrid_options={"n_fock": 3},
        )
    except ValueError as error:
        assert "classical" in str(error)
    else:
        raise AssertionError("A ValueError was expected")
