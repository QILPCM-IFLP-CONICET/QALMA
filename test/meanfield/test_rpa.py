"""Tests for the RPA reduction of Gibbs states."""

import numpy as np
import qutip

from qalma.meanfield import variational_quadratic_mfa
from qalma.meanfield.gibbs_partial_trace import gibbs_meanfield_generator
from qalma.meanfield.rpa import (
    RPAGibbsDensityOperator,
    _split_sites,
    build_rpa_structure,
    kubo_mori_susceptibility,
    local_boson_modes,
    rpa_coupling_kernel,
    rpa_dynamic_kernel,
)
from qalma.model import build_spin_chain, build_system
from qalma.operators.states.gibbs import GibbsDensityOperator

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
    """Gibbs state of the Hamiltonian, with a fixed mean-field state."""
    hamiltonian = system.global_operator("Hamiltonian")
    k_op = (beta * hamiltonian).simplify()
    sigma_mf = variational_quadratic_mfa(k_op.flat())
    state = GibbsDensityOperator(k_op, meanfield=sigma_mf)
    return state, sigma_mf


def _reduced_generator(state, sites, **kwargs):
    """Reduced generator of ``state`` over ``sites``, as a qutip operator."""
    generator, _ = gibbs_meanfield_generator(state, sites, **kwargs)
    return generator.to_qutip(tuple(sorted(sites)))


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
            w = np.where(small, p[:, None] * np.ones_like(delta), num / np.where(small, 1, den))
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
    ops = {s: structure.local_data(s) for s in sites_e}
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
    block = tuple(sorted(sites_a))
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
    block = tuple(sorted(sites_a))
    state, sigma_mf = _setup(system, 0.8)
    k_static = _reduced_generator(state, sites_a, rpa=True)
    k_dynamic = _reduced_generator(state, sites_a, rpa=True, dynamic=True)
    assert np.linalg.norm((k_static - k_dynamic).full()) < 1e-6


def test_dynamic_corrects_adjacent_bond():
    """The dynamical term renormalizes the coupling between adjacent sites."""
    system = build_spin_chain(10, field=0.3)
    sites = list(system.sites)
    sites_a = frozenset(sites[4:6])
    block = tuple(sorted(sites_a))
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

    mean_field = trace_distance()
    assert trace_distance(rpa=True) < 0.5 * mean_field
    assert trace_distance(rpa=True, dynamic=True) < 0.5 * mean_field
    # ``reduce`` forwards the options
    rho_reduce = state.reduce(sites_a, rpa=True).to_qutip(block)
    rho_trace = state.partial_trace(sites_a, rpa=True).to_qutip(block)
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
    """Its reductions coincide with those of a Gibbs state with ``rpa=True``."""
    system = build_spin_chain(10, field=0.3)
    sites = list(system.sites)
    sites_a = frozenset([sites[4], sites[6]])
    block = tuple(sorted(sites_a))
    rpa_state, plain_state = _rpa_state(system, 0.5)
    reduced = rpa_state.partial_trace(sites_a).to_qutip(block)
    expected_rpa = plain_state.partial_trace(sites_a, rpa=True).to_qutip(block)
    mean_field = plain_state.partial_trace(sites_a).to_qutip(block)
    assert (reduced - expected_rpa).norm("tr") < 1e-10
    assert (reduced - mean_field).norm("tr") > 1e-4
    # the options can be overridden
    no_rpa = rpa_state.partial_trace(sites_a, rpa=False).to_qutip(block)
    assert (no_rpa - mean_field).norm("tr") < 1e-10
    dynamic = rpa_state.partial_trace(sites_a, dynamic=True).to_qutip(block)
    expected_dyn = plain_state.partial_trace(
        sites_a, rpa=True, dynamic=True
    ).to_qutip(block)
    assert (dynamic - expected_dyn).norm("tr") < 1e-10


def test_rpa_state_dynamic_by_default():
    """The ``dynamic`` option given at construction is the default."""
    system = build_spin_chain(10, field=0.3)
    sites = list(system.sites)
    sites_a = frozenset(sites[4:6])
    block = tuple(sorted(sites_a))
    rpa_state, plain_state = _rpa_state(system, 0.5, dynamic=True)
    reduced = rpa_state.partial_trace(sites_a).to_qutip(block)
    expected = plain_state.partial_trace(sites_a, rpa=True, dynamic=True).to_qutip(
        block
    )
    assert (reduced - expected).norm("tr") < 1e-10


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


def test_rpa_state_exact_option():
    """Small systems are reduced exactly unless ``exact=False``."""
    system = build_spin_chain(4, field=0.3)
    sites = list(system.sites)
    sites_a = frozenset(sites[:2])
    block = tuple(sorted(sites_a))
    rpa_state, plain_state = _rpa_state(system, 0.5)
    exact = plain_state.to_qutip_operator().partial_trace(sites_a).to_qutip(block)
    reduced = rpa_state.partial_trace(sites_a).to_qutip(block)
    assert (reduced - exact).norm("tr") < 1e-10
    approx = rpa_state.partial_trace(sites_a, exact=False).to_qutip(block)
    mean_field = plain_state.partial_trace(sites_a, exact=False).to_qutip(block)
    assert (approx - exact).norm("tr") > 1e-6
    assert (approx - mean_field).norm("tr") > 1e-6
