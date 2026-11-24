"""Tests for qalma.correlations."""

import warnings
from itertools import product

import numpy as np
import pytest
import qutip

from qalma.correlations import (
    DEFAULT_MAX_DENSE_DIMENSION,
    ApproximateReductionWarning,
    concurrence,
    entanglement_entropy,
    entanglement_of_formation,
    log_negativity,
    mutual_information,
    negativity,
    pairwise,
    reduced_state,
    system_pairs,
)
from qalma.correlations._dense import (
    binary_entropy,
    concurrence_dense,
    entanglement_of_formation_from_concurrence,
    log_negativity_dense,
    mutual_information_dense,
    negativity_dense,
    partial_trace_dense,
    partial_transpose_dense,
    von_neumann_entropy_dense,
)
from qalma.correlations.reduction import resolve_subsystem
from qalma.model import FermionicSignWarning, build_spin_chain, build_system
from qalma.operators.states import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
    ProductDensityOperator,
)

TOL = 1e-10
RNG = np.random.default_rng(1234)


def random_density_matrix(dim, rank=None):
    """Random density matrix of the given dimension and rank."""
    rank = rank or dim
    gin = RNG.normal(size=(dim, rank)) + 1j * RNG.normal(size=(dim, rank))
    rho = gin @ gin.conj().T
    return rho / np.trace(rho)


def bell_states():
    """The four Bell states as 4x4 density matrices."""
    up, down = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    kets = [
        np.kron(up, up) + np.kron(down, down),
        np.kron(up, up) - np.kron(down, down),
        np.kron(up, down) + np.kron(down, up),
        np.kron(up, down) - np.kron(down, up),
    ]
    return [np.outer(k, k.conj()) / 2 for k in kets]


def werner(weight):
    """Werner state: weight * singlet + (1 - weight) * identity / 4."""
    return weight * bell_states()[3] + (1 - weight) * np.eye(4) / 4


def random_local_unitary():
    """Random U_A x U_B for two qubits."""
    u_a = qutip.rand_unitary(2, seed=int(RNG.integers(1 << 30))).full()
    u_b = qutip.rand_unitary(2, seed=int(RNG.integers(1 << 30))).full()
    return np.kron(u_a, u_b)


def spin_op(system, name, site):
    """Shorthand for a site operator."""
    return system.site_operator(f"{name}@{site}")


def heisenberg(system, bonds, field=None):
    """Heisenberg Hamiltonian on ``bonds`` plus optional local fields."""
    ham = sum(
        spin_op(system, f"S{a}", i) * spin_op(system, f"S{a}", j)
        for i, j in bonds
        for a in "xyz"
    )
    for (name, site), value in (field or {}).items():
        ham = ham + value * spin_op(system, name, site)
    return ham


# ---------------------------------------------------------------------------
# Dense kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("keep", [(0,), (2,), (0, 2), (2, 0), (1, 2, 0)])
def test_partial_trace_dense_matches_qutip(keep):
    dims = [2, 3, 2]
    rho = random_density_matrix(12)
    expected = qutip.Qobj(rho, dims=[dims, dims]).ptrace(sorted(keep))
    if list(keep) != sorted(keep):
        expected = expected.permute([sorted(keep).index(k) for k in keep])
    assert np.allclose(partial_trace_dense(rho, dims, keep), expected.full())


@pytest.mark.parametrize("mask", list(product([0, 1], repeat=3)))
def test_partial_transpose_dense_matches_qutip(mask):
    dims = [2, 3, 2]
    rho = random_density_matrix(12)
    expected = qutip.partial_transpose(qutip.Qobj(rho, dims=[dims, dims]), mask)
    transposed = [k for k, flag in enumerate(mask) if flag]
    assert np.allclose(partial_transpose_dense(rho, dims, transposed), expected.full())


def test_dense_rejects_incompatible_dims():
    with pytest.raises(ValueError):
        partial_transpose_dense(np.eye(4), [2, 3], [1])
    with pytest.raises(ValueError):
        concurrence_dense(np.eye(3))
    with pytest.raises(ValueError):
        negativity_dense(np.zeros((4, 4)), [2, 2], [1])


@pytest.mark.parametrize("rho", bell_states())
def test_bell_states(rho):
    assert abs(negativity_dense(rho, [2, 2], [1]) - 0.5) < TOL
    assert abs(log_negativity_dense(rho, [2, 2], [1]) - 1.0) < TOL
    assert abs(concurrence_dense(rho) - 1.0) < TOL
    assert abs(entanglement_of_formation_from_concurrence(1.0) - 1.0) < TOL
    assert abs(mutual_information_dense(rho, [2, 2], [0], [1]) - 2.0) < TOL
    assert abs(von_neumann_entropy_dense(rho)) < TOL


@pytest.mark.parametrize("weight", np.linspace(0, 1, 11))
def test_werner_states(weight):
    rho = werner(weight)
    expected_c = max(0.0, (3 * weight - 1) / 2)
    assert abs(concurrence_dense(rho) - expected_c) < TOL
    assert abs(negativity_dense(rho, [2, 2], [1]) - expected_c / 2) < TOL
    assert abs(log_negativity_dense(rho, [2, 2], [0]) - np.log2(1 + expected_c)) < TOL


def test_product_and_maximally_mixed_states_are_uncorrelated():
    rho_a, rho_b = random_density_matrix(2), random_density_matrix(3)
    rho = np.kron(rho_a, rho_b)
    assert negativity_dense(rho, [2, 3], [1]) < TOL
    assert mutual_information_dense(rho, [2, 3], [0], [1]) < TOL
    assert concurrence_dense(np.kron(rho_a, random_density_matrix(2))) < TOL
    assert abs(von_neumann_entropy_dense(np.eye(4)) - 2.0) < TOL
    assert abs(von_neumann_entropy_dense(np.eye(4), base=np.e) - np.log(4)) < TOL


def test_concurrence_matches_qutip():
    for _ in range(20):
        rho = random_density_matrix(4, rank=int(RNG.integers(1, 5)))
        expected = qutip.concurrence(qutip.Qobj(rho, dims=[[2, 2], [2, 2]]))
        # qutip takes square roots of rounding errors: allow for its 1e-8 noise.
        assert abs(concurrence_dense(rho) - expected) < 1e-7


def test_concurrence_of_pure_states_is_exact():
    """For pure states C = |<psi| sigma_y x sigma_y |psi*>|, to rounding precision."""
    sigma_yy = np.kron(qutip.sigmay().full(), qutip.sigmay().full())
    for _ in range(200):
        psi = RNG.normal(size=4) + 1j * RNG.normal(size=4)
        psi /= np.linalg.norm(psi)
        expected = abs(psi.conj() @ sigma_yy @ psi.conj())
        assert abs(concurrence_dense(np.outer(psi, psi.conj())) - expected) < 1e-12


def test_two_qubit_measure_relations():
    """Local-unitary invariance, PPT criterion and C >= ||rho^T||_1 - 1."""
    for _ in range(50):
        rho = random_density_matrix(4, rank=int(RNG.integers(1, 4)))
        c_val = concurrence_dense(rho)
        n_val = negativity_dense(rho, [2, 2], [1])
        assert c_val >= 2 * n_val - 1e-10
        assert (c_val > 1e-8) == (n_val > 1e-8)
        u_mat = random_local_unitary()
        rotated = u_mat @ rho @ u_mat.conj().T
        assert abs(concurrence_dense(rotated) - c_val) < 1e-8
        assert abs(negativity_dense(rotated, [2, 2], [1]) - n_val) < 1e-10
        assert abs(negativity_dense(rho, [2, 2], [0]) - n_val) < 1e-10


def test_entanglement_of_formation_monotonic():
    values = [
        entanglement_of_formation_from_concurrence(c) for c in np.linspace(0, 1, 21)
    ]
    assert values[0] == 0.0
    assert abs(values[-1] - 1.0) < TOL
    assert all(b > a for a, b in zip(values, values[1:]))
    assert binary_entropy(0.5) == 1.0
    assert binary_entropy(0.0) == binary_entropy(1.0) == 0.0


# ---------------------------------------------------------------------------
# Qobj front end
# ---------------------------------------------------------------------------


def ghz3():
    return (
        qutip.tensor([qutip.basis(2, 0)] * 3) + qutip.tensor([qutip.basis(2, 1)] * 3)
    ).unit()


def w3():
    kets = [
        qutip.tensor(*[qutip.basis(2, int(k == j)) for k in range(3)]) for j in range(3)
    ]
    return sum(kets[1:], kets[0]).unit()


def test_qobj_ghz_and_w():
    ghz, w_state = ghz3(), w3()
    # Tracing one qubit of a GHZ state leaves a separable pair.
    assert negativity(ghz, 0, 1) < TOL
    assert concurrence(ghz, 0, 2) < TOL
    # ... but each qubit is maximally entangled with the other two.
    assert abs(negativity(ghz, 0) - 0.5) < TOL
    assert abs(log_negativity(ghz, [0], [1, 2]) - 1.0) < TOL
    # Every pair of a W state has concurrence 2/3.
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        assert abs(concurrence(w_state, i, j) - 2 / 3) < 1e-8
    assert abs(entanglement_entropy(w_state, 0) - binary_entropy(1 / 3)) < TOL
    assert abs(entanglement_entropy(qutip.ket2dm(ghz), [1]) - 1.0) < TOL


def test_qobj_subsystem_order():
    rho = qutip.Qobj(random_density_matrix(8), dims=[[2, 2, 2], [2, 2, 2]])
    red = reduced_state(rho, 2, 0)
    assert red.labels == (2, 0)
    assert np.allclose(red.matrix, rho.ptrace([0, 2]).permute([1, 0]).full())
    assert abs(negativity(rho, 2, 0) - negativity(rho, 0, 2)) < TOL
    assert abs(mutual_information(rho, 2, 0) - mutual_information(rho, 0, 2)) < TOL


def test_qobj_invalid_subsystems():
    rho = ghz3()
    with pytest.raises(ValueError):
        negativity(rho, [0, 1], [1])
    with pytest.raises(KeyError):
        negativity(rho, 0, 5)
    with pytest.raises(ValueError):
        negativity(rho, 0, 1, max_dimension=2)


# ---------------------------------------------------------------------------
# QALMA states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.1, 0.5, 1.0, 2.0, 10.0])
def test_heisenberg_dimer_thermal_state(beta):
    """For H = S1.S2, rho ~ exp(-beta H) is a Werner state of singlet weight p."""
    system = build_system("dimer", "spin")
    site_a, site_b = sorted(system.sites)
    state = GibbsDensityOperator(beta * heisenberg(system, [(site_a, site_b)]), system)
    p_singlet = 1 / (1 + 3 * np.exp(-beta))
    expected_c = max(0.0, 2 * p_singlet - 1)
    assert abs(concurrence(state, site_a, site_b) - expected_c) < 1e-8
    assert abs(negativity(state, site_a, site_b) - expected_c / 2) < 1e-8
    assert abs(negativity(state, site_a) - expected_c / 2) < 1e-8
    assert (
        abs(
            entanglement_of_formation(state, site_a, site_b)
            - entanglement_of_formation_from_concurrence(expected_c)
        )
        < 1e-8
    )


def test_spin_one_dimer_ground_state():
    """The S=0 state of two spin-1 is maximally entangled: E_N = log2(3)."""
    system = build_system("dimer", "spin", local_S=1)
    site_a, site_b = sorted(system.sites)
    state = GibbsDensityOperator(40 * heisenberg(system, [(site_a, site_b)]), system)
    assert abs(log_negativity(state, site_a, site_b) - np.log2(3)) < 1e-8
    assert abs(negativity(state, site_a, site_b) - 1.0) < 1e-8
    with pytest.raises(ValueError):
        concurrence(state, site_a, site_b)


@pytest.fixture(name="chain_state")
def fixture_chain_state():
    """An asymmetric Gibbs state on a 4-site ring."""
    system = build_spin_chain(4)
    sites = sorted(system.sites)
    bonds = list(zip(sites, sites[1:] + sites[:1]))
    field = {("Sz", sites[0]): 0.7, ("Sx", sites[2]): -0.4}
    return GibbsDensityOperator(2.0 * heisenberg(system, bonds, field), system)


def test_gibbs_state_matches_dense_reference(chain_state):
    sites = sorted(chain_state.system.sites)
    dense = chain_state.to_qutip(tuple(sites))
    for i, j in [(0, 1), (0, 2), (3, 1), (2, 0)]:
        assert (
            abs(concurrence(chain_state, sites[i], sites[j]) - concurrence(dense, i, j))
            < 1e-8
        )
        assert (
            abs(negativity(chain_state, sites[i], sites[j]) - negativity(dense, i, j))
            < 1e-8
        )
        assert (
            abs(
                mutual_information(chain_state, sites[i], sites[j])
                - mutual_information(dense, i, j)
            )
            < 1e-8
        )
        red = reduced_state(chain_state, sites[i], sites[j])
        assert red.labels == (sites[i], sites[j])
        assert np.allclose(red.matrix, reduced_state(dense, i, j).matrix)
    assert (
        abs(log_negativity(chain_state, sites[:2]) - log_negativity(dense, [0, 1]))
        < 1e-8
    )


def test_mutual_information_bounds_correlations(chain_state):
    """<Sz_a Sz_b>_c^2 <= 2 ||Sz||^4 I(a:b) ln 2."""
    system = chain_state.system
    sites = sorted(system.sites)
    for i, j in [(0, 1), (0, 2)]:
        sz_a, sz_b = spin_op(system, "Sz", sites[i]), spin_op(system, "Sz", sites[j])
        connected = np.real(
            chain_state.expect(sz_a * sz_b)
            - chain_state.expect(sz_a) * chain_state.expect(sz_b)
        )
        info = mutual_information(chain_state, sites[i], sites[j])
        assert connected**2 <= 2 * 0.5**4 * info * np.log(2) + 1e-12


def test_product_states_are_uncorrelated():
    system = build_spin_chain(4)
    sites = sorted(system.sites)
    rho_0 = random_density_matrix(2)
    states = [
        ProductDensityOperator({sites[0]: rho_0}, system=system),
        GibbsProductDensityOperator(
            -spin_op(system, "Sz", sites[0]) - spin_op(system, "Sx", sites[1]),
            system=system,
        ),
    ]
    for state in states:
        assert negativity(state, sites[0], sites[1]) == 0.0
        assert log_negativity(state, sites[0]) == 0.0
        assert mutual_information(state, sites[0], sites[1:]) == 0.0
        assert concurrence(state, sites[0], sites[1]) < TOL
        with pytest.raises(ValueError):
            negativity(state, sites[:2], sites[1:])
        with pytest.raises(KeyError):
            negativity(state, sites[0], "nonexistent")


def test_gibbs_state_too_large_needs_explicit_approximation(chain_state):
    sites = sorted(chain_state.system.sites)
    with pytest.raises(ValueError, match="allow_approximate"):
        negativity(chain_state, sites[0], sites[1], max_exact_sites=2)
    with pytest.warns(ApproximateReductionWarning):
        value = mutual_information(
            chain_state, sites[0], sites[1], max_exact_sites=2, allow_approximate=True
        )
    assert value >= 0.0


def test_max_dimension_guard(chain_state):
    sites = sorted(chain_state.system.sites)
    assert DEFAULT_MAX_DENSE_DIMENSION >= 16
    with pytest.raises(ValueError, match="max_dimension"):
        negativity(chain_state, sites[0], sites[1:], max_dimension=8)


def test_physical_sites_expand_to_qubits_and_warn_fermions():
    system = build_system("open chain lattice", "fermion Hubbard", L=2)
    node_a, node_b = sorted(system.site_qubits)
    assert resolve_subsystem(system, node_a) == system.site_qubits[node_a]
    assert resolve_subsystem(system, [node_b, node_a]) == (
        system.site_qubits[node_b] + system.site_qubits[node_a]
    )
    state = ProductDensityOperator({}, system=system)
    with pytest.warns(FermionicSignWarning):
        red = reduced_state(state, node_a, node_b)
    assert red.dims == (2, 2, 2, 2)
    assert red.n_a == 2


def test_spin_systems_do_not_warn(chain_state):
    sites = sorted(chain_state.system.sites)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        negativity(chain_state, sites[0], sites[1])


# ---------------------------------------------------------------------------
# Pairwise maps
# ---------------------------------------------------------------------------


def test_system_pairs():
    system = build_spin_chain(4)
    edges = system_pairs(system)
    assert len(edges) == 4
    assert len({frozenset(e) for e in edges}) == 4
    assert len(system_pairs(system, "all")) == 6
    with pytest.raises(ValueError):
        system_pairs(system, "neighbours")


def test_pairwise(chain_state):
    sites = sorted(chain_state.system.sites)
    result = pairwise(chain_state, "concurrence")
    assert set(result) == set(system_pairs(chain_state.system))
    for (site_a, site_b), value in result.items():
        assert abs(value - concurrence(chain_state, site_a, site_b)) < 1e-10
    all_pairs = pairwise(chain_state, "negativity", "all")
    assert len(all_pairs) == 6
    custom = pairwise(
        chain_state,
        lambda state, a, b: mutual_information(state, a, b, base=np.e),
        [(sites[0], sites[2])],
    )
    expected = mutual_information(chain_state, sites[0], sites[2]) * np.log(2)
    assert abs(custom[(sites[0], sites[2])] - expected) < 1e-10
    with pytest.raises(ValueError):
        pairwise(chain_state, "discord")
