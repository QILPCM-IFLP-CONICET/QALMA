"""Tests for the MPS-based ground-state operator (qalma.mps)."""

import itertools
import pickle

import numpy as np
import pytest
import qutip
from scipy.sparse.linalg import eigsh

from qalma import build_system
from qalma.mps import MPS, MPSGSOperator, build_mpo, dmrg, mpo_to_dense
from qalma.mps.mpo import term_to_factors
from qalma.operators.states import GibbsDensityOperator, ProductDensityOperator

ATOL = 1e-7


def exact_ground_state(hamiltonian, system):
    """Return (E0, |gs>) with tensor factors in lexicographic site order."""
    block = tuple(sorted(system.sites))
    mat = hamiltonian.to_qutip(block).data_as("csr_matrix")
    vals, vecs = eigsh(mat, k=1, which="SA")
    dims = [system.dimensions[s] for s in block]
    return vals[0], qutip.Qobj(vecs[:, 0], dims=[dims, [1] * len(dims)])


def chain(n_sites, **kwargs):
    system = build_system("chain lattice", "spin", L=n_sites, **kwargs)
    return system, system.global_operator("Hamiltonian")


def chiral_chain(n_sites):
    """Heisenberg chain plus terms that are not symmetric under site exchange."""
    system = build_system("chain lattice", "spin", L=n_sites)
    ham = (
        system.global_operator("Hamiltonian")
        + 0.4 * system.global_operator("chiral_z")
        + 0.3 * system.global_operator("Sz")
    )
    return system, ham


def test_term_factors_roundtrip():
    rng = np.random.default_rng(1)
    dims = [2, 3, 2]
    mat = rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12))
    fac = term_to_factors(mat, dims)
    assert fac[0].shape[0] == 1 and fac[-1].shape[3] == 1
    tens = np.einsum("aijb,bklc,cmnd->ikmjln", fac[0], fac[1], fac[2])
    assert np.allclose(tens.reshape(12, 12), mat)


def _embed(dims, positions, mat):
    """Dense matrix of a term acting on ``positions``, identity elsewhere."""
    sub_dims = [dims[p] for p in positions]
    others = [p for p in range(len(dims)) if p not in positions]
    size = int(np.prod(dims))
    res = np.zeros((size, size), dtype=complex)
    ranges = [range(d) for d in dims]
    for idx_r in itertools.product(*ranges):
        for idx_c in itertools.product(*ranges):
            if any(idx_r[p] != idx_c[p] for p in others):
                continue
            r_sub = np.ravel_multi_index([idx_r[p] for p in positions], sub_dims)
            c_sub = np.ravel_multi_index([idx_c[p] for p in positions], sub_dims)
            res[
                np.ravel_multi_index(idx_r, dims), np.ravel_multi_index(idx_c, dims)
            ] = mat[r_sub, c_sub]
    return res


def test_mpo_matches_dense_with_long_range_and_non_symmetric_terms():
    rng = np.random.default_rng(2)
    dims = [2, 2, 3, 2, 2]
    terms = []
    expected = np.zeros((int(np.prod(dims)),) * 2, dtype=complex)
    for positions in [(0,), (2,), (0, 1), (1, 4), (0, 2, 3), (0, 4), (1, 2, 3, 4)]:
        size = int(np.prod([dims[p] for p in positions]))
        mat = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        mat = mat + mat.conj().T
        terms.append((positions, mat))
        expected += _embed(dims, positions, mat)
    mpo = build_mpo(terms, dims)
    assert np.allclose(mpo_to_dense(mpo), expected)


@pytest.mark.parametrize("n_sites", [2, 3])
def test_dmrg_tiny_systems(n_sites):
    system, ham = chiral_chain(n_sites)
    e_0, _ = exact_ground_state(ham, system)
    rho = MPSGSOperator(ham).solve()
    assert abs(rho.energy - e_0) < ATOL


@pytest.mark.parametrize("n_sites", [6, 8])
def test_energy_and_state_vs_exact_diagonalization(n_sites):
    system, ham = chiral_chain(n_sites)
    e_0, gs = exact_ground_state(ham, system)
    rho = MPSGSOperator(ham, chi_max=32)
    _ = rho.mps
    assert abs(rho.energy - e_0) < ATOL
    dense = rho.to_qutip().full()
    assert np.allclose(dense, (gs * gs.dag()).full(), atol=1e-6)
    assert abs(rho.expect(ham) - e_0) < ATOL


def test_unsolved_state_is_lazy():
    _, ham = chain(4)
    rho = MPSGSOperator(ham)
    assert not rho.solved
    assert "unsolved" in repr(rho)
    _ = rho.mps
    assert rho.energy is not None


@pytest.mark.parametrize(
    "permutation",
    [None, [3, 1, 0, 2, 5, 4, 7, 6], list(range(7, -1, -1))],
)
def test_reduced_states_with_permutation(permutation):
    n_sites = 8
    system, ham = chiral_chain(n_sites)
    e_0, gs = exact_ground_state(ham, system)
    rho_exact = qutip.ket2dm(gs)
    block = tuple(sorted(system.sites))
    rho = MPSGSOperator(ham, permutation=permutation, chi_max=64)
    assert abs(rho.energy - e_0) < ATOL
    cases = [
        ("1[2]",),
        ("1[5]", "1[1]"),
        ("1[0]", "1[7]", "1[3]"),
        ("1[3]", "1[4]", "1[5]"),
        ("1[6]", "1[0]", "1[2]", "1[4]"),
    ]
    for sites in cases:
        reduced = rho.partial_trace(frozenset(sites))
        kept = sorted(sites)
        idx = [block.index(s) for s in kept]
        expected = rho_exact.ptrace(idx)
        got = reduced.to_qutip(tuple(kept))
        assert np.allclose(got.full(), expected.full(), atol=1e-6)
        # And in a different tensor-factor order
        reordered = reduced.to_qutip(tuple(reversed(kept)))
        idx_r = [block.index(s) for s in reversed(kept)]
        assert np.allclose(
            reordered.full(),
            rho_exact.ptrace(sorted(idx_r))
            .permute([sorted(idx_r).index(i) for i in idx_r])
            .full(),
            atol=1e-6,
        )
        # dense request over the same block
        assert np.allclose(rho.to_qutip(tuple(kept)).full(), expected.full(), atol=1e-6)


def test_expectation_values_of_non_local_operators():
    n_sites = 8
    system, ham = chiral_chain(n_sites)
    _, gs = exact_ground_state(ham, system)
    block = tuple(sorted(system.sites))
    rho = MPSGSOperator(ham, permutation=[2, 0, 5, 7, 1, 3, 6, 4])
    obs = (
        system.site_operator("Sx", "1[0]") * system.site_operator("Sy", "1[5]")
        + 0.7 * system.site_operator("Sz", "1[7]") * system.site_operator("Sz", "1[2]")
        + system.site_operator("Splus", "1[1]") * system.site_operator("Sminus", "1[6]")
    )
    expected = qutip.expect(obs.to_qutip(block), gs)
    assert abs(rho.expect(obs) - expected) < 1e-6
    values = rho.expect([system.site_operator("Sz", "1[3]"), obs])
    assert abs(values[1] - expected) < 1e-6


def test_lexicographic_order_is_not_the_chain_order():
    """With 12 sites, "1[10]" sorts before "1[1]" and "1[2]" and the order must be fixed."""
    n_sites = 12
    system, ham = chain(n_sites)
    natural = [f"1[{i}]" for i in range(n_sites)]
    e_0, gs = exact_ground_state(ham, system)

    default = MPSGSOperator(ham, chi_max=64)
    assert default.order[:4] == ("1[0]", "1[10]", "1[11]", "1[1]")
    fixed = MPSGSOperator(ham, permutation=natural, chi_max=64)
    assert fixed.order == tuple(natural)
    assert abs(default.energy - e_0) < ATOL
    assert abs(fixed.energy - e_0) < ATOL
    # the natural order gives smaller bond dimensions
    assert max(fixed.bond_dims) <= max(default.bond_dims)

    sites = frozenset({"1[2]", "1[10]"})
    block = tuple(sorted(system.sites))
    expected = qutip.ket2dm(gs).ptrace([block.index(s) for s in sorted(sites)])
    got = fixed.partial_trace(sites).to_qutip(tuple(sorted(sites)))
    assert np.allclose(got.full(), expected.full(), atol=1e-6)


def test_two_dimensional_lattice_orderings_agree():
    system = build_system("square lattice", "spin", L=3, W=3)
    ham = system.global_operator("Hamiltonian")
    e_0, _ = exact_ground_state(ham, system)
    lex = tuple(sorted(system.sites))
    # lexicographic is x-major; build the y-major order
    names = {tuple(int(v) for v in s[2:-1].split(",")): s for s in lex}
    y_major = [names[(x, y)] for y in range(3) for x in range(3)]
    snake = [names[(x if y % 2 == 0 else 2 - x, y)] for y in range(3) for x in range(3)]
    for order in (None, y_major, snake):
        rho = MPSGSOperator(ham, permutation=order, chi_max=64)
        assert abs(rho.energy - e_0) < ATOL


def test_permutation_validation_and_names():
    system, ham = chain(4)
    with pytest.raises(ValueError):
        MPSGSOperator(ham, permutation=[0, 1, 2])
    with pytest.raises(ValueError):
        MPSGSOperator(ham, permutation=[0, 1, 2, 2])
    with pytest.raises(ValueError):
        MPSGSOperator(ham, permutation=["1[0]", "1[1]", "1[2]", "x"])
    rho = MPSGSOperator(ham, permutation=[1, 0, 3, 2])
    assert rho.order == ("1[1]", "1[0]", "1[3]", "1[2]")
    named = MPSGSOperator(ham, permutation=["1[1]", "1[0]", "1[3]", "1[2]"])
    assert named.order == rho.order


def test_from_mps_uses_the_given_order():
    system, ham = chiral_chain(6)
    e_0, gs = exact_ground_state(ham, system)
    order = [3, 1, 5, 0, 2, 4]
    block = tuple(sorted(system.sites))
    # state vector with the factors in MPS order
    gs_in_order = gs.permute(order)
    mps = MPS.from_dense(gs_in_order.full().ravel(), [2] * 6)
    rho = MPSGSOperator.from_mps(mps, system, permutation=order, hamiltonian=ham)
    assert rho.energy is None
    assert abs(rho.expect(ham) - e_0) < ATOL
    assert np.allclose(rho.to_qutip().full(), (gs * gs.dag()).full(), atol=1e-8)
    assert block  # silence unused
    with pytest.raises(ValueError):
        MPSGSOperator.from_mps(MPS.product([[1, 0]] * 5), system)


def test_arithmetic_preserves_type_and_matches_dense():
    system, ham = chiral_chain(6)
    e_0, _ = exact_ground_state(ham, system)
    rho = MPSGSOperator(ham)
    assert isinstance(2 * rho, MPSGSOperator) and (2 * rho).prefactor == 2
    assert isinstance(rho * 3.0, MPSGSOperator) and (rho * 3.0).prefactor == 3
    assert isinstance(rho * (3 + 0j), MPSGSOperator)
    assert isinstance(rho / 4, MPSGSOperator) and (rho / 4).prefactor == 0.25
    double = rho + rho
    assert isinstance(double, MPSGSOperator) and double.prefactor == 2
    # the scaled copies share the MPS, even if they are made before solving it
    assert (2 * rho).mps is rho.mps
    assert (rho / 2).mps is rho.mps
    # mixtures
    sigma = ProductDensityOperator({}, system=system)
    mix = 0.25 * rho + 0.75 * sigma
    trace_h = np.real(ham.to_qutip(tuple(sorted(system.sites))).tr()) / 2**6
    assert abs(mix.expect(ham) - (0.25 * e_0 + 0.75 * trace_h)) < 1e-6
    # products with operators fall back to the dense representation
    assert (rho * ham).tr() == pytest.approx(e_0, abs=1e-6)


def test_interface_of_a_pure_state():
    system, ham = chain(4)
    rho = MPSGSOperator(ham)
    assert not hasattr(rho, "k")
    assert not isinstance(rho, GibbsDensityOperator)
    with pytest.raises(NotImplementedError):
        rho.logm()
    assert rho.isherm and rho.tr() == 1
    assert rho.normalize() is rho
    assert rho.acts_over() == frozenset(system.sites)
    assert rho.variational_free_energy(ham) == pytest.approx(rho.energy, abs=1e-8)
    empty = rho.partial_trace(frozenset())
    assert empty.tr() == 1


def test_dense_request_is_guarded():
    system, ham = chain(12)
    rho = MPSGSOperator(ham, chi_max=64)
    with pytest.raises(ValueError, match="dense"):
        rho.to_qutip()
    with pytest.raises(ValueError, match="dense"):
        rho.to_qutip_operator()
    # but small reduced states are fine
    assert rho.partial_trace(frozenset({"1[0]", "1[5]"})).tr() == 1


def test_pickle_and_cache_invalidation():
    system, ham = chain(6)
    rho = MPSGSOperator(ham)
    clone = pickle.loads(pickle.dumps(rho))  # pickled before being solved
    assert not clone.solved
    rho.solve()  # nothing stale may be cached by the pickling above
    clone = pickle.loads(pickle.dumps(rho))
    assert clone.energy == rho.energy
    assert abs(clone.expect(ham) - rho.energy) < ATOL


def test_dmrg_function_and_mps_utilities():
    system, ham = chain(6)
    block = tuple(sorted(system.sites))
    terms = []
    for term in ham.flat().terms:
        sites = tuple(sorted(term.acts_over()))
        terms.append(
            (tuple(block.index(s) for s in sites), term.to_qutip(sites).full())
        )
    mpo = build_mpo(terms, [2] * 6)
    state, info = dmrg(mpo, chi_max=16, seed=3)
    assert info["converged"]
    assert state.norm() == pytest.approx(1.0)
    assert abs(state.overlap(state)) == pytest.approx(1.0)
    dense = state.to_dense()
    again = MPS.from_dense(dense, [2] * 6)
    assert abs(abs(again.overlap(state)) - 1) < 1e-10
    rdm = state.reduced_density_matrix([1, 4])
    assert np.trace(rdm) == pytest.approx(1.0)
    assert np.allclose(rdm, rdm.conj().T)
