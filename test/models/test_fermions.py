"""Tests for the treatment of fermionic models.

Fermionic models whose sites are products of two-level quantum numbers
(spinless fermions, Hubbard, Kondo) are represented by one qubit per
quantum number, with fermionic modes mapped to Pauli strings through the
Jordan-Wigner transformation, so their signs are exact. Models with other
local spaces (e.g. the constrained t-J basis) still treat operators on
different sites as commuting, and emit a warning.
"""

import itertools
import warnings

import numpy as np
import pytest

from qalma import FermionicSignWarning, build_system
from qalma.operators import PauliStringOperator
from qalma.operators.functions import anticommutator, commutator
from qalma.operators.pauli import pauli_terms


def _spinless(geometry="chain lattice", **parms):
    """Spinless fermions; by default free fermions on a periodic chain."""
    full_parms = {"t": 1, "V": 0, "mu": 0}
    full_parms.update(parms)
    return build_system(
        geometry_name=geometry, model_name="spinless fermions", **full_parms
    )


def _hopping_matrix(system, mu):
    """Single-particle Hamiltonian -sum_<ij> (c_i^+ c_j + h.c.) - mu n_i."""
    position = {site: k for k, site in enumerate(system.sites)}
    size = len(position)
    hop = -mu * np.eye(size)
    for edges in system.spec["graph"].edges.values():
        for src, dst in edges:
            hop[position[src], position[dst]] -= 1
            hop[position[dst], position[src]] -= 1
    return hop


def _dense_fermions(size):
    """Annihilation operators built with the JW string in the *reverse* order."""
    sz = np.diag([1.0, -1.0])
    lower = np.array([[0.0, 1.0], [0.0, 0.0]])
    eye = np.eye(2)
    result = []
    for j in range(size):
        factors = [eye] * j + [lower] + [sz] * (size - j - 1)
        op = factors[0]
        for factor in factors[1:]:
            op = np.kron(op, factor)
        result.append(op)
    return result


def test_disabled_split_warns():
    """Without the split, fermionic signs are not implemented: warning."""
    with pytest.warns(FermionicSignWarning):
        system = build_system(
            geometry_name="chain lattice", model_name="t-J", L=3, qubit_split=False
        )
    assert not system.qubit_representation


def test_spinless_fermions_do_not_warn():
    """Single-mode fermionic sites are handled exactly, without warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FermionicSignWarning)
        system = _spinless(L=3)
    assert system.jordan_wigner


def test_spin_model_does_not_warn():
    """Non-fermionic models do not emit the warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FermionicSignWarning)
        system = build_system(geometry_name="chain lattice", model_name="spin", L=3)
    assert not system.jordan_wigner


def test_free_fermions_on_a_ring():
    """Ground state energy of free fermions on a 4-site ring.

    The single particle energies are -2 cos(k) = -2, 0, 0, 2, so the ground
    state energy is -2 (hard-core bosons would give -2 sqrt(2)).
    """
    ham = _spinless(L=4).global_operator("Hamiltonian")
    assert np.isclose(min(ham.eigenenergies()), -2.0)


@pytest.mark.parametrize(
    "geometry, parms",
    [
        ("chain lattice", {"L": 5}),
        ("chain lattice", {"L": 6}),
        ("open chain lattice", {"L": 5}),
        ("square lattice", {"L": 3, "W": 2}),
    ],
)
def test_free_fermion_spectrum(geometry, parms):
    """The whole many-body spectrum is given by sums of single particle energies."""
    mu = 0.3
    system = _spinless(geometry, mu=mu, **parms)
    ham = system.global_operator("Hamiltonian")
    assert pauli_terms(ham) is not None
    eps = np.linalg.eigvalsh(_hopping_matrix(system, mu))
    expected = sorted(
        sum(subset)
        for k in range(len(eps) + 1)
        for subset in itertools.combinations(eps, k)
    )
    assert np.allclose(np.sort(ham.eigenenergies()), expected)


def test_interacting_spectrum_is_independent_of_the_string_order():
    """With V != 0, compare against a dense model built with another JW order."""
    mu, coupling = 0.2, 0.7
    system = _spinless("square lattice", L=3, W=2, mu=mu, V=coupling)
    ham = system.global_operator("Hamiltonian")
    position = {site: k for k, site in enumerate(system.sites)}
    c_ops = _dense_fermions(len(position))
    n_ops = [c.T @ c for c in c_ops]
    dense = -mu * sum(n_ops)
    for edges in system.spec["graph"].edges.values():
        for src, dst in edges:
            i, j = position[src], position[dst]
            dense = dense - (c_ops[i].T @ c_ops[j] + c_ops[j].T @ c_ops[i])
            dense = dense + coupling * n_ops[i] @ n_ops[j]
    assert np.allclose(np.sort(ham.eigenenergies()), np.linalg.eigvalsh(dense))


def test_canonical_anticommutation_relations():
    """{c_i, c_j^+} = delta_ij and {c_i, c_j} = 0 for the site operators."""
    system = _spinless("square lattice", L=2, W=2)
    sites = tuple(system.sites)
    c_ops = [system.site_operator("c", site) for site in sites]
    cdag_ops = [system.site_operator("cdag", site) for site in sites]
    for i, j in itertools.product(range(len(sites)), repeat=2):
        assert np.isclose(
            anticommutator(c_ops[i], cdag_ops[j]).tr() / 2 ** len(sites),
            1.0 if i == j else 0.0,
        )
        assert not anticommutator(c_ops[i], c_ops[j])
    # Even operators stay local.
    number = system.site_operator("n", sites[2])
    assert number.acts_over() == frozenset({sites[2]})


def test_heisenberg_evolution_stays_in_the_pauli_algebra():
    system = _spinless(L=6, V=0.5)
    ham = system.global_operator("Hamiltonian")
    obs = system.site_operator("c", tuple(system.sites)[0])
    result = obs
    for _ in range(3):
        result = commutator(ham, result)
        assert pauli_terms(result) is not None
    assert all(isinstance(term, PauliStringOperator) for term in pauli_terms(result))
    dense_h = ham.to_qutip().full()
    dense_o = obs.to_qutip().full()
    expected = dense_o
    for _ in range(3):
        expected = dense_h @ expected - expected @ dense_h
    assert np.allclose(result.to_qutip().full(), expected)


# ---------------------------------------------------------------------------
# Several modes per site: one qubit per species
# ---------------------------------------------------------------------------


def _hubbard(model="fermion Hubbard", geometry="chain lattice", **parms):
    full_parms = {"t": 1, "U": 1.7, "mu": 0.3, "V": 0.4}
    full_parms.update(parms)
    return build_system(geometry_name=geometry, model_name=model, **full_parms)


def test_hubbard_sites_are_split_into_qubits():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FermionicSignWarning)
        system = _hubbard(L=2)
    assert system.jordan_wigner
    assert tuple(system.sites) == ("1[0]:Nup", "1[0]:Ndown", "1[1]:Nup", "1[1]:Ndown")
    assert system.site_qubits["1[1]"] == ("1[1]:Nup", "1[1]:Ndown")
    assert system.qubit_parent["1[0]:Ndown"] == "1[0]"
    assert set(system.dimensions.values()) == {2}


def test_hubbard_canonical_anticommutation_relations():
    """Modes of the same site anticommute too."""
    system = _hubbard(L=2)
    modes = [(node, spin) for node in system.physical_sites for spin in ("up", "down")]
    c_ops = [system.site_operator(f"c_{spin}", node) for node, spin in modes]
    cdag_ops = [system.site_operator(f"cdag_{spin}", node) for node, spin in modes]
    dim = 2 ** len(system.sites)
    for i, j in itertools.product(range(len(modes)), repeat=2):
        assert np.isclose(
            anticommutator(c_ops[i], cdag_ops[j]).tr() / dim, 1.0 if i == j else 0.0
        )
        assert not anticommutator(c_ops[i], c_ops[j])


@pytest.mark.parametrize(
    "model",
    ["fermion Hubbard", "alternative fermion Hubbard", "t-J", "alternative t-J"],
)
def test_local_operators_follow_the_alps_convention(model):
    """Splus = c^+_up c_down, Sminus = c^+_down c_up, n = n_up + n_down."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system = build_system(geometry_name="chain lattice", model_name=model, L=2)
    node = tuple(system.physical_sites)[1]

    def dense(op):
        return op.to_qutip(tuple(system.sites)).full()

    def site_op(name):
        return system.site_operator(name, node)

    assert np.allclose(
        dense(site_op("Splus")), dense(site_op("cdag_up") * site_op("c_down"))
    )
    assert np.allclose(
        dense(site_op("Sminus")), dense(site_op("cdag_down") * site_op("c_up"))
    )
    number = site_op("cdag_up") * site_op("c_up") + site_op("cdag_down") * site_op(
        "c_down"
    )
    assert np.allclose(dense(site_op("n")), dense(number))
    if "double_occupancy" in system.physical_sites[node]["operators"]:
        assert np.allclose(
            dense(site_op("double_occupancy")),
            dense(site_op("n_up") * site_op("n_down")),
        )


def _dense_hubbard(system, parms):
    """Hubbard Hamiltonian built with a JW order: all down modes, then all up."""
    position = {node: k for k, node in enumerate(system.physical_sites)}
    size = len(position)
    c_ops = _dense_fermions(2 * size)
    c_down, c_up = c_ops[:size], c_ops[size:]
    number = [c_up[i].T @ c_up[i] + c_down[i].T @ c_down[i] for i in range(size)]
    dense = sum(
        -parms["mu"] * number[i]
        + parms["U"] * (c_up[i].T @ c_up[i]) @ (c_down[i].T @ c_down[i])
        for i in range(size)
    )
    for edges in system.spec["graph"].edges.values():
        for src, dst in edges:
            i, j = position[src], position[dst]
            for c_spin in (c_up, c_down):
                dense = dense - (c_spin[i].T @ c_spin[j] + c_spin[j].T @ c_spin[i])
            dense = dense + parms["V"] * number[i] @ number[j]
    return dense


@pytest.mark.parametrize("model", ["fermion Hubbard", "alternative fermion Hubbard"])
@pytest.mark.parametrize(
    "geometry, parms",
    [
        ("chain lattice", {"L": 3}),
        ("open chain lattice", {"L": 3}),
        ("square lattice", {"L": 2, "W": 2}),
    ],
)
def test_hubbard_spectrum(model, geometry, parms):
    hubbard_parms = {"t": 1, "U": 1.7, "mu": 0.3, "V": 0.4}
    system = _hubbard(model, geometry, **parms)
    ham = system.global_operator("Hamiltonian")
    assert pauli_terms(ham) is not None
    expected = np.linalg.eigvalsh(_dense_hubbard(system, hubbard_parms))
    assert np.allclose(np.sort(ham.eigenenergies()), expected)


def test_conserved_quantities_are_pauli_sums():
    system = _hubbard(L=3)
    n_up = system.operators["global_operators"]["Nup"]
    assert pauli_terms(n_up) is not None
    ham = system.global_operator("Hamiltonian")
    assert not commutator(ham, n_up)


def test_kondo_like_system():
    """Fermion sites are split; spin-1/2 sites are qubits outside the JW strings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FermionicSignWarning)
        system = build_system(
            geometry_name="2 band open chain lattice", model_name="Kondo lattice", L=2
        )
    assert system.jordan_wigner
    assert tuple(system.sites) == (
        "1[0]:Nup",
        "1[0]:Ndown",
        "2[0]",
        "1[1]:Nup",
        "1[1]:Ndown",
        "2[1]",
    )
    c_up = system.site_operator("c_up", "1[1]")
    spin_ops = [system.site_operator(name, "2[0]") for name in ("Splus", "Sz")]
    for spin_op in spin_ops:
        assert not commutator(c_up, spin_op)
    # The string skips the spin qubit.
    assert "2[0]" not in system.jordan_wigner_string("1[1]:Nup").acts_over()


def test_qubit_split_can_be_forced_or_disabled():
    spins = build_system(geometry_name="chain lattice", model_name="spin", L=4)
    qubits = build_system(
        geometry_name="chain lattice", model_name="spin", L=4, qubit_split=True
    )
    assert qubits.qubit_representation and not qubits.jordan_wigner
    assert np.allclose(
        spins.global_operator("Hamiltonian").to_qutip().full(),
        qubits.global_operator("Hamiltonian").to_qutip().full(),
    )
    with pytest.warns(FermionicSignWarning):
        bosonic = _hubbard(L=2, qubit_split=False)
    assert not bosonic.qubit_representation
    with pytest.raises(ValueError):
        build_system(
            geometry_name="chain lattice",
            model_name="spin",
            L=2,
            local_S=1,
            qubit_split=True,
        )


def test_subsystem_of_qubits():
    system = _hubbard(L=3)
    sub = system.subsystem(frozenset({"1[1]:Ndown", "1[0]:Nup"}))
    assert tuple(sub.sites) == ("1[0]:Nup", "1[1]:Ndown")
    assert sub.site_qubits == {"1[0]": ("1[0]:Nup",), "1[1]": ("1[1]:Ndown",)}


def test_gibbs_state_with_unsorted_site_names():
    """GibbsDensityOperator.to_qutip respects ``block`` when it differs from the site order."""
    import scipy.linalg

    from qalma.operators.states import GibbsDensityOperator

    system = _hubbard(L=2)
    ham = system.global_operator("Hamiltonian")
    for block in (None, tuple(system.sites)):
        dense_h = ham.to_qutip(block).full()
        expected = scipy.linalg.expm(-dense_h)
        expected /= np.trace(expected)
        state = GibbsDensityOperator(ham, system=system)
        for _ in range(2):  # before and after normalization
            assert np.allclose(state.to_qutip(block).full(), expected)
    full = GibbsDensityOperator(ham, system=system).to_qutip(tuple(system.sites))
    reduced = GibbsDensityOperator(ham, system=system).to_qutip(
        ("1[1]:Nup", "1[0]:Nup")
    )
    assert np.allclose(reduced.full(), full.ptrace([2, 0]).full())


# ---------------------------------------------------------------------------
# Constrained sites: t-J
# ---------------------------------------------------------------------------


def _t_j(model="t-J", geometry="chain lattice", **parms):
    full_parms = {"t": 1.0, "J": 0.6, "mu": 0.2, "V": 0.3}
    full_parms.update(parms)
    return build_system(geometry_name=geometry, model_name=model, **full_parms)


def _reference_t_j(system, parms):
    """t-J from projected electrons in the Fock space of Hubbard.

    Uses another Jordan-Wigner order (all the down modes, then all the up
    modes) and keeps the states without doubly occupied sites.
    """
    position = {node: k for k, node in enumerate(system.physical_sites)}
    size = len(position)
    c_ops = _dense_fermions(2 * size)
    c_down, c_up = c_ops[:size], c_ops[size:]
    identity = np.eye(4**size)
    n_up = [c.T @ c for c in c_up]
    n_down = [c.T @ c for c in c_down]
    # projected electrons
    p_up = [c_up[i] @ (identity - n_down[i]) for i in range(size)]
    p_down = [c_down[i] @ (identity - n_up[i]) for i in range(size)]
    number = [n_up[i] + n_down[i] for i in range(size)]
    s_z = [(n_up[i] - n_down[i]) / 2 for i in range(size)]
    s_plus = [c_up[i].T @ c_down[i] for i in range(size)]
    ham = sum(-parms["mu"] * number[i] for i in range(size))
    for edges in system.spec["graph"].edges.values():
        for src, dst in edges:
            i, j = position[src], position[dst]
            for p_spin in (p_up, p_down):
                ham = ham - parms["t"] * (
                    p_spin[i].T @ p_spin[j] + p_spin[j].T @ p_spin[i]
                )
            exchange = s_z[i] @ s_z[j] + 0.5 * (
                s_plus[i] @ s_plus[j].T + s_plus[i].T @ s_plus[j]
            )
            ham = ham + parms["J"] * exchange
            ham = ham + (parms["V"] - parms["J"] / 4) * number[i] @ number[j]
    double = sum(n_up[i] @ n_down[i] for i in range(size))
    physical = np.where(np.isclose(np.diag(double), 0))[0]
    return np.linalg.eigvalsh(ham[np.ix_(physical, physical)])


def _physical_spectrum(system):
    """Spectrum of the Hamiltonian in the sector without unphysical sites."""
    block = tuple(system.sites)
    ham = system.global_operator("Hamiltonian").to_qutip(block).full()
    unphysical = system.operators["global_operators"]["unphysical_sites"]
    q_matrix = unphysical.to_qutip(block).full()
    q_values = np.real(np.diag(q_matrix))
    assert np.allclose(q_matrix, np.diag(q_values))
    physical = np.isclose(q_values, 0)
    # The Hamiltonian does not mix the physical states with the others.
    assert np.allclose(ham[np.ix_(physical, ~physical)], 0)
    return np.sort(np.linalg.eigvalsh(ham[np.ix_(physical, physical)]))


def test_t_j_sites_are_split():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FermionicSignWarning)
        system = _t_j(L=2)
    assert system.jordan_wigner and system.constrained
    assert tuple(system.sites) == ("1[0]:N", "1[0]:Sz", "1[1]:N", "1[1]:Sz")
    assert system.dimensions == {"1[0]:N": 2, "1[0]:Sz": 3, "1[1]:N": 2, "1[1]:Sz": 3}
    assert system.site_qubits["1[1]"] == ("1[1]:N", "1[1]:Sz")
    # The rest subsites are outside the Jordan-Wigner strings.
    assert system.jordan_wigner_string("1[1]:N").acts_over() == frozenset({"1[0]:N"})
    alternative = _t_j("alternative t-J", L=2)
    assert alternative.constrained
    assert tuple(alternative.sites)[:2] == ("1[0]:Nup", "1[0]:Ndown")


@pytest.mark.parametrize("model", ["t-J", "alternative t-J"])
@pytest.mark.parametrize(
    "geometry, parms",
    [
        ("chain lattice", {"L": 3}),
        ("chain lattice", {"L": 4}),
        ("open chain lattice", {"L": 4}),
        ("square lattice", {"L": 2, "W": 2}),
    ],
)
def test_t_j_spectrum(model, geometry, parms):
    t_j_parms = {"t": 1.0, "J": 0.6, "mu": 0.2, "V": 0.3}
    system = _t_j(model, geometry, **parms)
    expected = _reference_t_j(system, t_j_parms)
    assert len(expected) == 3 ** len(system.physical_sites)
    assert np.allclose(_physical_spectrum(system), expected)


def test_t_j_without_signs_is_different():
    """The ring of 3 sites distinguishes fermions from hard-core particles."""
    t_j_parms = {"t": 1.0, "J": 0.6, "mu": 0.2, "V": 0.3}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FermionicSignWarning)
        bosonic = _t_j(L=3, qubit_split=False)
    expected = _reference_t_j(bosonic, t_j_parms)
    hard_core = np.sort(bosonic.global_operator("Hamiltonian").eigenenergies())
    assert not np.allclose(hard_core, expected)


def test_t_j_operators():
    system = _t_j(L=3)
    sites = tuple(system.physical_sites)
    block = tuple(system.sites)
    unphysical = system.operators["global_operators"]["unphysical_sites"]
    projector = np.diag(
        np.isclose(np.real(np.diag(unphysical.to_qutip(block).full())), 0).astype(float)
    )

    def dense(op):
        return projector @ op.to_qutip(block).full() @ projector

    ham = system.global_operator("Hamiltonian")
    assert not commutator(ham, unphysical)
    c_ops = {
        (site, spin): system.site_operator(f"c_{spin}", site)
        for site in sites
        for spin in ("up", "down")
    }
    for (site_a, spin_a), c_a in c_ops.items():
        for (site_b, spin_b), c_b in c_ops.items():
            # Projected electrons on different sites anticommute.
            if site_a != site_b:
                assert np.allclose(dense(anticommutator(c_a, c_b)), 0)
                assert np.allclose(dense(anticommutator(c_a, c_b.dag())), 0)
    for site in sites:
        s_plus, s_minus, s_z, number = (
            dense(system.site_operator(name, site))
            for name in ("Splus", "Sminus", "Sz", "n")
        )
        assert np.allclose(s_plus @ s_minus - s_minus @ s_plus, 2 * s_z)
        assert np.allclose(
            s_z @ s_z + 0.5 * (s_plus @ s_minus + s_minus @ s_plus), 0.75 * number
        )


def test_t_j_gibbs_state_with_penalty():
    """A large penalty on the unphysical sites gives the physical Gibbs state."""
    from qalma.operators.states import GibbsDensityOperator

    system = _t_j(L=3)
    ham = system.global_operator("Hamiltonian")
    unphysical = system.operators["global_operators"]["unphysical_sites"]
    state = GibbsDensityOperator(ham + 40 * unphysical, system)
    energy = state.expect(ham)
    eigenvalues = _physical_spectrum(system)
    weights = np.exp(-(eigenvalues - eigenvalues.min()))
    assert np.isclose(energy, np.sum(weights * eigenvalues) / np.sum(weights))
    assert abs(state.expect(unphysical)) < 1e-12


def test_kondo_lattice_with_spin_one():
    """Spin-1 sites are a rest subsite: no warning, and they commute with fermions."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FermionicSignWarning)
        system = build_system(
            geometry_name="2 band open chain lattice",
            model_name="Kondo lattice",
            L=2,
            local_S=1,
        )
    assert system.jordan_wigner and not system.constrained
    assert system.dimensions["2[0]"] == 3
    c_up = system.site_operator("c_up", "1[1]")
    for name in ("Splus", "Sminus", "Sz"):
        assert not commutator(c_up, system.site_operator(name, "2[0]"))
    assert "2[0]" not in system.jordan_wigner_string("1[1]:Nup").acts_over()
