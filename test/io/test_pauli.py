"""HDF5 storage of Pauli strings, Pauli sums and systems split into qubits."""

import warnings

import h5py
import numpy as np
import pytest

from qalma.evolution import Simulation
from qalma.io import (
    HDF5FormatError,
    load_operators,
    read_system,
    save_operators,
    write_system,
)
from qalma.model import build_spin_chain, build_system, build_system_from_dims
from qalma.operators import PauliStringOperator, PauliSumOperator, SumOperator
from qalma.operators.functions import commutator
from qalma.operators.states import GibbsDensityOperator


def build(geometry, model, **parms):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_system(geometry, model, **parms)


QUBIT_SYSTEMS = {
    "spinless fermions": lambda: build(
        "chain lattice", "spinless fermions", L=4, V=0.5
    ),
    "hubbard": lambda: build("chain lattice", "fermion Hubbard", L=2, U=2, mu=0.3),
    "kondo": lambda: build("2 band open chain lattice", "Kondo lattice", L=2),
    "spins, forced split": lambda: build(
        "chain lattice", "spin", L=3, qubit_split=True
    ),
    "t-J": lambda: build("chain lattice", "t-J", L=3, J=0.5),
    "alternative t-J": lambda: build("chain lattice", "alternative t-J", L=2, J=0.5),
    "kondo, spin 1": lambda: build(
        "2 band open chain lattice", "Kondo lattice", L=2, local_S=1
    ),
}


def roundtrip(operators, tmp_path):
    filename = tmp_path / "ops.h5"
    save_operators(filename, operators, "w")
    return load_operators(filename)


def dense(op):
    block = tuple(op.system.sites)
    return op.to_qutip(block).full()


@pytest.mark.parametrize("name", list(QUBIT_SYSTEMS))
def test_qubit_systems_roundtrip(name, tmp_path):
    system = QUBIT_SYSTEMS[name]()
    filename = tmp_path / "system.h5"
    with h5py.File(filename, "w") as h5file:
        write_system(h5file.create_group("system"), system)
    with h5py.File(filename, "r") as h5file:
        loaded = read_system(h5file["system"])
        n_kinds = len({basis for basis in h5file["system/sites"]["basis"]})
    assert list(loaded.sites) == list(system.sites)
    assert loaded.qubit_representation == system.qubit_representation
    assert loaded.jordan_wigner == system.jordan_wigner
    assert loaded.constrained == system.constrained
    assert loaded.dimensions == system.dimensions
    assert loaded.site_qubits == system.site_qubits
    # The qubits of the sites of the same type share their local basis.
    assert n_kinds <= 3
    for node in system.physical_sites:
        for op_name in system.physical_sites[node]["operators"]:
            original = system.site_operator(op_name, node)
            rebuilt = loaded.site_operator(op_name, node)
            assert type(rebuilt) is type(original)
            assert np.allclose(dense(rebuilt), dense(original))


@pytest.mark.parametrize("name", ["spinless fermions", "hubbard"])
def test_fermionic_operators_and_states(name, tmp_path):
    system = QUBIT_SYSTEMS[name]()
    ham = system.global_operator("Hamiltonian")
    node = tuple(system.physical_sites)[0]
    annihilation = system.site_operator(
        "c" if name == "spinless fermions" else "c_up", node
    )
    operators = {
        "H": ham,
        "c": annihilation,
        "commutator": commutator(ham, commutator(ham, annihilation)),
        "string": system.jordan_wigner_string(tuple(system.sites)[-1]),
        "gibbs": GibbsDensityOperator(ham, system),
    }
    loaded = roundtrip(operators, tmp_path)
    loaded_system = loaded["H"].system
    assert loaded_system.jordan_wigner
    for key in ("H", "c", "commutator"):
        assert isinstance(loaded[key], PauliSumOperator)
        assert loaded[key].system is loaded_system
        assert np.array_equal(loaded[key].codes, operators[key].codes)
        assert np.array_equal(loaded[key].coeffs, operators[key].coeffs)
    assert isinstance(loaded["string"], PauliStringOperator)
    assert loaded["string"].code == operators["string"].code
    gibbs = loaded["gibbs"]
    assert isinstance(gibbs.k, PauliSumOperator)
    assert np.allclose(dense(gibbs), dense(operators["gibbs"]))
    # The loaded operators work with the rebuilt system.
    rebuilt_c = loaded_system.site_operator(
        "c" if name == "spinless fermions" else "c_up", node
    )
    assert np.allclose(
        dense(commutator(loaded["H"], rebuilt_c)), dense(commutator(ham, annihilation))
    )


def test_operator_on_a_subsystem_with_another_site_order(tmp_path):
    """Codes are written in the order of the table of sites of the file."""
    system = build_system_from_dims({"a": 2, "b": 2, "c": 2, "d": 2})
    subsystem = system.subsystem(frozenset({"d", "b"}))
    rng = np.random.default_rng(3)
    on_subsystem = PauliSumOperator(
        rng.integers(0, 16, size=6).tolist(), rng.normal(size=6), subsystem
    )
    full = PauliSumOperator(
        rng.integers(0, 256, size=10).tolist(), rng.normal(size=10), system
    )
    loaded = roundtrip({"full": full, "sub": on_subsystem}, tmp_path)
    assert loaded["sub"].system.sites.keys() == subsystem.sites.keys()
    block = ("b", "d")
    assert np.allclose(
        loaded["sub"].to_qutip(block).full(), on_subsystem.to_qutip(block).full()
    )
    assert loaded["full"].to_dict() == full.to_dict()


def test_many_words(tmp_path):
    system = build_spin_chain(70)
    rng = np.random.default_rng(5)
    codes = [
        sum(int(c) << (2 * k) for k, c in enumerate(rng.integers(0, 4, size=70)))
        for _ in range(20)
    ]
    operator = PauliSumOperator(codes, rng.normal(size=20), system)
    loaded = roundtrip({"op": operator}, tmp_path)["op"]
    assert loaded.codes.shape == (20, 3)
    assert loaded.to_dict() == operator.to_dict()


def test_layout_can_be_read_without_qalma(tmp_path):
    system = QUBIT_SYSTEMS["hubbard"]()
    ham = system.global_operator("Hamiltonian")
    filename = tmp_path / "ops.h5"
    save_operators(filename, {"H": ham}, "w")
    with h5py.File(filename, "r") as h5file:
        group = h5file["operators/H"]
        assert group.attrs["type"] == "pauli_sum"
        assert group["codes"].dtype == np.uint64
        assert group["coeffs"].dtype == np.complex128
        names = [name.decode() for name in h5file["system/sites"]["name"]]
        codes, coeffs = group["codes"][()], group["coeffs"][()]
    # Rebuild the terms from the arrays and the table of sites.
    terms = {}
    for row, coeff in zip(codes, coeffs):
        code = int(row[0])
        label = tuple(
            (names[k], "IXYZ"[(code >> (2 * k)) & 3])
            for k in range(len(names))
            if (code >> (2 * k)) & 3
        )
        terms[label] = coeff
    expected = {}
    for term in ham.terms:
        paulis = getattr(term, "paulis", {})
        expected[tuple((s, "IXYZ"[c]) for s, c in paulis.items())] = term.prefactor
    assert terms == expected


def write_and_edit(tmp_path, edit):
    system = build_spin_chain(3)
    operator = PauliSumOperator([1, 6, 9], [1.0, 2.0, 3.0], system)
    filename = tmp_path / "ops.h5"
    save_operators(filename, {"op": operator}, "w")
    with h5py.File(filename, "r+") as h5file:
        edit(h5file["operators/op"])
    return filename


def replace(group, name, data):
    del group[name]
    group.create_dataset(name, data=data)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda g: replace(g, "codes", g["codes"][()].astype(np.int64)), "uint64"),
        (lambda g: replace(g, "coeffs", np.ones(2, complex)), "coefficients"),
        (
            lambda g: replace(g, "codes", np.array([[1 << 10], [1], [2]], np.uint64)),
            "more than 3",
        ),
        (lambda g: replace(g, "codes", np.zeros((3, 2), np.uint64)), "words"),
    ],
)
def test_invalid_files(tmp_path, edit, message):
    filename = write_and_edit(tmp_path, edit)
    with pytest.raises(HDF5FormatError, match=message):
        load_operators(filename)


def test_simulation_with_fermionic_states(tmp_path):
    system = QUBIT_SYSTEMS["spinless fermions"]()
    ham = system.global_operator("Hamiltonian")
    states = [GibbsDensityOperator(ham * beta, system) for beta in (0.5, 1.0)]
    simulation = Simulation(
        parameters={"H": ham},
        stats={},
        time_span=[0.0, 1.0],
        expect_ops={"energy": np.array([s.expect(ham) for s in states])},
        states=states,
    )
    filename = str(tmp_path / "sim.h5")
    simulation.save_hdf5(filename, mode="w")
    loaded = Simulation.load_hdf5(filename)
    assert isinstance(loaded.parameters["H"], PauliSumOperator)
    for original, state in zip(states, loaded.states):
        assert isinstance(state.k, PauliSumOperator)
        assert np.isclose(state.expect(loaded.parameters["H"]), original.expect(ham))


def test_t_j_operators_and_states(tmp_path):
    system = QUBIT_SYSTEMS["t-J"]()
    ham = system.global_operator("Hamiltonian")
    unphysical = system.operators["global_operators"]["unphysical_sites"]
    operators = {
        "H": ham,
        "Q": unphysical,
        "c": system.site_operator("c_up", "1[1]"),
        "gibbs": GibbsDensityOperator(ham + 30 * unphysical, system),
    }
    loaded = roundtrip(operators, tmp_path)
    loaded_system = loaded["H"].system
    assert loaded_system.constrained
    for key, original in operators.items():
        assert np.allclose(dense(loaded[key]), dense(original))
    assert np.isclose(
        loaded["gibbs"].expect(loaded["H"]), operators["gibbs"].expect(ham)
    )


def test_empty_and_generic_sums(tmp_path):
    system = build_spin_chain(3)
    strings = (
        PauliStringOperator.from_label("XXI", system, 1.0),
        PauliStringOperator.from_label("IZZ", system, 2.0),
    )
    loaded = roundtrip(
        {
            "empty": PauliSumOperator([], [], system),
            "generic": SumOperator(strings, system),
        },
        tmp_path,
    )
    assert isinstance(loaded["empty"], PauliSumOperator) and len(loaded["empty"]) == 0
    assert [type(term) for term in loaded["generic"].terms] == [PauliStringOperator] * 2
    assert {t.code: t.prefactor for t in loaded["generic"].terms} == {
        s.code: s.prefactor for s in strings
    }
