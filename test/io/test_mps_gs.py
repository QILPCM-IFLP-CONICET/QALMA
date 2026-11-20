"""Storage of ``MPSGSOperator`` in the HDF5 format."""

import os

import h5py
import numpy as np
import pytest

from qalma import build_system
from qalma.io import (
    HDF5FormatError,
    IncompleteStateWarning,
    load_operators,
    save_operators,
)
from qalma.mps import MPS, MPSGSOperator
from qalma.operators.states import ProductDensityOperator

ORDER_8 = [f"1[{i}]" for i in range(8)]


def chain(n_sites):
    system = build_system("open chain lattice", "spin", L=n_sites)
    return system, system.global_operator("Hamiltonian")


def save_and_load(tmp_path, operators, **kwargs):
    filename = tmp_path / "mps.h5"
    save_operators(filename, operators, "w", **kwargs)
    return filename, load_operators(filename)


def test_solved_state_is_restored_exactly(tmp_path):
    system, ham = chain(8)
    permutation = [3, 1, 0, 2, 5, 4, 7, 6]
    state = MPSGSOperator(
        ham,
        permutation=permutation,
        chi_max=16,
        prefactor=0.25,
        dmrg_options={"max_sweeps": 12, "tol": 1e-9, "seed": 4},
    ).solve()
    _, loaded = save_and_load(tmp_path, {"rho": state})
    loaded = loaded["rho"]

    assert type(loaded) is MPSGSOperator
    assert loaded.solved
    assert loaded.order == state.order
    assert loaded.prefactor == state.prefactor
    assert loaded.chi_max == state.chi_max
    assert loaded.dmrg_options == state.dmrg_options
    assert loaded.energy == state.energy
    assert loaded.mps.center == state.mps.center
    for tensor, original in zip(loaded.mps.tensors, state.mps.tensors):
        assert np.array_equal(tensor, original)
    assert loaded.dmrg_info["converged"] == state.dmrg_info["converged"]
    assert loaded.dmrg_info["sweeps"] == state.dmrg_info["sweeps"]
    assert loaded.dmrg_info["energies"] == state.dmrg_info["energies"]
    assert loaded.dmrg_info["bond_dims"] == state.dmrg_info["bond_dims"]
    assert list(loaded.system.sites) == list(state.system.sites)


def test_loaded_state_gives_the_same_results(tmp_path):
    system, ham = chain(8)
    state = MPSGSOperator(ham, permutation=ORDER_8, chi_max=16)
    _, loaded = save_and_load(tmp_path, {"rho": state})
    rho = loaded["rho"]
    loaded_system = rho.system
    obs = loaded_system.site_operator("Sz", "1[0]") * loaded_system.site_operator(
        "Sz", "1[5]"
    )
    original_obs = system.site_operator("Sz", "1[0]") * system.site_operator(
        "Sz", "1[5]"
    )
    assert abs(rho.expect(obs) - state.expect(original_obs)) < 1e-14
    sites = frozenset({"1[2]", "1[6]"})
    assert np.allclose(
        rho.partial_trace(sites).to_qutip(tuple(sorted(sites))).full(),
        state.partial_trace(sites).to_qutip(tuple(sorted(sites))).full(),
        atol=1e-14,
    )


def test_unsolved_state_is_stored_without_the_mps(tmp_path):
    system, ham = chain(6)
    state = MPSGSOperator(ham, chi_max=16, dmrg_options={"seed": 1})
    assert not state.solved
    filename, loaded = save_and_load(tmp_path, {"rho": state})
    # Saving does not run the DMRG.
    assert not state.solved
    with h5py.File(filename, "r") as h5file:
        group = h5file["operators/rho"]
        assert "mps" not in group and "energies" not in group
        assert "hamiltonian" in group
    rho = loaded["rho"]
    assert not rho.solved
    # The DMRG runs again, with the same options, when it is needed.
    assert abs(rho.energy - state.energy) < 1e-9


def test_state_without_hamiltonian(tmp_path):
    system, ham = chain(6)
    reference = MPSGSOperator(ham, permutation=[f"1[{i}]" for i in range(6)])
    order = [f"1[{i}]" for i in reversed(range(6))]
    # The same state with the chain reversed: no orthogonality center.
    mps = MPS([t.transpose(2, 1, 0) for t in reference.mps.tensors[::-1]])
    state = MPSGSOperator.from_mps(mps, system, permutation=order, prefactor=2.0)
    assert state.hamiltonian is None and state.energy is None
    _, loaded = save_and_load(tmp_path, {"rho": state})
    rho = loaded["rho"]
    assert rho.hamiltonian is None and rho.energy is None
    assert rho.order == tuple(order) and rho.prefactor == 2.0
    assert rho.mps.center is None
    assert (
        abs(rho.expect(rho.system.global_operator("Hamiltonian")) - reference.energy)
        < 1e-9
    )


def test_mixture_with_a_state_stored_in_a_file(tmp_path):
    system, ham = chain(6)
    gs = MPSGSOperator(ham, permutation=[f"1[{i}]" for i in range(6)], chi_max=16)
    mixture = 0.25 * gs + 0.75 * ProductDensityOperator({}, system=system)
    _, loaded = save_and_load(tmp_path, {"mix": mixture, "gs": gs})
    new_ham = loaded["mix"].system.global_operator("Hamiltonian")
    assert abs(loaded["mix"].expect(new_ham) - mixture.expect(ham)) < 1e-12
    assert any(isinstance(t, MPSGSOperator) for t in loaded["mix"].terms)


def test_no_pickle_is_needed(tmp_path):
    system, ham = chain(6)
    state = MPSGSOperator(ham, chi_max=16)
    filename = tmp_path / "state.h5"
    # allow_pickle=False is the default: it would raise if a pickle were needed.
    save_operators(filename, {"rho": state}, "w")

    def visit(_name, node):
        assert node.attrs.get("py_type", "") not in ("pickle", "pickled_operator")

    with h5py.File(filename, "r") as h5file:
        assert h5file["operators/rho"].attrs["type"] == "mps_gs"
        h5file.visititems(visit)
    assert load_operators(filename)["rho"].solved is False


def test_initial_state_of_the_dmrg_is_dropped_with_a_warning(tmp_path):
    system, ham = chain(6)
    seed_state = MPSGSOperator(ham, chi_max=4).mps
    state = MPSGSOperator(ham, chi_max=16, dmrg_options={"mps0": seed_state, "seed": 2})
    filename = tmp_path / "state.h5"
    with pytest.warns(IncompleteStateWarning, match="mps0"):
        save_operators(filename, {"rho": state})
    with h5py.File(filename, "r") as h5file:
        group = h5file["operators/rho"]
        assert list(group.attrs["dropped"]) == ["dmrg_options.mps0"]
        assert '"seed": 2' in group.attrs["dmrg_options"]
        assert "mps0" not in group.attrs["dmrg_options"]
    with pytest.warns(IncompleteStateWarning, match="mps0"):
        loaded = load_operators(filename)["rho"]
    assert loaded.dmrg_options == {"seed": 2}


def test_large_system_is_stored_compactly(tmp_path):
    """24 spins: the file holds the MPS, not a matrix of 2^24 x 2^24 entries."""
    system, ham = chain(24)
    order = [f"1[{i}]" for i in range(24)]
    state = MPSGSOperator(ham, permutation=order, chi_max=16).solve()
    filename, loaded = save_and_load(tmp_path, {"rho": state})
    assert os.path.getsize(filename) < 2_000_000
    rho = loaded["rho"]
    assert rho.energy == state.energy
    assert (
        abs(rho.expect(rho.system.global_operator("Hamiltonian")) - state.energy) < 1e-9
    )


@pytest.fixture(name="stored_state")
def fixture_stored_state(tmp_path):
    system, ham = chain(6)
    state = MPSGSOperator(ham, chi_max=16, dmrg_options={"seed": 1}).solve()
    filename = tmp_path / "state.h5"
    save_operators(filename, {"rho": state}, "w")
    return filename


def corrupt(filename, edit):
    with h5py.File(filename, "r+") as h5file:
        edit(h5file["operators/rho"])


def test_tensors_that_do_not_match_their_shapes(stored_state):
    def edit(group):
        data = group["mps/tensors"][()]
        del group["mps/tensors"]
        group["mps/tensors"] = data[:-1]

    corrupt(stored_state, edit)
    with pytest.raises(HDF5FormatError, match="number of entries"):
        load_operators(stored_state)


def test_invalid_shapes(stored_state):
    def edit(group):
        shapes = group["mps/shapes"][()]
        shapes[0, 0] = 0
        group["mps/shapes"][...] = shapes

    corrupt(stored_state, edit)
    with pytest.raises(HDF5FormatError, match="shapes"):
        load_operators(stored_state)


def test_bond_dimensions_that_do_not_match(stored_state):
    def edit(group):
        shapes = group["mps/shapes"][()]
        shapes[1, 0] += 1
        group["mps/shapes"][...] = shapes
        data = group["mps/tensors"][()]
        extra = int(shapes[1, 1] * shapes[1, 2])
        del group["mps/tensors"]
        group["mps/tensors"] = np.concatenate([data, np.zeros(extra, complex)])

    corrupt(stored_state, edit)
    with pytest.raises(HDF5FormatError, match="not a valid MPS state"):
        load_operators(stored_state)


def test_local_dimensions_that_do_not_match_the_system(stored_state):
    def edit(group):
        shapes = group["mps/shapes"][()]
        shapes[:, 1] = 3
        group["mps/shapes"][...] = shapes

    corrupt(stored_state, edit)
    with pytest.raises(HDF5FormatError):
        load_operators(stored_state)


def test_state_with_neither_mps_nor_hamiltonian(stored_state):
    def edit(group):
        del group["mps"]
        del group["hamiltonian"]

    corrupt(stored_state, edit)
    with pytest.raises(HDF5FormatError, match="neither an MPS nor a Hamiltonian"):
        load_operators(stored_state)


def test_order_that_is_not_a_permutation_of_the_sites(stored_state):
    def edit(group):
        order = group["order"][()]
        order[0] = order[1]
        group["order"][...] = order

    corrupt(stored_state, edit)
    with pytest.raises(HDF5FormatError):
        load_operators(stored_state)


def test_missing_required_attribute(stored_state):
    corrupt(stored_state, lambda group: group.attrs.__delitem__("chi_max"))
    with pytest.raises(HDF5FormatError, match="chi_max"):
        load_operators(stored_state)
