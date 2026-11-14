"""Tests documenting the current treatment of fermionic models."""

import warnings

import numpy as np
import pytest

from qalma import FermionicSignWarning, build_system


def _spinless_ring(length):
    """Spinless free fermions (t=1) on a periodic chain."""
    return build_system(
        geometry_name="chain lattice",
        model_name="spinless fermions",
        L=length,
        t=1,
        V=0,
        mu=0,
    )


def test_fermionic_model_warns():
    """Building a system with fermionic operators emits a warning."""
    with pytest.warns(FermionicSignWarning):
        _spinless_ring(3)


def test_spin_model_does_not_warn():
    """Non-fermionic models do not emit the warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FermionicSignWarning)
        build_system(geometry_name="chain lattice", model_name="spin", L=3)


@pytest.mark.xfail(
    strict=True,
    reason="fermionic signs are not implemented: the ring behaves as hard-core bosons",
)
@pytest.mark.filterwarnings("ignore::qalma.FermionicSignWarning")
def test_free_fermions_on_a_ring():
    """Ground state energy of free fermions on a 4-site ring.

    The single particle energies are -2 cos(k) = -2, 0, 0, 2, so the ground
    state energy is -2. Without the fermionic signs, the result is the one
    for hard-core bosons, -2 sqrt(2).
    """
    ham = _spinless_ring(4).global_operator("Hamiltonian")
    assert np.isclose(min(ham.eigenenergies()), -2.0)
