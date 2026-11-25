"""Bosonization of a site whose fluctuation operators were all discarded."""

import numpy as np

from qalma.meanfield.rpa.utils import local_boson_modes


def test_empty_local_basis():
    """A site without fluctuation operators has no bosons, and no error."""
    sigma = np.eye(2) / 2
    freqs, couplings = local_boson_modes(sigma, [])
    assert freqs.shape == (0,)
    assert couplings.shape == (0, 0)


def test_empty_local_basis_with_spectrum():
    """The same holds when the spectrum of the local state is given."""
    populations = np.array([0.5, 0.5])
    freqs, couplings = local_boson_modes(
        np.eye(2) / 2, [], spectrum=(populations, np.eye(2))
    )
    assert len(freqs) == 0 and couplings.size == 0


def test_nonempty_basis_is_unchanged():
    """With operators, the modes are those of the Pauli fluctuations."""
    sigma = np.diag([0.8, 0.2])
    basis = [np.array([[0, 1], [1, 0]], dtype=complex)]
    freqs, couplings = local_boson_modes(sigma, basis)
    assert len(freqs) >= 1
    assert couplings.shape == (1, len(freqs))
