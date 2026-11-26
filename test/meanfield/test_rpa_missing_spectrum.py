"""RPA of a site without local mean-field generator."""

import numpy as np

from qalma import build_system
from qalma.meanfield.rpa import RPAGibbsDensityOperator


def structure_of_a_small_chain():
    """RPA structure of a generator that couples two sites of a chain."""
    system = build_system("open chain lattice", "spin", L=4, J=1.0)
    sites = tuple(system.sites)
    sz0 = system.site_operator(f"Sz@{sites[0]}")
    sx0 = system.site_operator(f"Sx@{sites[0]}")
    sx1 = system.site_operator(f"Sx@{sites[1]}")
    generator = (0.5 * sz0 + 0.3 * sx0 * sx1).simplify()
    return sites, RPAGibbsDensityOperator(generator, system).rpa_structure


def test_site_without_contributions_is_maximally_mixed():
    """A site missing in the spectra gets the spectrum of the maximally mixed state."""
    sites, structure = structure_of_a_small_chain()
    structure.local_spectrum(sites[1])  # builds the dictionary of spectra
    structure._spectra.pop(sites[1])  # as if every projected term were zero
    populations, vectors = structure.local_spectrum(sites[1])
    assert np.allclose(populations, [0.5, 0.5])
    assert np.allclose(vectors, np.eye(2))
    freqs, couplings = structure.boson_modes(sites[1])
    assert len(freqs) == couplings.shape[1]
    # all the modes are static: their frequencies do not exceed the static gap
    assert np.all(freqs <= structure.static_gap + 1e-12)


def test_spectra_of_the_other_sites_are_unchanged():
    """The fallback applies only to the missing sites."""
    sites, structure = structure_of_a_small_chain()
    reference, _ = structure.local_spectrum(sites[0])
    structure._spectra.pop(sites[1], None)
    populations, _ = structure.local_spectrum(sites[0])
    assert np.allclose(populations, reference)
    assert not np.allclose(populations, [0.5, 0.5])
