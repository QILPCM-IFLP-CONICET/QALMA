"""RPA of states with local gaps ``beta*Delta`` beyond the range of a double.

The populations of a local mean-field state are ``exp(-beta*Delta)``, which
underflow when ``beta*Delta`` exceeds ``~745``. The gaps ``ln(p_i/p_j)`` that
determine the bosonic frequencies and the susceptibilities must be taken from
the logarithms of the populations, not from the (clipped) populations.
"""

from functools import reduce

import numpy as np
import pytest

from qalma.meanfield.rpa import RPAGibbsDensityOperator
from qalma.meanfield.rpa.utils import kubo_mori_susceptibility, local_boson_modes
from qalma.model import build_system
from qalma.operators.states.gibbs import GibbsProductDensityOperator

GAMMA = 2.5
SX = np.array([[0.0, 0.5], [0.5, 0.0]])


def test_boson_frequency_follows_a_huge_gap():
    """A gap of 1600 (``p_1 = e^-1600`` underflows) is not clipped at 690.8."""
    gap = 1600.0
    spectrum = (np.array([1.0, 0.0]), np.eye(2))
    freqs, _ = local_boson_modes(
        np.diag(spectrum[0]),
        [SX],
        spectrum=spectrum,
        log_populations=np.array([0.0, -gap]),
    )
    assert freqs == pytest.approx([gap], rel=1e-12)


def test_susceptibility_follows_a_huge_gap():
    """``chi^0 = 2 |<0|e|1>|^2 / gap`` for ``p_0 = 1`` and ``p_1 = e^-gap``."""
    gap = 1600.0
    spectrum = (np.array([1.0, 0.0]), np.eye(2))
    chi = kubo_mori_susceptibility(
        np.diag(spectrum[0]),
        [SX],
        spectrum=spectrum,
        log_populations=np.array([0.0, -gap]),
    )
    assert chi[0, 0] == pytest.approx(2 * 0.25 / gap, rel=1e-12)


def _paramagnetic_state(geometry, beta, **lattice):
    """TFIM and RPA state around the (exactly symmetric) paramagnetic mean field."""
    params = {"a": 1, "Jz0": 1.0, "Jxy0": 0.0, "Gamma": GAMMA, **lattice}
    if geometry == "open ladder":
        params.update(Jz1=1.0, Jxy1=0.0)
    system = build_system(geometry, "spin", **params)
    k_op = (beta * system.global_operator("Hamiltonian")).simplify()
    generator = reduce(
        lambda a, b: a + b,
        [(-beta * GAMMA) * system.site_operator("Sx", name) for name in system.sites],
    )
    sigma_mf = GibbsProductDensityOperator(generator).to_product_state()
    return system, RPAGibbsDensityOperator(k_op, meanfield=sigma_mf)


def test_local_quantities_scale_with_the_temperature():
    """Frequencies grow as ``beta`` and susceptibilities decay as ``1/beta``.

    Both are beyond the clipping of the populations for ``beta*Gamma > 690``.
    """
    modes, chi0 = {}, {}
    for beta in (640.0, 1280.0):
        system, state = _paramagnetic_state("open chain lattice", beta, L=4)
        site = sorted(system.sites)[1]
        structure = state.rpa_structure
        modes[beta] = np.max(structure.boson_modes(site)[0])
        chi0[beta] = structure.static_chi0(site)[0, 0]
    assert modes[1280.0] / modes[640.0] == pytest.approx(2.0, rel=1e-9)
    assert chi0[1280.0] / chi0[640.0] == pytest.approx(0.5, rel=1e-9)
    assert modes[640.0] == pytest.approx(640.0 * GAMMA, rel=1e-9)


def _pair_state(system, state, dynamic):
    names = sorted(system.sites)
    pair = frozenset(names[3:5])
    rho = state.partial_trace(pair, max_exact_sites=0, dynamic=dynamic)
    return rho.to_qutip(tuple(sorted(pair)))


def test_static_rpa_converges_in_the_temperature_of_a_gapped_state():
    """The static RPA does not depend on ``beta`` once the state is gapped.

    With the populations clipped, ``chi^0`` stops decreasing as ``1/beta`` and
    the result drifts with ``beta``.
    """
    results = []
    for beta in (1000.0, 2000.0):
        system, state = _paramagnetic_state("open ladder", beta, L=3, W=3)
        results.append(_pair_state(system, state, dynamic=False))
    assert (results[0] - results[1]).norm("tr") < 1e-6


@pytest.mark.parametrize("dynamic", [False, True])
def test_rpa_is_not_unstable_at_very_low_temperature(dynamic):
    """``1 + chi^0 M`` stays positive definite as ``beta -> infinity``.

    The clipped ``chi^0`` stops decreasing as ``1/beta`` while ``M`` grows as
    ``beta``, and the criterion fails at low enough temperature although the
    paramagnetic state is far from the critical point.
    """
    system, state = _paramagnetic_state("open ladder", 2000.0, L=3, W=3)
    rho = _pair_state(system, state, dynamic)
    assert rho.tr() == pytest.approx(1.0, abs=1e-9)
