"""RPA around a mean field with exactly vanishing magnetization.

The boundary terms of the generator project to zero when the mean field of the
environment has a vanishing expectation value of the coupled operators (for
instance, in the saturated paramagnetic phase of the transverse-field Ising
model). The subsystem remains coupled to the environment through the
fluctuations, so the RPA correction must still be applied, and it must not
depend on whether the mean-field magnetization is exactly zero or not.
"""

from functools import reduce

import numpy as np
import pytest

from qalma.meanfield.gibbs_partial_trace import gibbs_meanfield_generator
from qalma.meanfield.rpa import RPAGibbsDensityOperator
from qalma.model import build_system
from qalma.operators.states.gibbs import GibbsProductDensityOperator

GAMMA = 2.5
BETA = 10.0


def _setup(tilt):
    """TFIM chain and RPA state with a paramagnetic mean field.

    ``tilt`` is a staggered longitudinal field, in units of the transverse
    one, added to the generator of the mean field.
    """
    system = build_system(
        "open chain lattice", "spin", L=4, a=1, Jz0=1.0, Jxy0=0.0, Gamma=GAMMA
    )
    k_op = (BETA * system.global_operator("Hamiltonian")).simplify()
    names = sorted(system.sites, key=lambda name: int(name.split("[")[1][:-1]))
    terms = [
        (-BETA * GAMMA) * system.site_operator("Sx", name)
        + (-BETA * GAMMA * tilt * (-1) ** pos) * system.site_operator("Sz", name)
        for pos, name in enumerate(names)
    ]
    sigma_mf = GibbsProductDensityOperator(reduce(lambda a, b: a + b, terms))
    state = RPAGibbsDensityOperator(k_op, meanfield=sigma_mf.to_product_state())
    return state, frozenset(names[1:3])


def _reduced(state, sites, **options):
    block = tuple(sorted(sites))
    rho = state.partial_trace(sites, max_exact_sites=0, **options).to_qutip(block)
    return rho / rho.tr()


def _distance(rho, sigma):
    return 0.5 * float(np.abs(np.linalg.eigvalsh((rho - sigma).full())).sum())


def test_boundary_is_detected_with_vanishing_mean_field():
    """The coupling with the environment does not depend on the mean field."""
    state, sites = _setup(0.0)
    _, has_boundary = gibbs_meanfield_generator(state, sites)
    assert has_boundary


@pytest.mark.parametrize("dynamic", [False, True])
def test_rpa_does_not_depend_on_a_vanishing_magnetization(dynamic):
    """Exactly zero or tiny staggered magnetization give the same RPA state."""
    state_sym, sites = _setup(0.0)
    state_tilt, _ = _setup(1e-8)
    rho_sym = _reduced(state_sym, sites, dynamic=dynamic)
    rho_tilt = _reduced(state_tilt, sites, dynamic=dynamic)
    assert _distance(rho_sym, rho_tilt) < 1e-6


def test_dynamic_rpa_corrects_a_symmetric_mean_field():
    """The dynamical correction is not silently dropped."""
    state, sites = _setup(0.0)
    rho_mf = _reduced(state, sites, rpa=False)
    rho_dyn = _reduced(state, sites, dynamic=True)
    assert _distance(rho_dyn, rho_mf) > 1e-4
