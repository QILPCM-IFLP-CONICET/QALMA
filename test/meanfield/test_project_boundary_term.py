"""Regression tests for ``project_boundary_term``."""

import numpy as np

from qalma.meanfield.gibbs_partial_trace import project_boundary_term
from qalma.model import build_spin_chain
from qalma.operators import QutipOperator
from qalma.operators.states import ProductDensityOperator


def test_project_boundary_term_qutip_operator_with_prefactor():
    """The prefactor of a ``QutipOperator`` term is applied once."""
    system = build_spin_chain(2)
    site_a, site_b = list(system.sites)
    bond = (
        system.site_operator("Sx", site_a) * system.site_operator("Sx", site_b)
    ).to_qutip_operator()
    assert isinstance(bond, QutipOperator)
    sx_a = system.site_operator("Sx", site_a).to_qutip((site_a,))
    sx_b = system.site_operator("Sx", site_b).to_qutip((site_b,))
    # local states with <Sx_a> = 0.4 and <Sx_b> = 0.3
    sigma = ProductDensityOperator(
        {
            site_a: 0.5 * (1 + 0.8 * 2 * sx_a),
            site_b: 0.5 * (1 + 0.6 * 2 * sx_b),
        },
        system=system,
    )
    expected_field = 0.3  # <Sx_b>
    for prefactor in (1.0, 10.0, -3.0):
        term = prefactor * bond
        projected = project_boundary_term(term, sigma, frozenset([site_a]))
        matrix = projected.to_qutip((site_a,)).full()
        expected = prefactor * expected_field * sx_a.full()
        assert np.allclose(matrix, expected, atol=1e-10), (prefactor, matrix)
