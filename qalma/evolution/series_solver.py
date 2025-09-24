"""
Functions used to run MaxEnt simulations.
"""

from __future__ import annotations

from functools import reduce

from qalma.scalarprod.basis import HierarchicalOperatorBasis

from .simulation import Simulation


def series_evolution(ham, k0, t_span, order) -> Simulation:
    """
    Compute the solution of the Schrödinger equation

    dk
    -- = -i [H, k]
    dt

    as a linear combination of the iterated commutators

    k = k0 - i [H, k0] t - [H,[H, k0]] t**2 - ...

    Parameters
    ----------
    ham : Operator
        The Hamiltonian operator
    k0 : Operator
        The initial condition
    t_span: np.array
        the times for with the evolution is computed
    order:
        the order of the solution

    Returns
    -------
    Simulation:
        A simulation object.

    """
    # TODO: implement me for time-dependent hamiltonians
    # and solve in the interaction picture

    def compute_inst_k(t, basis, prefactor):
        norms = [row[pos] for pos, row in enumerate(basis.gen_matrix[1:])]
        t_coeffs = [
            reduce(
                lambda x, y: x * y,
                (t / (k + 1) * norm for k, norm in enumerate(norms[:n])),
                prefactor,
            )
            for n in range(len(basis.operator_basis))
        ]
        return basis.operator_from_coefficients(t_coeffs)

    h_basis = HierarchicalOperatorBasis(k0, ham, order)
    prefactor = h_basis.sp(h_basis.operator_basis[0], k0)
    states = [compute_inst_k(t, h_basis, prefactor) for t in t_span]
    return Simulation(
        parameters={"method": "series", "system": ham.system},
        stats={},
        time_span=list(t_span),
        states=states,
        expect_ops={},
    )
