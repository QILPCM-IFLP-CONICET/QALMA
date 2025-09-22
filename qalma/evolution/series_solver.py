"""
Functions used to run MaxEnt simulations.
"""

from __future__ import annotations

from math import factorial

from qalma.evolution.hierarchical_basis import build_hierarchical_basis

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
    h_basis = build_hierarchical_basis(ham, k0, order)
    states = [
        sum((t) ** p * bb / factorial(p) for p, bb in enumerate(h_basis))
        for t in t_span
    ]
    return Simulation(
        parameters={"method": "series"},
        stats={},
        time_span=list(t_span),
        states=states,
        expect_ops={},
    )
