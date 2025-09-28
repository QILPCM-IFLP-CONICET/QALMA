"""
Functions used to run MaxEnt simulations.
"""

from __future__ import annotations

from datetime import datetime
from functools import reduce
from typing import Any, Callable, Dict, Iterable, Optional, cast

from qalma.operators.states import GibbsDensityOperator
from qalma.scalarprod.basis import HierarchicalOperatorBasis

from .simulation import Simulation


def series_evolution(
    ham, k0, t_span, order, *, e_ops: Optional[Callable | Dict] = None
) -> Simulation:
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
    process_state: Optional[Callable] = None
    e_ops_dict: Dict
    expect_ops: Dict[Any, complex] = {}
    stats: Dict = {"solver": "series solver"}

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

    def compute_and_store_expect(t, rho_t):
        for key, val in rho_t.expect(e_ops_dict).items():
            expect_ops.setdefault(key, []).append(val)

    if e_ops is not None:
        if hasattr(e_ops, "__call__"):
            process_state = e_ops
        else:
            if not isinstance(e_ops, dict):
                e_ops_dict = {
                    pos: e_op for pos, e_op in enumerate(cast(Iterable, e_ops))
                }
            else:
                e_ops_dict = e_ops
            process_state = compute_and_store_expect

    start_time = datetime.now()
    h_basis = HierarchicalOperatorBasis(k0, ham, order)
    prefactor = h_basis.sp(h_basis.operator_basis[0], k0)
    states = []
    finish_time = datetime.now()
    stats["init time"] = finish_time - start_time

    start_time = datetime.now()
    for t in t_span:
        k = compute_inst_k(t, h_basis, prefactor)
        if process_state is None:
            states.append(k)
            continue
        rho = GibbsDensityOperator(k)
        process_state(t, rho)
    finish_time = datetime.now()
    stats["run time"] = finish_time - start_time

    return Simulation(
        parameters={"method": "series", "system": ham.system},
        stats=stats,
        time_span=list(t_span),
        states=states,
        expect_ops=expect_ops,
    )
