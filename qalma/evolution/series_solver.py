"""Power-series solution of the equation of motion of the generator."""

from __future__ import annotations

from datetime import datetime
from functools import reduce
from typing import Any, Dict, List, Optional

from qalma.operators import Operator
from qalma.scalarprod.basis import HierarchicalOperatorBasis

from .common import EOps, GeneratorRecorder
from .simulation import Simulation

__all__ = ["series_evolution"]


def series_evolution(
    ham: Operator,
    k0: Operator,
    t_span,
    order: int,
    *,
    e_ops: EOps = None,
    store_states: Optional[bool] = None,
) -> Simulation:
    """Solve the Schrödinger equation for the generator as a power series.

    ``K(t) = K0 - i [H, K0] t - [H, [H, K0]] t**2 / 2 + ...``, truncated at
    ``order``.

    Parameters
    ----------
    ham : Operator
        The Hamiltonian.
    k0 : Operator
        The initial generator, ``rho(0) ~ exp(-k0)``.
    t_span : array-like
        Output times.
    order : int
        Order of the truncated series.
    e_ops : dict, list or callable, optional
        Observables evaluated in ``rho(t) ~ exp(-K(t))``, or a callable
        ``e_ops(t, rho)``.
    store_states : bool, optional
        Store the generators ``K(t)``. By default, only when ``e_ops`` is not
        given.

    Returns
    -------
    Simulation
        ``states`` holds generators; ``stats`` has ``"method"``,
        ``"wall time"``, ``"init time"`` and ``"run time"`` (in seconds).
    """
    start_time = datetime.now()
    states: List[Operator] = []
    expect_ops: Dict[Any, Any] = {}
    record = GeneratorRecorder(e_ops, store_states, states, expect_ops)

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
    init_time = (datetime.now() - start_time).total_seconds()

    run_start = datetime.now()
    for t in t_span:
        record(t, compute_inst_k(t, h_basis, prefactor))
    run_time = (datetime.now() - run_start).total_seconds()

    stats = {
        "method": "series",
        "wall time": (datetime.now() - start_time).total_seconds(),
        "init time": init_time,
        "run time": run_time,
    }
    return Simulation(
        parameters={
            "method": "series",
            "system": ham.system,
            "states": "generator" if record.store_states else "none",
            "order": order,
        },
        stats=stats,
        time_span=list(t_span),
        states=states,
        expect_ops=expect_ops,
    )
