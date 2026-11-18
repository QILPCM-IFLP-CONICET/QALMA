"""
Conventions shared by the solvers of :mod:`qalma.evolution`.

Every solver returns a :class:`~qalma.evolution.simulation.Simulation` and
follows the same conventions:

* the signature is ``solver(ham, k0 | rho0, t_span, *, e_ops=None,
  store_states=None, ...)``, with every option given by keyword;
* ``e_ops`` is a dict of operators, a list of operators (stored with the keys
  ``0, 1, ...``), or a callable ``e_ops(t, rho)`` that receives the state at
  each output time;
* ``store_states=None`` stores the states only when ``e_ops`` is not given;
* ``parameters`` always contains ``"method"``, ``"system"`` and ``"states"``,
  which says what ``Simulation.states`` holds: ``"generator"`` (the
  generators ``K(t)``, with ``rho(t) ~ exp(-K(t))``), ``"density"`` (density
  operators) or ``"none"``;
* ``stats`` always contains ``"method"`` and ``"wall time"`` (in seconds),
  plus the diagnostics of each method.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Union, cast

from qalma.operators import Operator
from qalma.operators.states import GibbsDensityOperator

__all__ = ["normalize_e_ops", "resolve_store_states", "GeneratorRecorder"]

EOps = Union[Dict[Any, Operator], List[Operator], Callable, None]


def normalize_e_ops(e_ops: EOps) -> Union[Dict[Any, Operator], Callable, None]:
    """Return ``e_ops`` as a dict of operators, a callable, or None.

    A list or tuple of operators is converted to a dict with keys ``0, 1, ...``.
    """
    if e_ops is None or callable(e_ops) or isinstance(e_ops, dict):
        return e_ops
    return dict(enumerate(cast(Iterable, e_ops)))


def resolve_store_states(e_ops: EOps, store_states: Optional[bool]) -> bool:
    """Store the states by default only when no observables are requested."""
    return e_ops is None if store_states is None else bool(store_states)


class GeneratorRecorder:
    """Record the output of a solver that evolves generators ``K(t)``.

    At each output time, the generator ``k`` is stored (if ``store_states``)
    and the observables are evaluated in ``rho = exp(-k) / Tr exp(-k)``: a
    dict of operators gives the expectation values, stored in
    ``expect_ops``; a callable is called as ``e_ops(t, rho)``.
    """

    def __init__(
        self,
        e_ops: EOps,
        store_states: Optional[bool],
        states: List[Operator],
        expect_ops: Dict[Any, Any],
    ):
        """Set up the recorder for the given observables and storage."""
        self.e_ops = normalize_e_ops(e_ops)
        self.store_states = resolve_store_states(e_ops, store_states)
        self.states = states
        self.expect_ops = expect_ops

    def __call__(self, t: float, k: Operator) -> None:
        """Record the generator ``k`` at time ``t``."""
        if self.store_states:
            self.states.append(k)
        if self.e_ops is None:
            return
        rho = GibbsDensityOperator(k)
        if callable(self.e_ops):
            self.e_ops(t, rho)
            return
        values = cast(Dict[Any, Any], rho.expect(self.e_ops))
        for key, val in values.items():
            self.expect_ops.setdefault(key, []).append(val)
