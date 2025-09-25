"""
Functions used to run MaxEnt simulations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import qutip
from numpy.typing import ArrayLike
from qutip import Qobj

from qalma.operators import Operator, QutipOperator
from qalma.operators.states import QutipDensityOperator

from .simulation import Simulation


def qutip_me_solve(
    H: Operator,
    rho0: Operator,
    tlist: ArrayLike,
    *,
    c_ops: Optional[
        List[Operator] | dict[Any, Operator] | Callable[[float, "Qobj"], Any]
    ] = None,
    e_ops: Optional[
        list[Operator] | dict[Any, Operator] | Callable[[float, "Qobj"], Any]
    ] = None,
    args: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Simulation:
    """
    Compute the solution of the Schrödinger equation using qutip.mesolve

    Parameters
    ----------
    H : Operator
        Possibly time-dependent system Liouvillian or Hamiltonian as a Qobj or
        QobjEvo. List of [:obj:`.Qobj`, :obj:`.Coefficient`] or callable that
        can be made into :obj:`.QobjEvo` are also accepted.

    rho0 : Operator
        initial density matrix or state vector (ket).
    tlist : *list* / *array*
        list of times for :math:`t`.
    c_ops : list[Operator] | dict[Any, Operator]
            | Callable[[float, "Qobj"], Any], optional
        Single collapse operator, or list of collapse operators, or a list
        of Liouvillian superoperators. None (default) is equivalent to an empty
        list.
    e_ops : list[Operator] | dict[Any, Operator]
            | Callable[[float, "Qobj"], Any], optional
        Single operator, or list or dict of operators, for which to evaluate
        expectation values. Operator can be Qobj, QobjEvo or callables with the
        signature `f(t: float, state: Qobj) -> Any`.
    args : dict[str, Any], optional
        dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    options : dict[str, Any], optional
        Dictionary of options for the solver.

        - | store_final_state : bool
          | Whether or not to store the final state of the evolution in the
            result class.
        - | store_states : bool, None
          | Whether or not to store the state vectors or density matrices.
            On `None` the states will be saved if no expectation operators are
            given.
        - | normalize_output : bool
          | Normalize output state to hide ODE numerical errors. Only normalize
            the state if the initial state is already normalized.
        - | progress_bar : str {'text', 'enhanced', 'tqdm', ''}
          | How to present the solver progress.
            'tqdm' uses the python module of the same name and raise an error
            if not installed. Empty string or False will disable the bar.
        - | progress_kwargs : dict
          | kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - | method : str ["adams", "bdf", "lsoda", "dop853", "vern9", etc.]
          | Which differential equation integration method to use.
        - | atol, rtol : float
          | Absolute and relative tolerance of the ODE integrator.
        - | nsteps : int
          | Maximum number of (internally defined) steps allowed in one
          ``tlist`` step.
        - | max_step : float
          | Maximum length of one internal step. When using pulses, it
            should be less than half the width of the thinnest pulse.

        Other options could be supported depending on the integration method,
        see `Integrator <./classes.html#classes-ode>`_.

    Returns
    -------
    Simulation:
       A simulation object storing the parameters and results of the simulation.

    """
    system = None
    if isinstance(H, Operator):
        system = H.system
        h_qutip = H.to_qutip()
    else:
        h_qutip = H

    if isinstance(rho0, Operator):
        StateOperatorClass = (
            QutipDensityOperator if hasattr(rho0, "expect") else QutipOperator
        )
        if system is None:
            system = rho0.system
        rho0_qutip = rho0.to_qutip()
    else:
        # If rho0 is a Qobj, just return the Qutip output without changes.
        def StateOperatorClass(x, **kwargs):
            return x

        rho0_qutip = rho0

    if e_ops is not None:
        if isinstance(e_ops, dict):
            e_ops = {
                key: val if isinstance(val, Qobj) else val.to_qutip()
                for key, val in e_ops.items()
            }
        elif isinstance(e_ops, (tuple, list)):
            e_ops = [val if isinstance(val, Qobj) else val.to_qutip() for val in e_ops]
        elif hasattr(e_ops, "__call__"):
            e_ops_qutip = e_ops

            def e_ops_wrapper(t, rho):
                return e_ops_qutip(t, StateOperatorClass(rho, system=system))

            e_ops = e_ops_wrapper

    if c_ops is not None:
        if isinstance(c_ops, dict):
            c_ops = {
                key: val if isinstance(val, Qobj) else val.to_qutip()
                for key, val in c_ops.items()
            }
        elif isinstance(c_ops, (tuple, list)):
            c_ops = [val if isinstance(val, Qobj) else val.to_qutip() for val in c_ops]

    result = qutip.mesolve(
        h_qutip,
        rho0_qutip,
        tlist,
        c_ops=c_ops,
        e_ops=e_ops,
        options=options,
        args=args,
    )
    parameters = {**result.options}
    parameters["system"] = system

    return Simulation(
        time_span=result.times,
        stats=result.stats,
        expect_ops=result.e_data,
        states=[StateOperatorClass(state, system=system) for state in result.states],
        parameters=parameters,
    )
