"""
Functions used to run MaxEnt simulations.
"""

from __future__ import annotations

import logging
import pickle
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from qalma.meanfield import (
    variational_quadratic_mfa,
)
from qalma.operators import (
    LocalOperator,
    OneBodyOperator,
    Operator,
    QuadraticFormOperator,
    ScalarOperator,
    SumOperator,
)
from qalma.operators.states import GibbsDensityOperator, GibbsProductDensityOperator
from qalma.projections import n_body_projection
from qalma.scalarprod import (
    HierarchicalOperatorBasis,
    OperatorBasis,
    fetch_covar_scalar_product,
)

from .simulation import Simulation

# function used to safely and robustly map K-states to states


def compute_mean_field_state(k, sigma, **kwargs):
    sigma_result = variational_quadratic_mfa(k, sigma_ref=sigma)
    generator = -sigma_result.logm()
    return generator, sigma_result


def compute_n_body_sector(k: Operator):
    if isinstance(k, SumOperator):
        return max(compute_n_body_sector(term) for term in k.terms)
    if isinstance(k, ScalarOperator):
        return 0
    if isinstance(k, (LocalOperator, OneBodyOperator)):
        return 1
    if isinstance(k, QuadraticFormOperator):
        if k.offset is None:
            return 2

    return len(k.acts_over())


def occupation_factor(phi: NDArray, threshold: float = 0.995) -> int:
    """
    Compute an estimation of how spread is the operator over the basis.

    Return the number of terms in the partial sum  of the squared modules
    of the components of `phi` required to reach the `threshold`.

    Parameters
    ----------

    phi: NDArray
       an array of numerical coefficients.
    threshold: float
       the threshold value for the partial sums.

    Return
    ------
    int:
    num of terms in the partial sum which reach the threshold.

    """
    partial_sums = np.array(
        [sum(np.abs(phi[:i]) ** 2) ** 0.5 for i in range(1, len(phi))]
    )
    partial_sums = partial_sums / partial_sums[-1]
    for idx, val in enumerate(partial_sums):
        if val > threshold:
            return idx + 1
    return len(phi)


def update_basis(
    k, sigma, ham, order, n_body, extra_observables
) -> Tuple[HierarchicalOperatorBasis, Operator, Operator]:
    k_ref_new, sigma = compute_mean_field_state(k, sigma)
    new_basis = HierarchicalOperatorBasis(
        k,
        ham,
        order,
        fetch_covar_scalar_product(sigma),
        n_body_projection=lambda op_b: n_body_projection(
            op_b, nmax=n_body, sigma=sigma
        ),
    )
    rest_elements = tuple(extra_observables)
    if k is not k_ref_new:
        rest_elements = rest_elements + (k_ref_new,)
    if rest_elements:
        new_basis = rest_elements + new_basis

    k_ref_new = k_ref_new + sigma.expect(k - k_ref_new)
    return (
        new_basis,
        sigma,
        k_ref_new,
    )


def update_basis_light(k, sigma, ham, order, n_body, extra_observables):
    k_ref_new, sigma = compute_mean_field_state(k, sigma)
    new_basis = HierarchicalOperatorBasis(
        k_ref_new,
        ham,
        order,
        fetch_covar_scalar_product(sigma),
    )

    rest_elements = tuple(extra_observables)
    if k is not k_ref_new:
        rest_elements = rest_elements + (k,)
    if rest_elements:
        new_basis = rest_elements + new_basis
    return (
        new_basis,
        sigma,
        k_ref_new,
    )


def adaptive_projected_evolution(
    ham,
    k0,
    t_span,
    order,
    n_body: int = -1,
    tol=1e-3,
    *,
    e_ops: Optional[Dict | List | Callable] = None,
    on_update_basis_callback: Optional[Callable] = None,
    extra_observables: Tuple[Operator, ...] = tuple(),
    include_one_body_projection: bool = False,
    basis_update_callback: Callable[
        ..., Tuple[OperatorBasis, Operator, Operator]
    ] = update_basis,
) -> Simulation:
    """
    Compute the solution of the MaxEnt projected Schrödinger equation

    dk
    -- = -i [H, k]
    dt

    as a linear combination of a an operator basis

    k = sum phi_a(t) Q_a

    chosen adaptively along the evolution.

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

    n_body: int
        if non-negative, build a solution projected on
        the subspace of n_body operators.

    tol: float
        the maximum induced distance between the projected solution
        and the exact solution.

    basis_update_callback: Callable[..., Tuple[OperatorBasis, Operator]]
        the function that creates a new basis from the current
        state of the simulation. The default value is the function `update_basis`
        which generates a hierarchical basis from the current generator `K(t)`,
        projecting on each step to the `n_body` sector.
        Alternatively, `update_basis_light` is a light-weight version of `update_basis`
        which uses as the seed the self-consistent mean field approximation of K(t).

    on_update_basis_callback: Callable[dict], optional
        if not None, this function is called each time the basis is rebuilt.

    Returns
    -------
    Simulation:
        A Simulation object storing the results of the simulation.
    """
    checkpoint_name = f"__adaptative_{order}_{n_body}_{uuid.uuid4()}.pkl"
    t_0 = t_span[0]

    errors: List[float] = []
    expect_ops: Dict[Any, Operator] = {}
    local_evol_parms: Dict = {"t_ref": t_0, "last_t": t_0, "away": 0, "error": 0}
    oc_factors: List[float] = []
    saturated_tolerance: bool = False
    states: List[Operator] = []
    t_max = t_span[-1]
    tlist: List[float] = []

    parameters: Dict[str, Any] = {
        "n_body": n_body,
        "order": order,
        "tol": tol,
        "include_one_body_projection": include_one_body_projection,
        "basis_update_callback": basis_update_callback.__name__,
        "system": ham.system,
    }
    stats: Dict[str, Any] = {
        "method": "Adaptative Restricted Evolution",
        "errors": errors,
        "t_update_basis": [],
        "basis time costs": [],
        "occupation factor": oc_factors,
        "away_from_ref": [],
        "n_body_sector": [],
        "update_times": [],
    }
    simulation = Simulation(
        parameters=parameters,
        stats=stats,
        time_span=tlist,
        expect_ops=expect_ops,
        states=states,
    )

    ### Handle e_ops #####
    if e_ops is None:

        def call_on_success_evol(t, k):
            states.append(k)

    elif hasattr(e_ops, "__call__"):
        call_on_success_evol = cast(Callable, e_ops)
    else:
        if not isinstance(e_ops, dict):
            e_ops = {pos: e_op for pos, e_op in enumerate(cast(Iterable, e_ops))}

        def call_on_success_evol(t, k):
            curr_e_ops = GibbsDensityOperator(k).expect(e_ops)
            for key, val in curr_e_ops.items():
                expect_ops.setdefault(key, []).append(val)

    ####  Basis update ########
    def call_update_basis(local_evol_parms) -> bool:
        k_t = local_evol_parms["k_t"]
        start_basis_time = datetime.now()
        basis, sigma_ref, k_ref = basis_update_callback(
            k_t,
            local_evol_parms["sigma_ref"],
            ham,
            order,
            local_evol_parms["curr_n_body"],
            extra_observables,
        )
        build_basis_time_cost = datetime.now() - start_basis_time
        local_evol_parms["basis time cost"] = build_basis_time_cost.seconds
        local_evol_parms["sigma_ref"] = sigma_ref
        local_evol_parms["k_ref"] = k_ref
        local_evol_parms["basis"] = basis
        local_evol_parms["t_ref"] = local_evol_parms["last_t"]
        local_evol_parms["phi_0"] = basis.coefficient_expansion(k_t)

        if on_update_basis_callback is not None:
            on_update_basis_callback(local_evol_parms)
        return True

    # Initialize
    local_evol_parms["t"] = 0
    local_evol_parms["k_t"] = k0
    local_evol_parms["curr_n_body"] = compute_n_body_sector(k0)
    local_evol_parms["sigma_ref"] = GibbsProductDensityOperator({}, k0.system)
    local_evol_parms["max_error_speed"] = tol / t_max
    tlist.append(local_evol_parms["t_ref"])
    logging.info(f"max_error_speed:{local_evol_parms['max_error_speed']}")

    # Create the first base
    call_update_basis(local_evol_parms)
    stats["update_times"].append(t)
    stats["t_update_basis"].append(t_0)
    stats["basis time costs"].append(local_evol_parms["basis time cost"])
    stats["n_body_sector"].append(local_evol_parms["curr_n_body"])
    # Perform tasks that follows to a success evolution:
    call_on_success_evol(t_0, local_evol_parms["k_t"])
    away = local_evol_parms["basis"].operator_norm(
        (local_evol_parms["k_t"] - local_evol_parms["k_ref"]).simplify()
    )
    local_evol_parms["away"] = away
    stats["away_from_ref"].append(away)
    stats["occupation factor"].append(occupation_factor(local_evol_parms["phi_0"]))

    # Main loop
    for t in t_span[1:]:
        local_evol_parms["t"] = t
        # If K_t is too far from K_ref, update the basis:
        if away > tol:
            logging.info("updating K_ref")
            call_update_basis(local_evol_parms)

        delta_t = t - local_evol_parms["t_ref"]
        phi, error = local_evol_parms["basis"].evolve(
            delta_t, local_evol_parms["phi_0"]
        )

        # If the error is growing faster that the acceptable rate,
        # try to enlarge the n-body sector:
        while error > local_evol_parms["max_error_speed"] * delta_t:
            local_evol_parms["error"] = error
            call_update_basis(local_evol_parms)
            delta_t = t - local_evol_parms["t_ref"]
            phi, error = local_evol_parms["basis"].evolve(
                delta_t, local_evol_parms["phi_0"]
            )

            if error <= local_evol_parms["max_error_speed"] * delta_t:
                stats["update_times"].append(t)
                stats["t_update_basis"].append(t)
                stats["basis time costs"].append(local_evol_parms["basis time cost"])
                stats["n_body_sector"].append(local_evol_parms["curr_n_body"])
                break
            if n_body > local_evol_parms["curr_n_body"]:
                local_evol_parms["curr_n_body"] += 1
                logging.warning(
                    f"tolerance goal cannot be reached within this subspace. Try in {local_evol_parms['curr_n_body']} sector."
                )
                continue
            #
            saturated_tolerance = True
            logging.warning(f"tolerance goal cannot be reached within {n_body} sector.")
            break

        if saturated_tolerance:
            logging.warning("tolerance goal cannot be reached within this subspace.")
            break

        tlist.append(t)
        local_evol_parms["last_t"] = t
        local_evol_parms["k_t"] = local_evol_parms["basis"].operator_from_coefficients(
            phi
        )
        stats["errors"].append(error)
        local_evol_parms["error"] = error
        call_on_success_evol(t, local_evol_parms["k_t"])
        stats["n_body_sector"].append(local_evol_parms["curr_n_body"])
        stats["occupation factor"].append(occupation_factor(phi))
        away = local_evol_parms["basis"].operator_norm(
            (local_evol_parms["k_t"] - local_evol_parms["k_ref"]).simplify()
        )
        local_evol_parms["away"] = away
        stats["away_from_ref"].append(away)

        # Dump the simulation state
        with open(checkpoint_name, "wb") as f:
            pickle.dump(simulation, f)

    return simulation


def projected_evolution(ham, k0, t_span, order, n_body: int = -1) -> Simulation:
    """
    Compute the solution of the MaxEnt projected Schrödinger equation

    dk
    -- = -i [H, k]
    dt

    as a linear combination of the iterated commutators

    k = sum phi_a(t) Q_a

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
    n_body: int
        if non-negative, build a solution projected on
        the subspace of n_body operators.

    Returns
    -------
    Simulation:
        A simulation object with the results of the simulation.

    """
    sigma_0 = GibbsProductDensityOperator(k0)
    sp = fetch_covar_scalar_product(sigma_0)

    basis = HierarchicalOperatorBasis(
        k0,
        ham,
        order,
        sp,
        n_body_projection=lambda op_b: n_body_projection(
            op_b, nmax=n_body, sigma=sigma_0
        ),
    )
    phi_0 = basis.coefficient_expansion(k0)
    errors = []
    states = []
    for t in t_span:
        phi, error = basis.evolve(t, phi_0)
        errors.append(error)
        states.append(basis.operator_from_coefficients(phi))

    return Simulation(
        parameters={"n_body": n_body, "order": order, "system": ham.system},
        stats={"method": "Static Projected Evolution", "errors": errors},
        time_span=t_span,
        expect_ops={},
        states=states,
    )
