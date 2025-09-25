"""
Functions used to run MaxEnt simulations.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from qalma.meanfield import (
    variational_quadratic_mfa,
)
from qalma.operators import Operator
from qalma.operators.states import GibbsProductDensityOperator
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
    away = new_basis.operator_norm(k - k_ref_new)
    print("new reference is ", away, " away from k")

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
    basis_costs: List[float] = []
    last_t = t_ref = t_span[0]
    t_max = t_span[-1]
    t_update_basis = [t_span[0]]
    max_error_speed = tol / t_max
    tlist: List[float] = [t_ref]
    errors: List[float] = []
    away_from_ref = []
    update_times = []

    logging.info(f"max_error_speed:{max_error_speed}")

    k_t = k0
    start_basis_time = datetime.now()
    basis, sigma_ref, k_ref = basis_update_callback(
        k_t,
        GibbsProductDensityOperator({}, k_t.system),
        ham,
        order,
        n_body,
        extra_observables,
    )
    build_basis_time_cost = datetime.now() - start_basis_time
    basis_costs.append(build_basis_time_cost.seconds)
    phi_0 = basis.coefficient_expansion(k_t)
    logging.info(f"phi_0={phi_0}")
    result = [k_t]
    away = basis.operator_norm((k_t - k_ref).simplify())
    for t in t_span[1:]:
        delta_t = t - t_ref
        phi, error = basis.evolve(delta_t, phi_0)
        oc_factor = occupation_factor(phi)
        away = basis.operator_norm((k_t - k_ref).simplify())
        away_from_ref.append(away)
        print("away:", away)
        if error > max_error_speed * delta_t:  # or away > 5*tol:
            logging.info(
                (
                    f"At time {t} the estimated error {error} "
                    f"is beyond the expected limit{max_error_speed * delta_t}. Updating basis."
                )
            )
            update_times.append(t)
            start_basis_time = datetime.now()
            basis, sigma_ref, k_ref = basis_update_callback(
                k_t, sigma_ref, ham, order, n_body, extra_observables
            )
            build_basis_time_cost = datetime.now() - start_basis_time
            basis_costs.append(build_basis_time_cost.seconds)

            phi_0 = basis.coefficient_expansion(k_t)
            t_ref = last_t
            delta_t = t - t_ref
            phi, error = basis.evolve(delta_t, phi_0)

            if on_update_basis_callback is not None:
                on_update_basis_callback(
                    state={
                        "phi_0": phi_0,
                        "phi": phi,
                        "error": error,
                        "delta_t": delta_t,
                        "t_ref": t_ref,
                        "t": t,
                        "basis": basis,
                        "K_t": k_t,
                        "basis time cost": build_basis_time_cost,
                        "oc_factor": oc_factor,
                    }
                )
            if error > max_error_speed * delta_t:
                logging.warning(
                    "tolerance goal cannot be reached within this subspace."
                )
                break
            t_update_basis.append(t)
        k_t = basis.operator_from_coefficients(phi)
        result.append(k_t)
        errors.append(error)
        tlist.append(t)
        last_t = t

    return Simulation(
        parameters={
            "n_body": n_body,
            "order": order,
            "tol": tol,
            "include_one_body_projection": include_one_body_projection,
            "basis_update_callback": basis_update_callback.__name__,
            "away_from_ref": away_from_ref,
            "errors": errors,
            "update_times": update_times,
            "system": ham.system,
        },
        stats={
            "method": "Adaptative Restricted Evolution",
            "errors": errors,
            "t_update_basis": t_update_basis,
            "basis_costs": basis_costs,
        },
        time_span=tlist,
        expect_ops={},
        states=result,
    )


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
