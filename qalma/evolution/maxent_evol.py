r"""
Adaptive restricted Max-Ent evolution.

:func:`adaptive_maxent_evolution` propagates the generator
:math:`\tilde K(t)=\sum_a\phi_a(t)Q_a` of :math:`\rho(t)\propto e^{-\tilde K(t)}`
inside a hierarchical operator basis built from :math:`\tilde K` by iterated
commutators with :math:`H`, projected onto the n-body sector
(:func:`update_basis`). The scalar product is the covariance product relative
to a mean-field reference state, which makes the equations of motion linear
between rebuilds of the basis.

The first-order restricted evolution, :math:`\dot\phi=A\phi` with
:math:`A=G^{-1}h`, discards the feedback of the part of the dynamics that
leaves the span of the basis and is almost isentropic, while the
instantaneous Max-Ent projection :math:`\Pi_n\rho(t)` that it approximates
produces entropy, :math:`S(\rho\|\Pi_n\rho)=S(\Pi_n\rho)-S(\rho)`. Projecting
:math:`\ddot K=L^2K` (:math:`L=-i[H,\cdot]`) onto the basis restores that
feedback; with a memory kernel that decays in a time :math:`\tau`,

.. math::

    \dot\phi = A\phi + m, \qquad \dot m = -G^{+}E\,\phi - m/\tau,
    \qquad m(t_0) = 0 ,

where :math:`E` is the leakage Gram matrix
(:func:`~qalma.evolution.leakage_gram`). This system is continuous in time;
it reduces to the first-order evolution for :math:`\tau\to0` and to the
projected second-order equation :math:`\ddot\phi=(A^2-G^{+}E)\phi` for
:math:`\tau\to\infty`. :math:`\tau` is estimated from the short-time decay of
the leaked part of the dynamics
(:func:`~qalma.evolution.estimate_memory_time`).

The basis and the reference state are rebuilt when the leakage rate exceeds a
multiple of its value right after the last rebuild, or when the local drift
of the reference state exceeds a tolerance. At each rebuild the previous
velocity is added to the new basis, so that :math:`\dot\phi` is continuous.
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from qalma.meanfield import variational_quadratic_mfa
from qalma.operators import Operator
from qalma.operators.states import GibbsProductDensityOperator, ProductDensityOperator
from qalma.projections import n_body_projection
from qalma.scalarprod import (
    HierarchicalOperatorBasis,
    OperatorBasis,
    fetch_covar_scalar_product,
    trim_terms_by_tolerance,
)

from .common import EOps, GeneratorRecorder
from .leakage_control import (
    _norm,
    estimate_memory_time,
    leakage_gram,
    leakage_rate,
    local_mean_field_drift,
    sector_leakage_fraction,
)
from .simulation import Simulation

logger = logging.getLogger(__name__)

__all__ = [
    "adaptive_maxent_evolution",
    "compute_mean_field_state",
    "trim_and_project_function",
    "update_basis",
]

#: Above this sector-leakage fraction a warning recommends a larger ``n_body``.
SECTOR_LEAKAGE_WARNING = 0.9


def compute_mean_field_state(
    k: Operator, sigma: ProductDensityOperator, **kwargs
) -> Tuple[Operator, ProductDensityOperator]:
    """Build the generator associated to the mean field state.

    Parameters
    ----------
    k: Operator
       the generator to be approximated by a OneBodyOperator
    sigma: ProductDensityOperator
       A reference state to start the search of a generator

    Return
    ------

    generator: Operator
       The computed generator

    sigma_result: ProductDensityOperator
       The associated state.

    """
    sigma_result = variational_quadratic_mfa(k, sigma_ref=sigma)
    generator = -sigma_result.logm()
    assert generator.isherm, "generator should be hermitian"
    return generator, sigma_result


def update_basis(
    k, sigma, ham, order, n_body, extra_observables, k_ref=None
) -> Tuple[OperatorBasis, Operator, Operator]:
    """Build a hierarchical operator basis adapted to the current state ``k``.

    Computes the mean-field approximation of ``k`` to define a reference
    generator ``k_ref``, then constructs a :class:`HierarchicalOperatorBasis`
    using the covariant scalar product with respect to ``sigma``. The basis
    is projected onto the ``n_body`` sector at each hierarchical level.

    Parameters
    ----------
    k : Operator
        The current generator :math:`K(t)`.
    sigma : ProductDensityOperator
        The current reference state, used to define the scalar product
        and the mean-field approximation.
    ham : Operator
        The Hamiltonian, used to build the hierarchical basis via iterated
        commutators.
    order : int
        The order of the hierarchical basis (number of commutator levels).
    n_body : int
        Maximum n-body sector for the projection. Negative values disable
        the projection.
    extra_observables : Iterable[Operator]
        Additional operators to append to the basis.
    k_ref : Operator or None, optional
        If provided, skip the mean-field computation and use this as the
        reference generator directly.

    Returns
    -------
    new_basis : OperatorBasis
        The updated hierarchical basis.
    sigma : ProductDensityOperator
        The updated mean-field reference state.
    k_ref_new : Operator
        The reference generator used to seed the basis.

    """
    if k_ref is None:
        k_ref_new, sigma = compute_mean_field_state(k, sigma)
        k_ref_new = k_ref_new + sigma.expect(k - k_ref_new)
        k_ref_new = k_ref_new.hermitian_part()
    else:
        k_ref_new = k_ref

    new_basis = HierarchicalOperatorBasis(
        k,
        ham,
        order,
        fetch_covar_scalar_product(sigma),
        n_body_projection=trim_and_project_function(sigma, n_body, tol=1e-9),
    )
    rest_elements = tuple(extra_observables)
    if k is not k_ref_new and not _is_in_span(new_basis, k_ref_new):
        rest_elements = rest_elements + (k_ref_new,)

    if rest_elements:
        new_basis = new_basis + rest_elements

    return (
        new_basis,
        sigma,
        k_ref_new,
    )


def _is_in_span(basis: OperatorBasis, operator: Operator, rtol: float = 1e-6) -> bool:
    """Check whether ``operator`` is numerically in the span of ``basis``.

    Appending such an operator makes the Gram matrix singular, and the
    coefficients of the evolution ill-defined.
    """
    norm_sq = np.real(basis.sp(operator, operator))
    if norm_sq <= 0:
        return True
    coeffs = basis.coefficient_expansion(operator)
    par_sq = np.real(coeffs @ basis.gram @ coeffs)
    return norm_sq - par_sq <= rtol**2 * norm_sq


def trim_and_project_function(sigma, n_body, tol=1e-6):
    """Build a projection function that trims and projects operators.

    Returns a callable that, given an operator, first projects it onto the
    ``n_body`` sector and then removes terms whose contribution is below
    ``tol`` relative to the scalar product defined by ``sigma``.

    Parameters
    ----------
    sigma : ProductDensityOperator
        Reference state defining the covariant scalar product used for
        trimming.
    n_body : int
        Maximum n-body sector for the projection.
    tol : float, optional
        Tolerance for trimming small terms. Default is ``1e-6``.

    Returns
    -------
    Callable[[Operator], Operator]
        A function that accepts an :class:`Operator` and returns its
        projected and trimmed version.

    """

    def trim_and_project(op_b):
        """Project ``op_b`` onto the n-body sector and trim small terms.

        Parameters
        ----------
        op_b : Operator
            The operator to project and trim.

        Returns
        -------
        Operator
            The projected and trimmed operator, enforcing Hermiticity if
            the input was Hermitian.

        """
        logger.debug(
            "trim and project %s terms in the %s sector up to tol=%s",
            op_b.num_terms(),
            op_b.n_body_sector(),
            tol,
        )
        isherm = op_b.isherm
        op_b = op_b.simplify()
        op_b = n_body_projection(op_b, n_max=n_body, sigma=sigma).simplify()
        op_b = trim_terms_by_tolerance(sigma, op_b, tol)
        if isherm:
            op_b = op_b.hermitian_part()
        else:
            logger.debug("basis element of type %s is not hermitian", type(op_b))
        return op_b

    return trim_and_project


def _segment_block(gen: NDArray, mem_mat: NDArray, tau: Optional[float]) -> NDArray:
    """Return the generator of (phi, m) in a segment."""
    size = len(gen)
    if tau is None:
        return np.block([[gen, np.zeros((size, size))], [np.zeros((size, 2 * size))]])
    return np.block([[gen, np.eye(size)], [-mem_mat, -np.eye(size) / tau]])


def _propagate(segment: Dict[str, Any], tau: float, e_mat, n_quad: int):
    """Propagate a segment and integrate the leakage over ``[0, tau]``.

    Returns phi, m, the leakage rate at ``tau``, the integral of the leakage
    rate and the integral of phi.
    """
    size = len(segment["phi_0"])
    y_0 = np.concatenate([segment["phi_0"], segment["m_0"]])
    if tau <= 0:
        phi = segment["phi_0"]
        return phi, segment["m_0"], leakage_rate(e_mat, phi), 0.0, np.zeros(size)
    n_quad = max(int(n_quad), 1)
    step = expm((tau / n_quad) * segment["block"])
    states = [y_0]
    for _ in range(n_quad):
        states.append(step @ states[-1])
    phis = np.array([y[:size] for y in states])
    weights = np.full(n_quad + 1, tau / n_quad)
    weights[0] = weights[-1] = 0.5 * tau / n_quad
    rates = np.array([leakage_rate(e_mat, phi) for phi in phis])
    return (
        phis[-1],
        states[-1][size:],
        float(rates[-1]),
        float(weights @ rates),
        weights @ phis,
    )


def adaptive_maxent_evolution(
    ham: Operator,
    k0: Operator,
    t_span,
    order: int,
    n_body: int = 2,
    *,
    memory_time: Union[float, str, None] = "auto",
    default_memory_time: float = 0.5,
    adaptive: bool = True,
    rate_factor: float = 2.0,
    rate_min: float = 0.05,
    drift_tol: float = 0.01,
    continuous_velocity: bool = True,
    n_quad: int = 20,
    e_ops: EOps = None,
    store_states: Optional[bool] = None,
    extra_observables: Tuple[Operator, ...] = tuple(),
    on_rebuild: Optional[Callable[[Dict[str, Any]], Any]] = None,
    checkpoint_file: Optional[Union[str, os.PathLike]] = None,
) -> Simulation:
    """Restricted Max-Ent evolution with adaptive bases and memory closure.

    Propagates ``K~(t) = sum_a phi_a(t) Q_a`` with ``phi' = A phi + m`` and
    ``m' = -G^+ E phi - m / tau`` (see the module documentation).

    Parameters
    ----------
    ham : Operator
        The Hamiltonian.
    k0 : Operator
        The initial generator, ``rho(0) ~ exp(-k0)``.
    t_span : array-like
        Output times; the first one is the initial time.
    order : int
        Depth of the hierarchical basis.
    n_body : int
        Many-body sector onto which the basis is projected.
    memory_time : float, "auto" or None
        Lifetime of the memory kernel. ``"auto"`` (default) estimates it at
        each rebuild (:func:`~qalma.evolution.estimate_memory_time`); ``None``
        disables the memory term (first-order evolution).
    default_memory_time : float
        Memory time used with ``"auto"`` until some leakage appears.
    adaptive : bool
        If False, a single basis built from ``k0`` is used for the whole
        evolution.
    rate_factor, rate_min : float
        The basis is rebuilt when the leakage rate exceeds
        ``max(rate_factor * floor, rate_min)``, ``floor`` being its value
        right after the last rebuild.
    drift_tol : float
        The basis and the reference are rebuilt when the local drift of the
        mean-field reference, ``max_i S(sigma'_i || sigma0_i)``, exceeds it.
    continuous_velocity : bool
        If True (and the memory is active), the previous velocity is added to
        the new basis at each rebuild and the memory is chosen so that
        ``phi'`` is continuous.
    n_quad : int
        Trapezoidal intervals used for the leakage integrals.
    e_ops : dict, list or callable, optional
        Observables evaluated in ``rho(t) ~ exp(-K~(t))``, or a callable
        ``e_ops(t, rho)``.
    store_states : bool, optional
        Store the generators ``K~(t)``. By default, only when ``e_ops`` is
        not given.
    extra_observables : tuple of Operator
        Operators always added to the basis.
    on_rebuild : callable, optional
        Called with a dictionary describing each new segment (keys
        ``"basis"``, ``"sigma_ref"``, ``"k_ref"``, ``"phi_0"``, ``"m_0"``,
        ``"gen"``, ``"e_mat"``, ``"tau"``, ``"floor"``, ``"t_ref"``).
    checkpoint_file : str or path-like, optional
        If given, the simulation is saved there after each output time.

    Returns
    -------
    Simulation
        ``states`` holds generators (``parameters["states"] ==
        "generator"``). Besides ``"method"`` and ``"wall time"``, ``stats``
        has, per output time, ``"errors"`` (accumulated Galerkin estimate),
        ``"irreducible entropy estimate"`` (``1/2 ||R||^2``, an estimate of
        ``S(rho || Pi_n rho)``), ``"leakage rate"`` and ``"local drift"``;
        and, per rebuild, ``"basis update times"``, ``"update reasons"``,
        ``"leakage floors"``, ``"memory times"`` (NaN without memory),
        ``"sector leakage fraction"``, ``"velocity jump"`` and
        ``"basis time costs"``.

    Notes
    -----
    A ``RuntimeWarning`` is issued if, at a rebuild, more than
    ``SECTOR_LEAKAGE_WARNING`` of ``[H, K~]`` lies outside the n-body
    sector: the basis cannot represent the dynamics, and the memory term
    then produces spurious entropy. Increase ``n_body`` in that case.
    """
    auto = memory_time == "auto"
    if not (auto or memory_time is None or float(memory_time) > 0):
        raise ValueError("memory_time must be positive, 'auto' or None.")
    use_memory = memory_time is not None
    wall_start = datetime.now()
    t_span = list(t_span)
    t_0 = t_span[0]
    states: List[Operator] = []
    expect_ops: Dict[Any, Any] = {}
    tlist: List[float] = []
    parameters: Dict[str, Any] = {
        "method": "adaptive maxent",
        "system": ham.system,
        "states": "generator",
        "order": order,
        "n_body": n_body,
        "memory_time": (
            memory_time if isinstance(memory_time, float) else str(memory_time)
        ),
        "default_memory_time": default_memory_time,
        "adaptive": adaptive,
        "rate_factor": rate_factor,
        "rate_min": rate_min,
        "drift_tol": drift_tol,
        "continuous_velocity": continuous_velocity,
        "n_quad": n_quad,
    }
    stats: Dict[str, Any] = {
        "method": "adaptive maxent",
        "wall time": 0.0,
        "errors": [],
        "irreducible entropy estimate": [],
        "leakage rate": [],
        "local drift": [],
        "basis update times": [],
        "update reasons": [],
        "leakage floors": [],
        "memory times": [],
        "sector leakage fraction": [],
        "velocity jump": [],
        "basis time costs": [],
    }
    simulation = Simulation(
        parameters=parameters,
        stats=stats,
        time_span=tlist,
        expect_ops=expect_ops,
        states=states,
    )
    record = GeneratorRecorder(e_ops, store_states, states, expect_ops)
    if not record.store_states:
        parameters["states"] = "none"
    current_tau = [default_memory_time]

    def make_segment(basis, sigma, phi_0, m_0, t, tau, extra) -> Dict[str, Any]:
        """Assemble the propagation data of a segment."""
        gen = np.real(np.asarray(basis.gen_matrix))
        e_mat = leakage_gram(basis)
        mem_mat = np.linalg.pinv(basis.gram, rcond=1e-10, hermitian=True) @ e_mat
        segment = {
            "basis": basis,
            "sigma_ref": sigma,
            "gen": gen,
            "e_mat": e_mat,
            "tau": tau,
            "block": _segment_block(gen, mem_mat, tau),
            "phi_0": phi_0,
            "m_0": m_0 if tau is not None else np.zeros_like(phi_0),
            "t_ref": t,
            "floor": leakage_rate(e_mat, phi_0),
        }
        segment.update(extra)
        return segment

    def rebuild(k, velocity, sigma_prev, t, reason) -> Dict[str, Any]:
        """Build basis, reference and memory data from ``k`` at time ``t``."""
        start = datetime.now()
        extra_ops = tuple(extra_observables)
        carry_velocity = continuous_velocity and use_memory and velocity is not None
        if carry_velocity:
            extra_ops = extra_ops + (velocity,)
        basis, sigma, k_ref = update_basis(
            k.simplify(), sigma_prev, ham, order, n_body, extra_ops
        )
        phi_0 = np.real(basis.coefficient_expansion(k))
        gen = np.real(np.asarray(basis.gen_matrix))
        if carry_velocity:
            m_0 = np.real(basis.coefficient_expansion(velocity)) - gen @ phi_0
        else:
            m_0 = np.zeros_like(phi_0)
        jump = 0.0
        if velocity is not None:
            new_velocity = basis.operator_from_coefficients(gen @ phi_0 + m_0)
            denom = _norm(basis, velocity)
            if denom > 0:
                jump = _norm(basis, (new_velocity - velocity).simplify()) / denom
        estimated = True
        tau: Optional[float]
        if auto:
            tau_est, leak_norm, _ = estimate_memory_time(
                basis, k, fallback=current_tau[0]
            )
            estimated = leak_norm > 1e-10
            current_tau[0] = tau_est
            tau = tau_est
        else:
            tau = None if memory_time is None else float(memory_time)
        fraction = sector_leakage_fraction(ham, k, sigma, n_body)
        if fraction > SECTOR_LEAKAGE_WARNING:
            warnings.warn(
                f"At t={t}, a fraction {fraction:.2f} of [H, K] lies outside the "
                f"{n_body}-body sector; the restricted dynamics cannot represent "
                "it. Consider a larger n_body.",
                RuntimeWarning,
                stacklevel=3,
            )
        segment = make_segment(
            basis,
            sigma,
            phi_0,
            m_0,
            t,
            tau,
            {"k_ref": k_ref, "tau_estimated": estimated, "rebuilt_at": t},
        )
        cost = datetime.now() - start
        stats["basis update times"].append(float(t))
        stats["update reasons"].append(reason)
        stats["leakage floors"].append(segment["floor"])
        stats["memory times"].append(np.nan if tau is None else tau)
        stats["sector leakage fraction"].append(fraction)
        stats["velocity jump"].append(jump)
        stats["basis time costs"].append(cost.total_seconds())
        logger.info(f"basis rebuilt at t={t} ({reason}); memory time {tau}")
        if on_rebuild is not None:
            on_rebuild(segment)
        return segment

    segment = rebuild(
        k0, None, GibbsProductDensityOperator({}, k0.system), t_0, "initial"
    )
    accumulated_eps = 0.0
    accumulated_leak = 0.0
    last: Dict[str, Any] = {"t": t_0, "k": k0, "eps": 0.0, "leak": 0.0, "v": None}

    tlist.append(t_0)
    stats["errors"].append(0.0)
    stats["irreducible entropy estimate"].append(0.0)
    stats["leakage rate"].append(segment["floor"])
    stats["local drift"].append(0.0)
    record(t_0, k0)

    for t in t_span[1:]:
        while True:
            phi, mem, rate, eps, psi = _propagate(
                segment, t - segment["t_ref"], segment["e_mat"], n_quad
            )
            k_t = segment["basis"].operator_from_coefficients(phi).simplify()
            drift, _ = local_mean_field_drift(k_t, segment["sigma_ref"])
            if not adaptive:
                break
            reason = None
            if rate > max(rate_factor * segment["floor"], rate_min):
                reason = "leakage"
            elif drift > drift_tol:
                reason = "drift"
            # Never rebuild twice at the same time (restarts do not count).
            fresh = abs(segment["rebuilt_at"] - last["t"]) < 1e-12
            if reason is None or fresh:
                break
            accumulated_eps += last["eps"]
            accumulated_leak += last["leak"]
            segment = rebuild(
                last["k"], last["v"], segment["sigma_ref"], last["t"], reason
            )

        velocity = (
            segment["basis"]
            .operator_from_coefficients(segment["gen"] @ phi + mem)
            .simplify()
        )
        leak = leakage_rate(segment["e_mat"], psi)
        last = {"t": t, "k": k_t, "eps": eps, "leak": leak, "v": velocity}
        tlist.append(float(t))
        stats["errors"].append(accumulated_eps + eps)
        stats["irreducible entropy estimate"].append(
            0.5 * (accumulated_leak + leak) ** 2
        )
        stats["leakage rate"].append(rate)
        stats["local drift"].append(drift)
        record(t, k_t)

        # With "auto", estimate the memory time as soon as some leakage
        # appears inside a segment that started without it. The segment is
        # restarted at t with the same basis, so phi and m stay continuous.
        if auto and not segment["tau_estimated"] and rate > 1e-8:
            tau, leak_norm, _ = estimate_memory_time(
                segment["basis"], k_t, fallback=current_tau[0]
            )
            if leak_norm > 1e-10:
                current_tau[0] = tau
                stats["memory times"][-1] = tau
                accumulated_eps += eps
                accumulated_leak += leak
                last.update(eps=0.0, leak=0.0)
                segment = make_segment(
                    segment["basis"],
                    segment["sigma_ref"],
                    np.real(phi),
                    np.real(mem),
                    t,
                    tau,
                    {
                        "k_ref": segment["k_ref"],
                        "tau_estimated": True,
                        "rebuilt_at": segment["rebuilt_at"],
                    },
                )
        stats["wall time"] = (datetime.now() - wall_start).total_seconds()
        if checkpoint_file is not None:
            simulation.save_hdf5(checkpoint_file, mode="w")
    stats["wall time"] = (datetime.now() - wall_start).total_seconds()
    if checkpoint_file is not None:
        simulation.save_hdf5(checkpoint_file, mode="w")
    return simulation
