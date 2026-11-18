r"""
Restricted Max-Ent evolution with a memory closure.

The first-order restricted evolution propagates the generator
:math:`\tilde K = \sum_a \phi_a Q_a` with :math:`\dot\phi = A\phi`, where
:math:`A = G^{-1}h` is the generator matrix of the basis
(:func:`leakage_controlled_evolution`). It discards the feedback of the part of
the dynamics that leaves the span of the basis,
:math:`\pi L Q L \tilde K` with :math:`L = -i[H, \cdot]` and :math:`Q = 1 - \pi`,
and is therefore almost isentropic, while the instantaneous Max-Ent projection
:math:`\Pi_n\rho(t)` that it approximates produces entropy:
:math:`S(\rho\,\|\,\Pi_n\rho) = S(\Pi_n\rho) - S(\rho)`.

Projecting :math:`\ddot K = L^2 K` onto the basis gives, keeping the
first-order part exact and with :math:`E` the leakage Gram matrix
(:func:`leakage_gram`),

.. math::

    \ddot\phi = (A^2 - G^{+} E)\,\phi ,

which reproduces the second-order Mori-Zwanzig memory term and makes
:math:`\tfrac12\|\tilde K\|^2` decrease as :math:`-\phi^T E\phi` at short times.
Its memory never decays. Here the memory kernel decays exponentially with a
time :math:`\tau`, which gives the local system

.. math::

    \dot\phi = A\phi + m, \qquad \dot m = -G^{+}E\,\phi - m/\tau,
    \qquad m(t_0) = 0 ,

continuous in time, which reduces to the first-order evolution for
:math:`\tau \to 0` and to the second-order equation for :math:`\tau\to\infty`.
:func:`estimate_memory_time` estimates :math:`\tau` from the short-time
(Gaussian) decay of the autocorrelation of the leaked part,
:math:`\tau = \sqrt{\pi/2}\,\|R\|/\|QLR\|` with :math:`R = QL\tilde K`.

When the basis is rebuilt, the previous velocity :math:`\dot{\tilde K}` is
added to the new basis and the memory is set so that :math:`\dot\phi` is
continuous across the rebuild.
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from qalma.operators import Operator
from qalma.operators.functions import commutator
from qalma.operators.states import GibbsDensityOperator, GibbsProductDensityOperator
from qalma.projections import n_body_projection
from qalma.scalarprod import OperatorBasis

from .leakage_control import leakage_gram, leakage_rate, local_mean_field_drift
from .maxent_evol import update_basis
from .simulation import Simulation

logger = logging.getLogger(__name__)

__all__ = [
    "estimate_memory_time",
    "memory_projected_evolution",
    "sector_leakage_fraction",
]

#: Above this sector-leakage fraction a warning recommends a larger ``n_body``.
SECTOR_LEAKAGE_WARNING = 0.9


def _norm(basis: OperatorBasis, operator: Operator) -> float:
    """Return the norm of ``operator`` in the scalar product of ``basis``."""
    return float(np.sqrt(max(np.real(basis.sp(operator, operator)), 0.0)))


def _residual(basis: OperatorBasis, operator: Operator) -> Operator:
    """Return the component of ``operator`` orthogonal to the span of ``basis``."""
    operator = operator.simplify()
    parallel = basis.operator_from_coefficients(basis.coefficient_expansion(operator))
    return (operator - parallel).simplify()


def estimate_memory_time(
    basis: OperatorBasis, k: Operator, fallback: float = 0.5, tiny: float = 1e-10
) -> Tuple[float, float, float]:
    r"""Estimate the lifetime of the memory kernel for the generator ``k``.

    With :math:`R = (1 - \pi) L k` the leakage of ``k`` out of the span of the
    basis, the autocorrelation of the leaked part decays at short times as
    :math:`\|R\|^2 (1 - s^2 \|QLR\|^2 / 2\|R\|^2)`, a Gaussian with time
    :math:`\tau_G = \|R\| / \|QLR\|`. The exponential kernel with the same
    integral has :math:`\tau = \sqrt{\pi/2}\,\tau_G`.

    Parameters
    ----------
    basis : OperatorBasis
        Basis with a generator (``-i H`` up to the convention of the basis).
    k : Operator
        The current generator.
    fallback : float
        Value returned when there is no leakage.
    tiny : float
        Threshold below which a norm is considered zero.

    Returns
    -------
    tau : float
        Estimated memory time (``fallback`` if there is no leakage).
    leak_norm : float
        :math:`\|R\|`.
    second_norm : float
        :math:`\|QLR\|`.
    """
    leak = _residual(basis, commutator(k.simplify(), basis.generator))
    leak_norm = _norm(basis, leak)
    if leak_norm < tiny:
        return fallback, leak_norm, 0.0
    second = _residual(basis, commutator(leak, basis.generator))
    second_norm = _norm(basis, second)
    if second_norm < tiny:
        return fallback, leak_norm, second_norm
    return float(np.sqrt(np.pi / 2) * leak_norm / second_norm), leak_norm, second_norm


def sector_leakage_fraction(
    ham: Operator, k: Operator, sigma, n_body: int, tiny: float = 1e-12
) -> float:
    r"""Fraction of :math:`[H, k]` outside the ``n_body`` sector.

    Returns :math:`\|(1 - \pi^{(n)})[H, k]\| / \|[H, k]\|` in the covariance
    geometry of the product state ``sigma``. A value close to one means that
    the dynamics leaves the n-body sector entirely, so that no n-body basis
    can represent it and ``n_body`` should be increased.
    """
    from qalma.scalarprod import fetch_covar_scalar_product

    if hasattr(sigma, "to_product_state"):
        sigma = sigma.to_product_state()
    sp = fetch_covar_scalar_product(sigma)
    comm = commutator(ham, k.simplify()).simplify()
    total = np.real(sp(comm, comm))
    if total < tiny:
        return 0.0
    projected = n_body_projection(comm, n_max=n_body, sigma=sigma).simplify()
    outside = (comm - projected).simplify()
    return float(np.sqrt(max(np.real(sp(outside, outside)), 0.0) / total))


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


def memory_projected_evolution(
    ham: Operator,
    k0: Operator,
    t_span,
    order: int,
    n_body: int = 2,
    *,
    memory_time: Union[float, str, None] = "auto",
    default_memory_time: float = 0.5,
    rate_factor: float = 2.0,
    rate_min: float = 0.05,
    drift_tol: float = 0.01,
    continuous_velocity: bool = True,
    n_quad: int = 20,
    e_ops: Optional[Dict | List | Callable] = None,
    extra_observables: Tuple[Operator, ...] = tuple(),
    on_update_basis_callback: Optional[Callable] = None,
    store_states: bool = False,
    checkpoint_file: Optional[Union[str, os.PathLike]] = None,
) -> Simulation:
    """Restricted Max-Ent evolution with a decaying memory closure.

    Propagates ``K~(t) = sum_a phi_a(t) Q_a`` in a hierarchical basis with
    ``phi' = A phi + m`` and ``m' = -G^+ E phi - m / tau`` (see the module
    documentation). The basis and the mean-field reference are rebuilt with
    the criteria of :func:`leakage_controlled_evolution`: when the leakage
    rate exceeds ``max(rate_factor * floor, rate_min)``, or when the local
    drift of the reference exceeds ``drift_tol``.

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
        each rebuild with :func:`estimate_memory_time`; ``None`` disables the
        memory, which gives the first-order evolution of
        :func:`leakage_controlled_evolution`.
    default_memory_time : float
        Memory time used with ``"auto"`` until some leakage appears.
    rate_factor, rate_min, drift_tol : float
        Rebuild criteria, as in :func:`leakage_controlled_evolution`.
    continuous_velocity : bool
        If True, the previous velocity is added to the new basis at each
        rebuild and the memory is chosen so that ``phi'`` is continuous.
    n_quad : int
        Trapezoidal intervals used for the leakage integrals.
    e_ops : dict, list or callable, optional
        Observables to evaluate at each output time, or a callable
        ``e_ops(t, k)``.
    extra_observables : tuple of Operator
        Operators always added to the basis.
    on_update_basis_callback : callable, optional
        Called with a dictionary describing each new segment.
    store_states : bool
        Store the generators ``K~(t)`` also when ``e_ops`` is given.
    checkpoint_file : str or path-like, optional
        If given, the simulation is saved there after each output time.

    Returns
    -------
    Simulation
        With ``stats`` keys:

        * ``"errors"``: accumulated Galerkin estimate
          ``int ||pi_perp(-i[H, K~])|| dt``;
        * ``"irreducible entropy estimate"``: ``1/2 (sum_segments ||R||)^2``,
          a second-order estimate of ``S(rho || Pi_n rho)``;
        * ``"leakage rate"``, ``"local drift"``: per output time;
        * ``"basis update times"``, ``"update reasons"``, ``"leakage floors"``,
          ``"memory times"``, ``"sector leakage fraction"``,
          ``"velocity jump"`` (relative jump of ``K~'`` at the rebuild; zero up
          to round-off with ``continuous_velocity``), ``"basis time costs"``:
          per rebuild.

    Notes
    -----
    A ``RuntimeWarning`` is issued if, at a rebuild, more than
    :data:`SECTOR_LEAKAGE_WARNING` of ``[H, K~]`` lies outside the n-body
    sector: the basis cannot represent the dynamics, and the memory term then
    produces spurious entropy. Increase ``n_body`` in that case.
    """
    auto = memory_time == "auto"
    if not (auto or memory_time is None or float(memory_time) > 0):
        raise ValueError("memory_time must be positive, 'auto' or None.")
    t_span = list(t_span)
    t_0 = t_span[0]
    states: List[Operator] = []
    expect_ops: Dict[Any, Any] = {}
    tlist: List[float] = []
    parameters: Dict[str, Any] = {
        "n_body": n_body,
        "order": order,
        "memory_time": memory_time,
        "default_memory_time": default_memory_time,
        "rate_factor": rate_factor,
        "rate_min": rate_min,
        "drift_tol": drift_tol,
        "continuous_velocity": continuous_velocity,
        "n_quad": n_quad,
        "system": ham.system,
        "method": "Memory-Closure Restricted Evolution",
    }
    stats: Dict[str, Any] = {
        "method": "Memory-Closure Restricted Evolution",
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

    if e_ops is None:

        def on_success(t, k):
            """Store the state ``k``."""
            states.append(k)

    elif hasattr(e_ops, "__call__"):
        on_success = cast(Callable, e_ops)
    else:
        if not isinstance(e_ops, dict):
            e_ops = {pos: e_op for pos, e_op in enumerate(cast(Iterable, e_ops))}

        def on_success(t, k):
            """Evaluate the observables and optionally store ``k``."""
            for key, val in GibbsDensityOperator(k).expect(e_ops).items():
                expect_ops.setdefault(key, []).append(val)
            if store_states:
                states.append(k)

    current_tau = [default_memory_time]

    def make_segment(basis, sigma, phi_0, m_0, t, tau, extra=None):
        """Assemble the propagation data of a segment."""
        gen = np.real(np.asarray(basis.gen_matrix))
        e_mat = leakage_gram(basis)
        g_pinv = np.linalg.pinv(basis.gram, rcond=1e-10, hermitian=True)
        mem_mat = g_pinv @ e_mat
        segment = {
            "basis": basis,
            "sigma_ref": sigma,
            "gen": gen,
            "e_mat": e_mat,
            "mem_mat": mem_mat,
            "tau": tau,
            "block": _segment_block(gen, mem_mat, tau),
            "phi_0": phi_0,
            "m_0": m_0 if tau is not None else np.zeros_like(phi_0),
            "t_ref": t,
            "floor": leakage_rate(e_mat, phi_0),
            "tau_estimated": not auto,
        }
        if extra:
            segment.update(extra)
        return segment

    def rebuild(k, velocity, sigma_prev, t, reason) -> Dict[str, Any]:
        """Build basis, reference and memory data from ``k`` at time ``t``."""
        start = datetime.now()
        extra_ops = tuple(extra_observables)
        if continuous_velocity and velocity is not None and memory_time is not None:
            extra_ops = extra_ops + (velocity,)
        basis, sigma, k_ref = update_basis(
            k.simplify(), sigma_prev, ham, order, n_body, extra_ops
        )
        phi_0 = np.real(basis.coefficient_expansion(k))
        gen = np.real(np.asarray(basis.gen_matrix))
        jump = 0.0
        if memory_time is None or velocity is None:
            m_0 = np.zeros_like(phi_0)
        elif continuous_velocity:
            m_0 = np.real(basis.coefficient_expansion(velocity)) - gen @ phi_0
        else:
            m_0 = np.zeros_like(phi_0)
        if velocity is not None:
            new_velocity = basis.operator_from_coefficients(gen @ phi_0 + m_0)
            diff = (new_velocity - velocity).simplify()
            denom = _norm(basis, velocity)
            jump = _norm(basis, diff) / denom if denom > 0 else 0.0
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
        stats["basis update times"].append(t)
        stats["update reasons"].append(reason)
        stats["leakage floors"].append(segment["floor"])
        stats["memory times"].append(tau)
        stats["sector leakage fraction"].append(fraction)
        stats["velocity jump"].append(jump)
        stats["basis time costs"].append(cost.seconds + 1e-6 * cost.microseconds)
        logger.info(f"basis rebuilt at t={t} ({reason}); memory time {tau}")
        if on_update_basis_callback is not None:
            on_update_basis_callback(segment)
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
    on_success(t_0, k0)

    for t in t_span[1:]:
        while True:
            phi, mem, rate, eps, psi = _propagate(
                segment, t - segment["t_ref"], segment["e_mat"], n_quad
            )
            k_t = segment["basis"].operator_from_coefficients(phi).simplify()
            drift, _ = local_mean_field_drift(k_t, segment["sigma_ref"])
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

        velocity_coeffs = segment["gen"] @ phi + mem
        velocity = (
            segment["basis"].operator_from_coefficients(velocity_coeffs).simplify()
        )
        leak = leakage_rate(segment["e_mat"], psi)
        last = {"t": t, "k": k_t, "eps": eps, "leak": leak, "v": velocity}
        tlist.append(t)
        stats["errors"].append(accumulated_eps + eps)
        stats["irreducible entropy estimate"].append(
            0.5 * (accumulated_leak + leak) ** 2
        )
        stats["leakage rate"].append(rate)
        stats["local drift"].append(drift)
        on_success(t, k_t)

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
        if checkpoint_file is not None:
            simulation.save_hdf5(checkpoint_file, mode="w")
    return simulation
