r"""
Leakage-controlled Max-Ent projected evolution (experimental).

The restricted Max-Ent evolution of :func:`adaptive_projected_evolution`
propagates the generator ``K~(t) = sum_a phi_a(t) Q_a`` inside the span of an
operator basis ``B = {Q_a}``, using the covariance geometry of a mean-field
reference state ``sigma0``. This module provides error indicators based on
the *leakage* of the dynamics out of that span,

.. math::

    r_a = \pi_\perp(-i[H, Q_a]),\qquad
    E_{ab} = (r_a, r_b)^{\rm cov}_{\sigma_0},

which is computed once per basis. From ``E``:

* the instantaneous leakage rate ``||pi_perp(-i[H, K~])|| = sqrt(phi^T E phi)``;
* the Galerkin error estimate
  ``eps(tau) = int_0^tau sqrt(phi(s)^T E phi(s)) ds``, which, unlike the
  estimate of :meth:`OperatorBasis.evolve`, does not use the triangle
  inequality over the basis elements;
* the accumulated leakage ``R(tau) = sum_a psi_a(tau) r_a`` with
  ``psi(tau) = int_0^tau phi(s) ds``. To second order, ``R`` approximates the
  component of the exact generator outside the n-body sector, and
  ``1/2 ||R||^2`` estimates the entropy production of the instantaneous
  Max-Ent projection, ``S(Pi_n rho(t)) - S(rho(0)) = S(rho(t) || Pi_n rho(t))``
  (the error that no n-body generator can remove).

Right after a basis is built from ``K~(t)``, the leakage rate equals
``||(1 - pi^(n)) [H, K~]||``, the part of the commutator outside the n-body
sector, which no rebuild of the basis can remove. It grows afterwards because
of the finite depth of the basis. :func:`leakage_controlled_evolution`
therefore rebuilds the basis only when the rate exceeds a multiple of this
floor, and rebuilds the reference state when its local (per site) drift
exceeds a tolerance. Both criteria are intensive: they do not grow with the
system size for local perturbations.

"""

import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import qutip
from numpy.typing import NDArray
from scipy.linalg import expm

from qalma.operators import Operator
from qalma.operators.functions import commutator
from qalma.operators.states import GibbsDensityOperator, GibbsProductDensityOperator
from qalma.projections import n_body_projection
from qalma.scalarprod import OperatorBasis

from .maxent_evol import update_basis
from .simulation import Simulation

logger = logging.getLogger(__name__)


# ─────────────────────────── leakage tensors ────────────────────────────────


def leakage_gram(basis: OperatorBasis) -> NDArray:
    """Return the Gram matrix of the leakage of the basis elements.

    ``E[a, b] = (r_a, r_b)``, with ``r_a`` the component of
    ``-i [H, Q_a]`` orthogonal to the span of the basis, in the scalar product
    of the basis. The result is cached on the basis object.

    Linearly dependent elements of the basis are handled through a
    pseudo-inverse of the Gram matrix.
    """
    cached = getattr(basis, "_leakage_gram", None)
    if cached is not None:
        return cached
    if basis.generator is None:
        raise ValueError("The basis has no generator.")
    sp = basis.sp
    ops = list(basis.operator_basis)
    comms = [commutator(op, basis.generator).simplify() for op in ops]
    size = len(ops)
    comm_gram = np.array(
        [[np.real(sp(comms[a], comms[b])) for b in range(size)] for a in range(size)]
    )
    h_mat = np.array(
        [[np.real(sp(ops[a], comms[b])) for b in range(size)] for a in range(size)]
    )
    gram_pinv = np.linalg.pinv(basis.gram, rcond=1e-10, hermitian=True)
    e_mat = comm_gram - h_mat.T @ gram_pinv @ h_mat
    e_mat = 0.5 * (e_mat + e_mat.T)
    # Remove the round-off negative part.
    evals, evecs = np.linalg.eigh(e_mat)
    e_mat = (evecs * np.clip(evals, 0.0, None)) @ evecs.T
    basis._leakage_gram = e_mat  # type: ignore[attr-defined]
    return e_mat


def leakage_rate(e_mat: NDArray, phi: NDArray) -> float:
    """Return ``sqrt(phi^T E phi)``, the norm of the leakage of ``sum phi_a Q_a``."""
    phi = np.real(np.asarray(phi))
    return float(np.sqrt(max(phi @ e_mat @ phi, 0.0)))


def leakage_integrals(
    gen_matrix: NDArray,
    phi_0: NDArray,
    e_mat: NDArray,
    tau: float,
    n_quad: int = 20,
) -> Tuple[NDArray, float, NDArray, float]:
    """Propagate the coefficients and integrate the leakage over ``[0, tau]``.

    Parameters
    ----------
    gen_matrix : NDArray
        The generator matrix of the basis (``basis.gen_matrix``).
    phi_0 : NDArray
        Coefficients at ``tau = 0``.
    e_mat : NDArray
        Leakage Gram matrix (:func:`leakage_gram`).
    tau : float
        Elapsed time since ``phi_0``.
    n_quad : int
        Number of trapezoidal intervals.

    Returns
    -------
    phi : NDArray
        Coefficients at ``tau``.
    eps : float
        ``int_0^tau ||pi_perp(-i[H, K~(s)])|| ds``, the Galerkin error estimate.
    psi : NDArray
        ``int_0^tau phi(s) ds``, coefficients of the accumulated leakage.
    rate : float
        Leakage rate at ``tau``.
    """
    gen_matrix = np.real(np.asarray(gen_matrix))
    phi_0 = np.real(np.asarray(phi_0))
    if tau <= 0:
        return phi_0, 0.0, np.zeros_like(phi_0), leakage_rate(e_mat, phi_0)
    n_quad = max(int(n_quad), 1)
    step = expm((tau / n_quad) * gen_matrix)
    phis = [phi_0]
    for _ in range(n_quad):
        phis.append(step @ phis[-1])
    phis_arr = np.array(phis)
    weights = np.full(n_quad + 1, tau / n_quad)
    weights[0] = weights[-1] = 0.5 * tau / n_quad
    rates = np.array([leakage_rate(e_mat, phi) for phi in phis_arr])
    eps = float(weights @ rates)
    psi = weights @ phis_arr
    return phis_arr[-1], eps, psi, float(rates[-1])


# ─────────────────────────── reference drift ────────────────────────────────


def _local_factor(state, site, dim):
    """Return the factor of ``state`` on ``site``, or the maximally mixed one."""
    factors = state.site_factors_qutip
    if site in factors:
        return factors[site]
    return qutip.qeye(dim) / dim


def local_mean_field_drift(k: Operator, sigma) -> Tuple[float, float]:
    """Estimate how far the mean-field state of ``exp(-k)`` is from ``sigma``.

    Performs one self-consistent step, ``kappa' = pi^(1)_sigma k``, and
    compares the resulting product state ``sigma'`` with ``sigma`` site by
    site.

    Returns
    -------
    max_drift : float
        ``max_i S(sigma'_i || sigma_i)``, an intensive measure.
    total_drift : float
        ``sum_i S(sigma'_i || sigma_i) = S(sigma' || sigma)``.
    """
    if hasattr(sigma, "to_product_state"):
        sigma = sigma.to_product_state()
    kappa = n_body_projection(k, n_max=1, sigma=sigma).hermitian_part()
    sigma_new = GibbsProductDensityOperator(kappa).to_product_state()
    dims = k.system.dimensions
    drifts = [
        qutip.entropy_relative(
            _local_factor(sigma_new, site, dims[site]),
            _local_factor(sigma, site, dims[site]),
        )
        for site in dims
    ]
    drifts_arr = np.clip(np.real(np.array(drifts, dtype=float)), 0.0, None)
    return float(drifts_arr.max(initial=0.0)), float(drifts_arr.sum())


# ─────────────────────────── evolution ──────────────────────────────────────


def leakage_controlled_evolution(
    ham: Operator,
    k0: Operator,
    t_span,
    order: int,
    n_body: int = 2,
    *,
    rate_factor: float = 2.0,
    rate_min: float = 0.05,
    drift_tol: float = 0.01,
    n_quad: int = 20,
    e_ops: Optional[Dict | List | Callable] = None,
    extra_observables: Tuple[Operator, ...] = tuple(),
    basis_update_callback: Callable[
        ..., Tuple[OperatorBasis, Operator, Operator]
    ] = update_basis,
    on_update_basis_callback: Optional[Callable] = None,
    store_states: bool = False,
    checkpoint_file: Optional[Union[str, os.PathLike]] = None,
) -> Simulation:
    """Max-Ent projected evolution with leakage-based error control (experimental).

    Propagates ``K~(t)`` as in :func:`adaptive_projected_evolution`, with a
    different control of the basis and of the reference state:

    * the basis is rebuilt from ``K~`` at the last accepted time when the
      leakage rate ``||pi_perp(-i[H, K~])||`` exceeds
      ``max(rate_factor * floor, rate_min)``, where ``floor`` is the rate
      right after the last rebuild (the leakage out of the n-body sector,
      which a rebuild cannot remove);
    * the basis and the reference are rebuilt when the local drift of the
      mean-field reference, ``max_i S(sigma'_i || sigma0_i)``, exceeds
      ``drift_tol`` (:func:`local_mean_field_drift`);
    * the basis is never rebuilt twice at the same time; if the criteria
      still fail right after a rebuild, the step is accepted.

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
    rate_factor : float
        Allowed growth of the leakage rate over its value after a rebuild.
    rate_min : float
        Leakage rate below which the basis is never rebuilt.
    drift_tol : float
        Tolerance for the local drift of the mean-field reference.
    n_quad : int
        Trapezoidal intervals used for the leakage integrals.
    e_ops, extra_observables, basis_update_callback, on_update_basis_callback,
    store_states, checkpoint_file :
        As in :func:`adaptive_projected_evolution`.

    Returns
    -------
    Simulation
        With ``stats`` keys:

        * ``"errors"``: accumulated Galerkin estimate
          ``sum_segments int ||pi_perp(-i[H, K~])|| dt``;
        * ``"irreducible entropy estimate"``: ``1/2 (sum_segments ||R||)^2``,
          a second-order estimate of ``S(rho(t) || Pi_n rho(t))``;
        * ``"leakage rate"``, ``"local drift"``: per output time;
        * ``"basis update times"``, ``"update reasons"``,
          ``"leakage floors"``, ``"basis time costs"``: per rebuild.

    Notes
    -----
    This method is experimental; its interface may change.
    """
    t_span = list(t_span)
    t_0 = t_span[0]
    states: List[Operator] = []
    expect_ops: Dict[Any, Any] = {}
    tlist: List[float] = []
    parameters: Dict[str, Any] = {
        "n_body": n_body,
        "order": order,
        "rate_factor": rate_factor,
        "rate_min": rate_min,
        "drift_tol": drift_tol,
        "n_quad": n_quad,
        "basis_update_callback": basis_update_callback.__name__,
        "system": ham.system,
        "method": "Leakage-Controlled Restricted Evolution",
    }
    stats: Dict[str, Any] = {
        "method": "Leakage-Controlled Restricted Evolution",
        "errors": [],
        "irreducible entropy estimate": [],
        "leakage rate": [],
        "local drift": [],
        "basis update times": [],
        "update reasons": [],
        "leakage floors": [],
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

    def rebuild(k, sigma_prev, t, reason) -> Dict[str, Any]:
        """Build basis, reference and leakage tensors from ``k`` at time ``t``."""
        start = datetime.now()
        basis, sigma, k_ref = basis_update_callback(
            k.simplify(), sigma_prev, ham, order, n_body, extra_observables
        )
        phi_0 = np.real(basis.coefficient_expansion(k))
        e_mat = leakage_gram(basis)
        floor = leakage_rate(e_mat, phi_0)
        cost = datetime.now() - start
        stats["basis update times"].append(t)
        stats["update reasons"].append(reason)
        stats["leakage floors"].append(floor)
        stats["basis time costs"].append(cost.seconds + 1e-6 * cost.microseconds)
        segment = {
            "basis": basis,
            "sigma_ref": sigma,
            "k_ref": k_ref,
            "phi_0": phi_0,
            "gen": np.real(np.asarray(basis.gen_matrix)),
            "e_mat": e_mat,
            "floor": floor,
            "t_ref": t,
        }
        logger.info(f"basis rebuilt at t={t} ({reason}), leakage floor {floor:.3g}")
        if on_update_basis_callback is not None:
            on_update_basis_callback(segment)
        return segment

    segment = rebuild(k0, GibbsProductDensityOperator({}, k0.system), t_0, "initial")
    accumulated_eps = 0.0  # error of the closed segments
    accumulated_leak = 0.0  # sum of ||R|| over the closed segments
    last = {"t": t_0, "k": k0, "eps": 0.0, "leak": 0.0}

    tlist.append(t_0)
    stats["errors"].append(0.0)
    stats["irreducible entropy estimate"].append(0.0)
    stats["leakage rate"].append(segment["floor"])
    stats["local drift"].append(0.0)
    on_success(t_0, k0)

    for t in t_span[1:]:
        while True:
            tau = t - segment["t_ref"]
            phi, eps, psi, rate = leakage_integrals(
                segment["gen"], segment["phi_0"], segment["e_mat"], tau, n_quad
            )
            k_t = segment["basis"].operator_from_coefficients(phi).simplify()
            drift, _ = local_mean_field_drift(k_t, segment["sigma_ref"])
            reason = None
            if rate > max(rate_factor * segment["floor"], rate_min):
                reason = "leakage"
            elif drift > drift_tol:
                reason = "drift"
            fresh = abs(segment["t_ref"] - last["t"]) < 1e-12
            if reason is None or fresh:
                break
            accumulated_eps += last["eps"]
            accumulated_leak += last["leak"]
            segment = rebuild(last["k"], segment["sigma_ref"], last["t"], reason)

        leak = leakage_rate(segment["e_mat"], psi)
        last = {"t": t, "k": k_t, "eps": eps, "leak": leak}
        tlist.append(t)
        stats["errors"].append(accumulated_eps + eps)
        stats["irreducible entropy estimate"].append(
            0.5 * (accumulated_leak + leak) ** 2
        )
        stats["leakage rate"].append(rate)
        stats["local drift"].append(drift)
        on_success(t, k_t)
        if checkpoint_file is not None:
            simulation.save_hdf5(checkpoint_file, mode="w")
    return simulation
