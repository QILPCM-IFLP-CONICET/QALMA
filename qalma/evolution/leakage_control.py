r"""
Leakage and memory diagnostics of the restricted Max-Ent evolution.

The restricted evolution of :func:`~qalma.evolution.adaptive_maxent_evolution`
propagates the generator :math:`\tilde K(t)=\sum_a\phi_a(t)Q_a` inside the span
of an operator basis :math:`B=\{Q_a\}`, using the covariance geometry of a
mean-field reference state :math:`\sigma_0`. This module provides the
quantities built from the *leakage* of the dynamics out of that span,

.. math::

    r_a = \pi_\perp(-i[H, Q_a]),\qquad
    E_{ab} = (r_a, r_b)^{\rm cov}_{\sigma_0},

which is computed once per basis (:func:`leakage_gram`). From :math:`E`:

* the leakage rate :math:`\|\pi_\perp(-i[H,\tilde K])\|=\sqrt{\phi^T E\,\phi}`
  (:func:`leakage_rate`);
* the Galerkin error estimate
  :math:`\varepsilon(\tau)=\int_0^\tau\sqrt{\phi^T E\,\phi}\,ds`, and the
  accumulated leakage :math:`R=\sum_a\psi_a r_a` with
  :math:`\psi=\int\phi\,ds` (:func:`leakage_integrals`); to second order,
  :math:`\tfrac12\|R\|^2` estimates the irreducible error
  :math:`S(\rho\|\Pi_n\rho)=S(\Pi_n\rho)-S(\rho)`.

It also provides the local drift of the mean-field reference
(:func:`local_mean_field_drift`), the lifetime of the memory kernel
(:func:`estimate_memory_time`), and the fraction of the dynamics that leaves
the n-body sector (:func:`sector_leakage_fraction`).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import qutip
from numpy.typing import NDArray
from scipy.linalg import expm

from qalma.operators import Operator
from qalma.operators.functions import commutator
from qalma.operators.states import GibbsProductDensityOperator
from qalma.projections import n_body_projection
from qalma.scalarprod import OperatorBasis, fetch_covar_scalar_product

logger = logging.getLogger(__name__)

__all__ = [
    "basis_commutators",
    "estimate_memory_time",
    "leaked_operator",
    "leakage_gram",
    "leakage_integrals",
    "leakage_rate",
    "local_mean_field_drift",
    "sector_leakage_fraction",
]


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
    ops = tuple(basis.operator_basis)
    comms = basis_commutators(basis)
    size = len(ops)
    if hasattr(sp, "compute_gram_matrix"):
        # Upper triangle only; each operator is flattened once.
        comm_gram = np.real(sp.compute_gram_matrix(comms))
    else:
        comm_gram = np.array(
            [
                [np.real(sp(comms[a], comms[b])) for b in range(size)]
                for a in range(size)
            ]
        )
    h_mat = getattr(basis, "_hij", None)
    if h_mat is None or np.shape(h_mat) != (size, size):
        h_mat = np.array(
            [[np.real(sp(ops[a], comms[b])) for b in range(size)] for a in range(size)]
        )
    h_mat = np.real(np.asarray(h_mat))
    gram_pinv = np.linalg.pinv(basis.gram, rcond=1e-10, hermitian=True)
    e_mat = comm_gram - h_mat.T @ gram_pinv @ h_mat
    e_mat = 0.5 * (e_mat + e_mat.T)
    # Remove the round-off negative part.
    evals, evecs = np.linalg.eigh(e_mat)
    e_mat = (evecs * np.clip(evals, 0.0, None)) @ evecs.T
    basis._leakage_gram = e_mat  # type: ignore[attr-defined]
    basis._leakage_comm_gram = comm_gram  # type: ignore[attr-defined]
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


# ─────────────────────────── memory and sector diagnostics ──────────────────


def basis_commutators(basis: OperatorBasis) -> Tuple[Operator, ...]:
    """Return the commutators ``[Q_a, G]`` of the basis elements with its generator.

    They are computed once and cached on the basis.
    """
    cached = getattr(basis, "_leakage_commutators", None)
    if cached is not None:
        return cached
    if basis.generator is None:
        raise ValueError("The basis has no generator.")
    comms = tuple(
        commutator(op, basis.generator).simplify() for op in basis.operator_basis
    )
    basis._leakage_commutators = comms  # type: ignore[attr-defined]
    return comms


def leaked_operator(basis: OperatorBasis, phi: NDArray) -> Operator:
    """Return ``(1 - pi) L K`` for ``K = sum_a phi_a Q_a``.

    ``L K = sum_a phi_a [Q_a, G]`` is a combination of the cached commutators,
    and its projection onto the basis is ``sum_a (A phi)_a Q_a``, with ``A``
    the generator matrix: no new commutator or scalar product is needed.
    """
    phi = np.real(np.asarray(phi))
    comms = basis_commutators(basis)
    gen = np.real(np.asarray(basis.gen_matrix))
    lk = sum((c * comm for c, comm in zip(phi, comms) if c != 0), start=0 * comms[0])
    parallel = basis.operator_from_coefficients(gen @ phi)
    return (lk - parallel).simplify()


def _norm(basis: OperatorBasis, operator: Operator) -> float:
    """Return the norm of ``operator`` in the scalar product of ``basis``."""
    return float(np.sqrt(max(np.real(basis.sp(operator, operator)), 0.0)))


def _residual(basis: OperatorBasis, operator: Operator) -> Operator:
    """Return the component of ``operator`` orthogonal to the span of ``basis``."""
    operator = operator.simplify()
    parallel = basis.operator_from_coefficients(basis.coefficient_expansion(operator))
    return (operator - parallel).simplify()


def estimate_memory_time(
    basis: OperatorBasis,
    k: Operator,
    fallback: float = 0.5,
    tiny: float = 1e-10,
    phi: Optional[NDArray] = None,
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
    phi : NDArray, optional
        Coefficients of ``k`` in the basis, if ``k`` lies in its span. Then the
        leaked part and its norm are obtained from the cached commutators and
        the leakage Gram matrix, without a new commutator.

    Returns
    -------
    tau : float
        Estimated memory time (``fallback`` if there is no leakage).
    leak_norm : float
        :math:`\|R\|`.
    second_norm : float
        :math:`\|QLR\|`.
    """
    if phi is not None:
        phi = np.real(np.asarray(phi))
        leak_sq = float(phi @ leakage_gram(basis) @ phi)
        # E = C - h^T G^+ h suffers cancellations of relative size ~1e-12 of
        # ||L k||^2 = phi^T C phi: below that, there is no leakage.
        comm_gram = getattr(basis, "_leakage_comm_gram")
        total_sq = float(phi @ comm_gram @ phi)
        if leak_sq <= max(tiny**2, 1e-10 * total_sq):
            return fallback, 0.0, 0.0
        leak_norm = float(np.sqrt(leak_sq))
        leak = leaked_operator(basis, phi)
    else:
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
    ham: Operator,
    k: Operator,
    sigma,
    n_body: int,
    tiny: float = 1e-12,
    *,
    comm: Optional[Operator] = None,
    sp=None,
) -> float:
    r"""Fraction of :math:`[H, k]` outside the ``n_body`` sector.

    Returns :math:`\|(1 - \pi^{(n)})[H, k]\| / \|[H, k]\|` in the covariance
    geometry of the product state ``sigma``. A value close to one means that
    the dynamics leaves the n-body sector entirely, so that no n-body basis
    can represent it and ``n_body`` should be increased.

    ``comm`` (any multiple of ``[H, k]``, the ratio does not depend on it) and
    ``sp`` (the covariance product of ``sigma``) can be passed to reuse them.
    """
    if hasattr(sigma, "to_product_state"):
        sigma = sigma.to_product_state()
    if sp is None:
        sp = fetch_covar_scalar_product(sigma)
    if comm is None:
        comm = commutator(ham, k.simplify())
    comm = comm.simplify()
    total = np.real(sp(comm, comm))
    if total < tiny:
        return 0.0
    projected = n_body_projection(comm, n_max=n_body, sigma=sigma).simplify()
    outside = (comm - projected).simplify()
    return float(np.sqrt(max(np.real(sp(outside, outside)), 0.0) / total))
