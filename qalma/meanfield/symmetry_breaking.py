"""Symmetry-breaking wrapper for the variational mean-field approximation.

When the exact Gibbs state breaks a symmetry of the Hamiltonian (e.g.
magnetic order, chiral order), the self-consistent mean-field equations
may have the symmetric state as a stable fixed point.  In that case the
plain :func:`variational_quadratic_mfa` gets stuck regardless of the
number of auxiliary fields.

The strategy implemented here is to perturb the generator ``k = beta * H``
with a small random Hermitian operator acting on a single site before the
first optimisation round.  This breaks all symmetries simultaneously at
that site, giving the linearised many-body terms a non-trivial mean-field
contribution.  A second round without the perturbation then refines the
solution using the symmetry-broken state as a warm start.
"""

from typing import Optional, Tuple

import numpy as np

from qalma.model import SystemDescriptor
from qalma.operators import LocalOperator, Operator
from qalma.operators.states import ProductDensityOperator

from .variational import compute_free_energy, variational_quadratic_mfa


def _random_hermitian(d: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random Hermitian matrix of size d with unit spectral norm."""
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = (A + A.conj().T) / 2
    norm = np.linalg.norm(H, ord=2)   # spectral norm = largest singular value
    return H / norm if norm > 1e-15 else H


def symmetry_breaking_mfa(
    k: Operator,
    system: SystemDescriptor,
    numfields: int = 6,
    epsilon: float = 1e-3,
    n_attempts: int = 1,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[ProductDensityOperator, float]:
    r"""Variational mean-field with a symmetry-breaking perturbation.

    Applies a small random Hermitian perturbation on a single randomly
    chosen site before the first optimisation round, then refines the
    result without the perturbation.  This allows the optimiser to escape
    symmetric fixed points and find symmetry-broken phases.

    The perturbation is model-agnostic: it is constructed from the local
    Hilbert-space dimension only, so it works for any on-site basis (spin,
    boson, fermion, etc.).

    **Two-round strategy**

    1. Build :math:`k_\epsilon = k + \epsilon\,\delta h_i`, where
       :math:`\delta h_i` is a random Hermitian operator of unit spectral
       norm acting on a single site :math:`i` chosen uniformly at random.
    2. Run :func:`variational_quadratic_mfa` on :math:`k_\epsilon` to
       obtain a symmetry-broken warm start :math:`\sigma_\epsilon`.
    3. Run :func:`variational_quadratic_mfa` on the original :math:`k`
       starting from :math:`\sigma_\epsilon`.

    If ``n_attempts > 1``, steps 1–3 are repeated with different random
    perturbations and the result with the lowest variational free energy
    is returned.

    Parameters
    ----------
    k : Operator
        The generator of the target Gibbs state, i.e. ``beta * H``.
    system : SystemDescriptor
        The lattice system.  Used to read site dimensions and to
        construct the local perturbation operator.
    numfields : int, optional
        Number of auxiliary variational fields.  Forwarded to
        :func:`variational_quadratic_mfa` in both rounds.
        Default is 6.
    epsilon : float, optional
        Spectral norm of the symmetry-breaking field.  Should be small
        enough not to bias the final result (default ``1e-3``).
    n_attempts : int, optional
        Number of independent random perturbations to try.  The attempt
        with the lowest variational free energy is returned.
        Default is 1.
    seed : int or None, optional
        Seed for the random number generator.  Passing an integer gives
        reproducible results.  Default is ``None`` (non-reproducible).
    **kwargs
        Additional keyword arguments forwarded to
        :func:`variational_quadratic_mfa` in both rounds (e.g.
        ``max_self_consistent_steps``, ``callback``).

    Returns
    -------
    sigma : ProductDensityOperator
        The best variational product state found across all attempts.
    f : float
        Variational free energy :math:`F[\sigma] = \mathrm{Tr}[\sigma(k
        + \log\sigma)]` of the returned state.

    See Also
    --------
    variational_quadratic_mfa : The underlying optimiser.
    compute_free_energy : Variational free energy.

    Examples
    --------
    >>> from qalma.meanfield import symmetry_breaking_mfa
    >>> sigma, f = symmetry_breaking_mfa(
    ...     beta * ham, system,
    ...     numfields=6, epsilon=1e-3, n_attempts=3, seed=42,
    ... )
    >>> print(f"F = {f:.6f}")
    """
    rng = np.random.default_rng(seed)
    sites = list(system.sites.keys())

    best_sigma: Optional[ProductDensityOperator] = None
    best_f: float = float("inf")

    for attempt in range(n_attempts):
        # --- Step 1: build perturbation on a random site ------------------
        site = sites[rng.integers(len(sites))]
        d = system.dimensions[site]
        delta_h = _random_hermitian(d, rng)
        perturbation = LocalOperator(site, epsilon * delta_h, system)
        k_eps = k + perturbation

        # --- Step 2: first round on perturbed generator -------------------
        sigma_eps, _ = variational_quadratic_mfa(
            k_eps, numfields=numfields, **kwargs
        )

        # --- Step 3: second round on original generator -------------------
        sigma, _ = variational_quadratic_mfa(
            k, numfields=numfields, sigma_ref=sigma_eps, **kwargs
        )

        f = compute_free_energy(sigma, k)

        if f < best_f:
            best_f = f
            best_sigma = sigma

        if n_attempts > 1:
            import logging
            logging.info(
                f"symmetry_breaking_mfa attempt {attempt + 1}/{n_attempts}: "
                f"site={site}  F={f:.6f}  best={best_f:.6f}"
            )

    return best_sigma, best_f
