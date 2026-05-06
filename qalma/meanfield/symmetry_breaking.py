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

An optional ``sigma_ref`` warm start can be supplied (e.g. the converged
state from a neighbouring point in a parameter sweep).  When provided it
is refined on the unperturbed ``k`` and its free energy is compared against
all random-perturbation attempts; the overall best is returned.  This makes
the function suitable as a drop-in replacement for
:func:`variational_quadratic_mfa` in sweeps, combining adiabatic continuity
(warm start) with escape from local minima (random restarts) in a
model-agnostic way.
"""

from typing import Optional

import numpy as np

from qalma.model import SystemDescriptor
from qalma.operators import LocalOperator, Operator
from qalma.operators.states import ProductDensityOperator

from .variational import compute_free_energy, variational_quadratic_mfa


def _random_hermitian(d: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random Hermitian matrix of size d with unit spectral norm."""
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = (A + A.conj().T) / 2
    norm = np.linalg.norm(H, ord=2)  # spectral norm = largest singular value
    return H / norm if norm > 1e-15 else H


def symmetry_breaking_mfa(
    k: Operator,
    system: SystemDescriptor,
    numfields: int = 6,
    epsilon: float = 1e-3,
    n_attempts: int = 1,
    sigma_ref: Optional[ProductDensityOperator] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> ProductDensityOperator:
    r"""Variational mean-field with a symmetry-breaking perturbation.

    Applies a small random Hermitian perturbation on a single randomly
    chosen site before the first optimisation round, then refines the
    result without the perturbation.  This allows the optimiser to escape
    symmetric fixed points and find symmetry-broken phases.

    The perturbation is model-agnostic: it is constructed from the local
    Hilbert-space dimension only, so it works for any on-site basis (spin,
    boson, fermion, etc.).

    **Strategy**

    *Warm-start candidate* (only when ``sigma_ref`` is provided):
      Refine ``sigma_ref`` on the unperturbed ``k``.  This preserves
      adiabatic continuity across a parameter sweep.

    *Random-restart candidates* (``n_attempts`` times):

    1. Build :math:`k_\epsilon = k + \epsilon\,\delta h_i`, where
       :math:`\delta h_i` is a random Hermitian operator of unit spectral
       norm acting on a single site :math:`i` chosen uniformly at random.
    2. Run :func:`variational_quadratic_mfa` on :math:`k_\epsilon` to
       obtain a symmetry-broken warm start :math:`\sigma_\epsilon`.
    3. Run :func:`variational_quadratic_mfa` on the original :math:`k`
       starting from :math:`\sigma_\epsilon`.

    The candidate with the lowest variational free energy across the warm
    start (if any) and all random-restart attempts is returned.  Combining
    both mechanisms in a single call makes this function suitable as a
    drop-in replacement for :func:`variational_quadratic_mfa` in parameter
    sweeps, without any model-specific logic.

    Parameters
    ----------
    k : Operator
        The generator of the target Gibbs state, i.e. ``beta * H``.
    system : SystemDescriptor
        The lattice system.  Used to read site dimensions and to
        construct the local perturbation operator.
    numfields : int, optional
        Number of auxiliary variational fields.  Forwarded to
        :func:`variational_quadratic_mfa` in all rounds.
        Default is 6.
    epsilon : float, optional
        Spectral norm of the symmetry-breaking field.  Should be small
        enough not to bias the final result (default ``1e-3``).
    n_attempts : int, optional
        Number of independent random-perturbation attempts.  The candidate
        with the lowest variational free energy across all attempts (and the
        warm-start candidate, if ``sigma_ref`` is given) is returned.
        Default is 1.
    sigma_ref : ProductDensityOperator or None, optional
        External warm-start state, e.g. the converged solution from a
        neighbouring point in a parameter sweep.  When provided it is
        refined on the unperturbed ``k`` and competes against the
        random-restart candidates.  Default is ``None``.
    seed : int or None, optional
        Seed for the random number generator.  Passing an integer gives
        reproducible results across the random-restart attempts.
        Default is ``None`` (non-reproducible).
    **kwargs
        Additional keyword arguments forwarded to
        :func:`variational_quadratic_mfa` in all rounds (e.g.
        ``max_self_consistent_steps``, ``callback``).

    Returns
    -------
    ProductDensityOperator
        The best variational product state found, i.e. the one with the
        lowest variational free energy
        :math:`F[\sigma] = \mathrm{Tr}[\sigma(k + \log\sigma)]`.

    See Also
    --------
    variational_quadratic_mfa : The underlying optimiser.
    compute_free_energy : Variational free energy.

    Examples
    --------
    Basic use (single random restart, no warm start):

    >>> from qalma.meanfield import symmetry_breaking_mfa
    >>> sigma = symmetry_breaking_mfa(
    ...     beta * ham, system,
    ...     numfields=6, epsilon=1e-3, n_attempts=3, seed=42,
    ... )
    >>> print(f"F = {compute_free_energy(sigma, beta * ham):.6f}")

    Use in a parameter sweep (warm start + random restarts):

    >>> sigma_prev = None
    >>> for B in field_values:
    ...     ham_B = ham_0 + zeeman * B
    ...     sigma = symmetry_breaking_mfa(
    ...         beta * ham_B, system,
    ...         numfields=6, n_attempts=3, sigma_ref=sigma_prev,
    ...     )
    ...     sigma_prev = sigma
    """
    import logging

    rng = np.random.default_rng(seed)
    sites = list(system.sites.keys())

    best_sigma: Optional[ProductDensityOperator] = None
    best_f: float = float("inf")

    # --- Candidate 0: refine the external warm start (if provided) --------
    if sigma_ref is not None:
        sigma_ws = variational_quadratic_mfa(
            k, numfields=numfields, sigma_ref=sigma_ref, **kwargs
        )
        best_f = compute_free_energy(sigma_ws, k)
        best_sigma = sigma_ws
        logging.debug(f"symmetry_breaking_mfa warm-start candidate: F={best_f:.6f}")

    # --- Candidates 1..n_attempts: random perturbations -------------------
    for attempt in range(n_attempts):
        # Step 1: build perturbation on a random site
        site = sites[rng.integers(len(sites))]
        d = system.dimensions[site]
        delta_h = _random_hermitian(d, rng)
        perturbation = LocalOperator(site, epsilon * delta_h, system)
        k_eps = k + perturbation

        # Seed numpy's global random state so that variational_quadratic_mfa
        # (which uses numpy.random.random_sample internally) is also
        # reproducible across calls with the same seed.
        np.random.seed(int(rng.integers(2**31)))

        # Step 2: first round on perturbed generator
        sigma_eps = variational_quadratic_mfa(k_eps, numfields=numfields, **kwargs)

        np.random.seed(int(rng.integers(2**31)))

        # Step 3: second round on original generator
        sigma = variational_quadratic_mfa(
            k, numfields=numfields, sigma_ref=sigma_eps, **kwargs
        )

        f = compute_free_energy(sigma, k)

        if f < best_f:
            best_f = f
            best_sigma = sigma

        logging.debug(
            f"symmetry_breaking_mfa attempt {attempt + 1}/{n_attempts}: "
            f"site={site}  F={f:.6f}  best={best_f:.6f}"
        )

    assert best_sigma is not None
    return best_sigma
