"""Dynamical RPA: imaginary-time kernel and generator correction."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from qutip import Qobj

from qalma.operators import Operator, QutipOperator
from qalma.operators.basic import LocalOperator

from .environment import environment_modes
from .rpa_structure import RPAStructure, _split_sites
from .utils import _DEGENERACY_TOL

__all__ = [
    "MAX_N_TAU",
    "TAU_RTOL",
    "dynamic_generator_correction",
    "rpa_dynamic_kernel",
]

LOGGER = logging.getLogger(__name__)

TAU_RTOL = 1e-6
"""Relative tolerance of the adaptive imaginary-time quadrature."""

MAX_N_TAU = 1024
"""Maximum number of nodes of the adaptive imaginary-time quadrature."""

# Smallest population (relative to the largest one) kept when the corrected
# reduced state is converted back into a generator.
_POPULATION_FLOOR = 1e-14

# Relative tolerance to tell degenerate levels apart in the secular/admixture
# split of the dynamical correction.
_ADMIXTURE_TOL = 1e-9


def _dynamic_correction(
    k_mf: Operator,
    structure: RPAStructure,
    sites: frozenset,
    n_tau: int,
    environment: str = "isolated",
) -> Optional[Operator]:
    r"""Dynamical RPA correction to the mean-field generator.

    The imaginary-time quadrature starts with ``n_tau`` nodes, which are
    doubled until the correction converges (see :data:`TAU_RTOL` and
    :data:`MAX_N_TAU`): at low temperature the kernel varies on scales
    :math:`1/\omega` much shorter than the imaginary-time interval.
    """
    system = structure.system
    modes = environment_modes(structure, sites, environment)
    if modes is None:
        return None
    block = tuple(sorted(sites))
    k0_qobj: Qobj = k_mf.to_qutip(block)
    k0 = k0_qobj.full()
    d_ops = [
        LocalOperator(site, e, system).to_qutip(block).full()
        for site in modes.sites_a
        for e in modes.local_basis[site]
    ]

    def correction(n_nodes: int) -> Optional[np.ndarray]:
        """Return the correction computed with ``n_nodes`` nodes, if any."""
        nodes, weights = _gauss_legendre(n_nodes)
        sigma_tau = _kernel_at(modes, nodes)
        if not np.any(np.abs(sigma_tau) > 1e-14):
            return None
        return dynamic_generator_correction(k0, d_ops, sigma_tau, nodes, weights)

    n_nodes = max(int(n_tau), 1)
    d_k = correction(n_nodes)
    if d_k is None:
        return None
    while True:
        if n_nodes >= MAX_N_TAU:
            LOGGER.warning(
                "The imaginary-time quadrature of the dynamical RPA did not "
                "converge with %d nodes.",
                n_nodes,
            )
            break
        n_nodes = min(2 * n_nodes, MAX_N_TAU)
        d_k_new = correction(n_nodes)
        assert d_k_new is not None
        change = float(np.linalg.norm(d_k_new - d_k))
        d_k = d_k_new
        if change <= TAU_RTOL * max(float(np.linalg.norm(d_k)), 1.0):
            break
    return QutipOperator(
        Qobj(d_k, dims=k0_qobj.dims),
        system,
        {site: pos for pos, site in enumerate(block)},
    )


def _gauss_legendre(n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on ``(0, 1)``."""
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _kernel_at(environment, nodes: np.ndarray) -> np.ndarray:
    """Kernel ``Sigma(u)`` of the bosonized environment at the nodes ``u``."""
    # B_mu = sum_m c_{mu m} Psi_m, Sigma(tau) = sum_m c c^* f_m exp(-lam_m tau)
    c_mat, lam = environment.c_mat, environment.lam
    omega = np.abs(lam)
    expo = np.where(
        lam[None, :] > 0,
        -omega[None, :] * nodes[:, None],
        -omega[None, :] * (1.0 - nodes[:, None]),
    )
    occupation = np.exp(expo) / (-np.expm1(-omega))[None, :]
    return np.einsum("im,tm,jm->tij", c_mat, occupation, c_mat.conj())


def rpa_dynamic_kernel(
    structure: RPAStructure,
    sites: frozenset,
    n_tau: int = 32,
    environment: str = "isolated",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List], List[str]]:
    r"""Compute the imaginary-time kernel :math:`\Sigma_{\mu\nu}(\tau)`.

    The environment is bosonized site by site (see
    :func:`local_boson_modes`) and the resulting quadratic Hamiltonian
    :math:`H_b` is diagonalized with a Bogoliubov transformation. The
    kernel is then a sum of exponentials over the normal modes
    (see the module documentation), which is equivalent to the RPA dressing
    summed over all the Matsubara frequencies.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA. The frequency of the static
        bosons is ``structure.static_gap``.
    sites : frozenset
        Sites of the subsystem ``A``.
    n_tau : int, optional
        Number of Gauss-Legendre nodes in :math:`(0, 1)`. The default is 32.
    environment : str, optional
        Origin of the correlations of the environment: ``"isolated"``
        (default, the environment with ``A`` removed) or ``"global"`` (the
        bosonization of the whole system). See
        :func:`~qalma.meanfield.rpa.environment.environment_modes`.

    Returns
    -------
    sigma_tau : numpy.ndarray
        Complex array ``(n_tau, nA, nA)`` with :math:`\Sigma_{\mu\nu}(u_j)`.
    nodes, weights : numpy.ndarray
        Gauss-Legendre nodes and weights on :math:`(0, 1)`.
    local_basis : dict
        Centred local basis of each site.
    sites_a : list of str
        Sites of ``A`` in the order used to index ``sigma_tau``.

    Raises
    ------
    ValueError
        If the RPA is unstable.
    """
    nodes, weights = _gauss_legendre(n_tau)
    modes = environment_modes(structure, sites, environment)
    if modes is None:
        _, _, sites_a, _ = _split_sites(structure, sites)
        n_a = sum(len(structure.local_basis[site]) for site in sites_a)
        empty = np.zeros((n_tau, n_a, n_a), dtype=complex)
        return empty, nodes, weights, structure.local_basis, sites_a
    sigma_tau = _kernel_at(modes, nodes)
    return sigma_tau, nodes, weights, modes.local_basis, modes.sites_a


# ### Dynamical correction of the generator ###


def _exp_weight(delta: np.ndarray, length) -> np.ndarray:
    """Return ``(1 - exp(-length |delta|)) / |delta|``, or ``length`` if ``delta = 0``."""
    abs_delta = np.abs(delta)
    small = abs_delta < _DEGENERACY_TOL
    safe = np.where(small, 1.0, abs_delta)
    return np.where(small, length, -np.expm1(-length * safe) / safe)


def dynamic_generator_correction(
    k0: np.ndarray,
    d_ops: List[np.ndarray],
    sigma_tau: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    r"""Compute the generator correction from an imaginary-time kernel.

    In the eigenbasis :math:`\{|m\rangle\}` of :math:`K_0 = K_{A,MF}`, with
    eigenvalues :math:`\epsilon_m`, the second-order contribution of the
    environment to :math:`e^{-K_A'}` is

    .. math::

        Y_{ij} = \sum_k \int_0^1 du\; [\Sigma_{\mu\nu}(u)\,
        (\delta e_\mu)_{ik}(\delta e_\nu)_{kj}]\, e^{-u\epsilon_k}
        \int_0^{1-u} ds\; e^{-s\epsilon_i-(1-u-s)\epsilon_j} .

    The terms whose intermediate state :math:`k` is not below both
    :math:`i` and :math:`j` (*secular* terms, which contain the shifts of
    the levels and grow with :math:`\beta`) are exponentiated, inverting
    the derivative of the matrix exponential:
    :math:`\delta K_{ij} = -Y^{sec}_{ij}/w_{ij}`, with
    :math:`w_{ij} = \int_0^1 ds\, e^{-s\epsilon_i-(1-s)\epsilon_j}`. The
    terms with :math:`\epsilon_k < \min(\epsilon_i,\epsilon_j)`
    (*admixture* terms, the populations that the coupling to the
    environment transfers from :math:`k` to higher levels) are added to the
    density, with the weight of :math:`k` dressed by the secular shift:

    .. math::

        e^{-K_A'} = e^{-(K_0+\delta K)} + \sum_k e^{-\epsilon'_k} R_k ,
        \qquad R_k = e^{\epsilon_k} Y^{adm,k},
        \qquad \epsilon'_k = \epsilon_k + \delta K_{kk}.

    Both forms coincide at second order in the coupling. At high
    temperature the admixture is negligible and this is the first-order
    (Kubo-Mori) correction of the generator; at low temperature the
    admixture gives the populations of the excited states, which a
    correction of the generator at first order cannot describe. All the
    exponentials are evaluated relative to the smallest exponent involved,
    so there is no overflow nor underflow.

    Parameters
    ----------
    k0 : numpy.ndarray
        Dense Hermitian generator :math:`K_{A,MF}` of the subsystem.
    d_ops : list of numpy.ndarray
        Dense centred operators :math:`\delta e_\mu` acting on ``A``.
    sigma_tau : numpy.ndarray
        Kernel ``(n_tau, n_mu, n_mu)`` evaluated at ``nodes``.
    nodes, weights : numpy.ndarray
        Quadrature on :math:`(0, 1)`.

    Returns
    -------
    numpy.ndarray
        Hermitian matrix :math:`\delta K` such that
        :math:`K_A' = K_{A,MF} + \delta K` (in the original basis).
    """
    eps, vecs = np.linalg.eigh(k0)
    eps = eps - eps.min()
    q_ops = np.array([vecs.conj().T @ op @ vecs for op in d_ops])
    delta = eps[:, None] - eps[None, :]
    floor = np.minimum(eps[:, None], eps[None, :])  # min(eps_i, eps_j)
    # gap[i, k, j] = eps_k - min(eps_i, eps_j)
    gap = eps[None, :, None] - floor[:, None, :]
    # Degenerate levels are secular: only clearly lower intermediate levels
    # (beyond the numerical noise of the eigenvalues) give admixture terms.
    tolerance = _ADMIXTURE_TOL * (1.0 + float(eps.max()))
    admixture = gap < -tolerance

    # Y^sec scaled by exp(min(eps_i, eps_j)); R_k = exp(eps_k) Y^{adm, k}.
    y_sec = np.zeros(k0.shape, dtype=complex)
    r_adm = np.zeros((len(eps),) + k0.shape, dtype=complex)
    for u, w_q, sigma_u in zip(nodes, weights, sigma_tau):
        c_ops = np.einsum("mn,mik->nik", sigma_u, q_ops)
        amplitudes = np.einsum("nik,nkj->ikj", c_ops, q_ops)
        tail = _exp_weight(delta, 1.0 - u)
        secular = np.where(admixture, 0.0, np.exp(-u * np.where(admixture, 0.0, gap)))
        mixed = np.where(
            admixture, np.exp((1.0 - u) * np.where(admixture, gap, 0.0)), 0.0
        )
        y_sec += w_q * np.einsum("ikj,ikj,ij->ij", amplitudes, secular, tail)
        r_adm += w_q * np.einsum("ikj,ikj,ij->kij", amplitudes, mixed, tail)

    d_sec = -y_sec / _exp_weight(delta, 1.0)
    d_sec = 0.5 * (d_sec + d_sec.conj().T)
    if not np.any(admixture):
        return vecs @ d_sec @ vecs.conj().T

    k_sec = np.diag(eps) + d_sec
    lam_sec, u_sec = np.linalg.eigh(k_sec)
    eps_dressed = np.real(np.diag(k_sec))
    shift = min(lam_sec.min(), eps_dressed.min())
    rho = (u_sec * np.exp(-(lam_sec - shift))) @ u_sec.conj().T
    rho += np.einsum("k,kij->ij", np.exp(-(eps_dressed - shift)), r_adm)
    rho = 0.5 * (rho + rho.conj().T)
    pops, u_rho = np.linalg.eigh(rho)
    if pops.min() < -1e-8 * pops.max():
        LOGGER.debug(
            "Dynamical RPA: negative eigenvalue %.2e of the second-order state "
            "(relative to %.2e); it is clipped.",
            pops.min(),
            pops.max(),
        )
    pops = np.clip(pops, pops.max() * _POPULATION_FLOOR, None)
    # Undo the shift of the exponents, so that without admixture this is
    # exactly the secular correction.
    k_new = -(u_rho * (np.log(pops) - shift)) @ u_rho.conj().T
    d_k = k_new - np.diag(eps)
    d_k = 0.5 * (d_k + d_k.conj().T)
    return vecs @ d_k @ vecs.conj().T
