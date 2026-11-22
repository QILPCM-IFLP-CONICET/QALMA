"""Bosonized environment of a subsystem in the RPA."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from qutip import Qobj

from qalma.gaussian.bogoliubov import bogoliubov_reduced, quadrature_map

from .rpa_structure import RPAStructure, _split_sites

__all__ = ["ENVIRONMENTS", "environment_modes", "rpa_zero_modes", "static_kernel"]

ENVIRONMENTS = ("isolated", "global")
"""Origins of the correlations of the environment in the RPA corrections."""

LOGGER = logging.getLogger(__name__)


@dataclass
class _ZeroModes:
    """Charges conjugate to the zero modes of the environment.

    Attributes
    ----------
    e_coupling : numpy.ndarray
        ``(nA, m)``: coupling ``B_mu = ... + e_coupling[mu, a] P_a`` of the
        boundary operators to the charges.
    c_drive : numpy.ndarray
        ``(m, 2 n_reduced)``: coefficients of the drive of the charges on the
        normal modes, as ``c_mat`` for the boundary operators.
    kappa_eff : numpy.ndarray
        Inverse covariance of the charges.
    kappa : numpy.ndarray
        Bare inverse covariance of the charges.
    """

    e_coupling: np.ndarray
    c_drive: np.ndarray
    kappa_eff: np.ndarray
    kappa: np.ndarray


@dataclass
class _EnvironmentBogoliubov:
    """Normal modes of the environment (the subsystem removed).

    ``B_mu = sum_m c_mat[mu, m] Psi_m`` for the normal-mode operators
    ``Psi_m`` of signed frequency ``lam[m]``.
    """

    sites_a: List[str]
    local_basis: Dict[str, List[Qobj]]
    c_mat: np.ndarray
    lam: np.ndarray
    n_zero: int = 0
    zero: Optional["_ZeroModes"] = None


def _environment_bogoliubov(
    structure: RPAStructure, sites: frozenset
) -> Optional[_EnvironmentBogoliubov]:
    """Bosonize the environment of ``sites`` and diagonalize it.

    Returns ``None`` if the subsystem is not coupled to the environment.
    """
    local_basis = structure.local_basis
    m_full = structure.m_full
    idx_a, idx_e, sites_a, sites_e = _split_sites(structure, sites)
    if not idx_a or not idx_e:
        return None
    freqs_list = []
    row = 0
    blocks = []
    for site in sites_e:
        size = len(local_basis[site])
        freqs, coup = structure.boson_modes(site)
        freqs_list.append(freqs)
        blocks.append((row, size, coup))
        row += size
    freqs_all = np.concatenate(freqs_list)
    n_modes = len(freqs_all)
    if n_modes == 0:
        return None
    g_mat = np.zeros((row, n_modes), dtype=complex)
    col = 0
    for r_0, size, coup in blocks:
        g_mat[r_0 : r_0 + size, col : col + coup.shape[1]] = coup
        col += coup.shape[1]

    # Quadratic bosonic Hamiltonian of the environment and its modes.
    m_ae = m_full[np.ix_(idx_a, idx_e)]
    m_ee = m_full[np.ix_(idx_e, idx_e)]
    a_mat = np.diag(freqs_all) + g_mat.conj().T @ m_ee @ g_mat
    b_mat = g_mat.conj().T @ m_ee @ g_mat.conj()
    a_mat = 0.5 * (a_mat + a_mat.conj().T)
    b_mat = 0.5 * (b_mat + b_mat.T)
    h_mat = np.block([[a_mat, b_mat], [b_mat.conj(), a_mat.conj()]])
    static_modes = np.concatenate(
        [freqs <= structure.static_gap * (1.0 + 1e-6) for freqs in freqs_list]
    )
    try:
        lam, t_mat, n_zero, zero_data = bogoliubov_reduced(
            h_mat, structure.zero_rtol, static_modes
        )
    except ValueError as err:
        raise ValueError(f"RPA of the environment: {err}") from err
    g_b = m_ae @ g_mat
    s_phi = np.hstack([g_b, g_b.conj()])
    c_mat = s_phi @ t_mat
    zero = None
    if zero_data is not None:
        w_map = quadrature_map(n_modes)
        s_hat = s_phi @ w_map  # real rows of the operators in the quadratures
        zero = _ZeroModes(
            e_coupling=s_hat.real @ zero_data.s0p,
            c_drive=(zero_data.drive_rows @ w_map.conj().T) @ t_mat,
            kappa_eff=zero_data.kappa_eff,
            kappa=zero_data.kappa,
        )
    return _EnvironmentBogoliubov(sites_a, local_basis, c_mat, lam, n_zero, zero)


@dataclass
class _GlobalModes:
    """Normal modes of the bosonization of the whole system.

    ``g_mat`` maps the centred local basis of all the sites (rows indexed as
    the coupling matrix) to the local bosons, as in the environment.
    """

    g_mat: np.ndarray
    lam: np.ndarray
    t_mat: np.ndarray
    n_zero: int


def _global_modes(structure: RPAStructure) -> Optional[_GlobalModes]:
    """Bosonize the whole system, static modes included, and diagonalize it.

    The result does not depend on the subsystem, and it is cached in the
    structure.
    """
    # pylint: disable=protected-access
    if structure._global_modes is not None:
        return structure._global_modes
    freqs_list, columns = [], []
    for site, start in structure.positions.items():
        freqs, coup = structure.boson_modes(site)
        freqs_list.append(freqs)
        columns.append((start, coup))
    freqs_all = np.concatenate(freqs_list) if freqs_list else np.zeros(0)
    n_modes = len(freqs_all)
    if n_modes == 0:
        return None
    g_mat = np.zeros((structure.m_full.shape[0], n_modes), dtype=complex)
    col = 0
    for start, coup in columns:
        g_mat[start : start + coup.shape[0], col : col + coup.shape[1]] = coup
        col += coup.shape[1]
    m_full = structure.m_full
    a_mat = np.diag(freqs_all) + g_mat.conj().T @ m_full @ g_mat
    b_mat = g_mat.conj().T @ m_full @ g_mat.conj()
    a_mat = 0.5 * (a_mat + a_mat.conj().T)
    b_mat = 0.5 * (b_mat + b_mat.T)
    h_mat = np.block([[a_mat, b_mat], [b_mat.conj(), a_mat.conj()]])
    static_modes = freqs_all <= structure.static_gap * (1.0 + 1e-6)
    try:
        lam, t_mat, n_zero, _ = bogoliubov_reduced(
            h_mat, structure.zero_rtol, static_modes
        )
    except ValueError as err:
        raise ValueError(f"RPA of the whole system: {err}") from err
    structure._global_modes = _GlobalModes(g_mat, lam, t_mat, n_zero)
    return structure._global_modes


def _global_environment_bogoliubov(
    structure: RPAStructure, sites: frozenset
) -> Optional[_EnvironmentBogoliubov]:
    """Operators ``B_mu`` of ``sites`` on the normal modes of the whole system.

    The correlations of the environment are those of the bosonization of the
    whole system (the subsystem included), as in :cite:t:`MateraThesis`
    (chapter 6), instead of those of the environment with the subsystem
    removed. The Bogoliubov transformation is shared by all the subsystems.
    """
    idx_a, idx_e, sites_a, _ = _split_sites(structure, sites)
    if not idx_a or not idx_e:
        return None
    modes = _global_modes(structure)
    if modes is None:
        return None
    g_b = structure.m_full[np.ix_(idx_a, idx_e)] @ modes.g_mat[idx_e, :]
    c_mat = np.hstack([g_b, g_b.conj()]) @ modes.t_mat
    return _EnvironmentBogoliubov(
        sites_a, structure.local_basis, c_mat, modes.lam, modes.n_zero, None
    )


def environment_modes(
    structure: RPAStructure, sites: frozenset, environment: str = "isolated"
) -> Optional[_EnvironmentBogoliubov]:
    """Return the normal modes coupled to ``sites`` for an ``environment``.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites : frozenset
        Sites of the subsystem.
    environment : str, optional
        ``"isolated"`` (default) bosonizes the environment with the
        subsystem removed; ``"global"`` uses the bosonization of the whole
        system, computed once for all the subsystems.

    Returns
    -------
    _EnvironmentBogoliubov or None
        The modes, or ``None`` if the subsystem is not coupled to the
        environment.

    Raises
    ------
    ValueError
        If ``environment`` is not one of :data:`ENVIRONMENTS`.
    """
    if environment == "isolated":
        return _environment_bogoliubov(structure, sites)
    if environment == "global":
        return _global_environment_bogoliubov(structure, sites)
    raise ValueError(f"environment must be one of {ENVIRONMENTS}, not {environment!r}.")


def static_kernel(environment: _EnvironmentBogoliubov) -> np.ndarray:
    r"""Return the static kernel :math:`\int_0^1 \Sigma(\tau) d\tau` of the modes.

    For a normal mode of signed frequency :math:`\lambda_m`, the integral of
    its thermal correlation over the imaginary-time interval is
    :math:`1/|\lambda_m|`, so that
    :math:`\chi_{\mu\nu} = \sum_m c_{\mu m} c^*_{\nu m}/|\lambda_m|`.

    Parameters
    ----------
    environment : _EnvironmentBogoliubov
        Normal modes coupled to the subsystem.

    Returns
    -------
    numpy.ndarray
        Real symmetric kernel.
    """
    c_mat = environment.c_mat
    sigma = np.real(
        np.einsum("im,m,jm->ij", c_mat, 1.0 / np.abs(environment.lam), c_mat.conj())
    )
    return 0.5 * (sigma + sigma.T)


def rpa_zero_modes(structure: RPAStructure, sites: Optional[frozenset] = None) -> int:
    """Return the number of zero modes removed from the bosonic RPA problem.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites : frozenset, optional
        If given, count the zero modes of the environment of ``sites`` (the
        problem used by the reductions of a subsystem). Otherwise, count
        those of the whole system (the problem of the full bosonization).

    Returns
    -------
    int
        The number of zero modes.
    """
    if sites is None:
        return structure.boson_system().n_zero_modes
    environment = _environment_bogoliubov(structure, frozenset(sites))
    return 0 if environment is None else environment.n_zero
