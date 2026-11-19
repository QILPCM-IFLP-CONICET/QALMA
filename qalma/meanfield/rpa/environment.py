"""Bosonized environment of a subsystem in the RPA."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from qutip import Qobj

from qalma.gaussian.bogoliubov import bogoliubov_reduced, quadrature_map

from .rpa_structure import RPAStructure, _split_sites

__all__ = ["rpa_zero_modes"]

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
