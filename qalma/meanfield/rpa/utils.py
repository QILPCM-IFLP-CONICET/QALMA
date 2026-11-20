"""Single-site quantities of the RPA.

Kubo-Mori susceptibilities and bosonic modes of the local mean-field states.
The Bogoliubov transformations are in :mod:`qalma.gaussian`.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from qalma.gaussian.bogoliubov import ZERO_MODE_RTOL

_DEGENERACY_TOL = 1e-12
_STATIC_GAP = 1e-6
# The mean-field state is only self-consistent up to the tolerance of its
# solver, so the zero modes of the RPA have eigenvalues of the order 1e-9-1e-7
# (relative to the norm), with random sign. The first non-zero eigenvalues are
# orders of magnitude larger, so the generic default is adequate.
_ZERO_MODE_RTOL = ZERO_MODE_RTOL

__all__ = ["kubo_mori_susceptibility", "local_boson_modes"]

LOGGER = logging.getLogger(__name__)


# ### Local (single-site) quantities ###


def _spectral_data(
    sigma_loc: np.ndarray,
    basis: List[np.ndarray],
    spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None,
):
    """Return populations, ``-log`` populations and the basis in the eigenbasis.

    If ``spectrum = (populations, eigenvectors)`` is given, it is used
    instead of diagonalizing ``sigma_loc``. This allows to keep the accuracy
    of very small populations, which are lost in a density matrix.
    """
    if spectrum is None:
        p, vecs = np.linalg.eigh(sigma_loc)
    else:
        p, vecs = spectrum
    p = np.clip(p, 1e-300, None)
    ops = np.array([vecs.conj().T @ np.asarray(e) @ vecs for e in basis])
    return p, -np.log(p), ops


def kubo_mori_susceptibility(
    sigma_loc: np.ndarray,
    basis: List[np.ndarray],
    spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None,
):
    r"""Return the connected Kubo-Mori susceptibility of a local state.

    .. math::

        \chi^0_{\mu\nu} = \sum_{mn} \frac{p_m - p_n}{\ln p_m - \ln p_n}
        (e_\mu)_{mn}(e_\nu)_{nm} - \langle e_\mu\rangle\langle e_\nu\rangle

    where the quotient equals :math:`p_m` if :math:`p_m = p_n`.

    Parameters
    ----------
    sigma_loc : numpy.ndarray
        Local density matrix.
    basis : list of numpy.ndarray
        Hermitian local operators :math:`e_\mu`.
    spectrum : tuple of numpy.ndarray, optional
        Populations and eigenvectors of ``sigma_loc``, if known accurately.

    Returns
    -------
    numpy.ndarray
        Real symmetric matrix :math:`\chi^0_{\mu\nu}`.
    """
    p, eps, ops = _spectral_data(sigma_loc, basis, spectrum)
    delta = eps[:, None] - eps[None, :]
    small = np.abs(delta) < _DEGENERACY_TOL
    w = np.where(
        small,
        p[:, None] * np.ones_like(delta),
        (p[:, None] - p[None, :]) / np.where(small, 1.0, -delta),
    )
    avg = np.array([np.sum(p * np.diag(o)).real for o in ops])
    chi = np.einsum("mn,amn,bnm->ab", w, ops, ops).real
    return chi - np.outer(avg, avg)


def _ordered_spectrum(
    spectrum: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sort populations (decreasing) and eigenvectors of a local state."""
    p, vecs = spectrum
    order = np.argsort(-p)
    return np.clip(p[order], 1e-300, None), vecs[:, order]


def _boson_pairs(p: np.ndarray) -> List[Tuple[int, int, float]]:
    """Pairs ``(i, j, ln(p_i/p_j))``, ``i < j``, of the local bosons.

    This is the order of the modes of :func:`local_boson_modes`.
    """
    return [
        (i, j, float(np.log(p[i] / p[j])))
        for i in range(len(p))
        for j in range(i + 1, len(p))
    ]


def _is_dynamic(freqs, static_gap: float):
    """Whether bosons of frequency ``freqs`` are dynamic (not static)."""
    return np.asarray(freqs) > static_gap * (1.0 + 1e-6)


def local_boson_modes(
    sigma_loc: np.ndarray,
    basis: List[np.ndarray],
    static_gap: float = _STATIC_GAP,
    spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Bosonize the fluctuation operators of a single site.

    In the eigenbasis of ``sigma_loc`` (populations :math:`p_i\ge p_j` for
    :math:`i<j`) the off-diagonal operators are mapped to
    :math:`|i\rangle\langle j|\to\sqrt{p_i-p_j}\,a_{ij}`, and
    :math:`\delta e_\mu` is truncated to its part linear in the bosons:

    .. math::

        \delta e_\mu \simeq \sum_k g_{\mu k} a_k + g^*_{\mu k} a_k^\dagger .

    Pairs with :math:`\ln(p_i/p_j) <` ``static_gap`` and the diagonal
    (intraband) covariance of the populations are time independent. They are
    represented by bosons of frequency ``static_gap`` and couplings scaled
    so that the static susceptibility
    :math:`\sum_k (g_{\mu k} g^*_{\nu k} + c.c.)/\Delta_k` is exact.

    Parameters
    ----------
    sigma_loc : numpy.ndarray
        Local density matrix.
    basis : list of numpy.ndarray
        Hermitian local operators :math:`e_\mu` (centred or not).
    static_gap : float, optional
        Frequency assigned to the static modes. The default is 1e-6.
    spectrum : tuple of numpy.ndarray, optional
        Populations and eigenvectors of ``sigma_loc``, if known accurately
        (for instance, from the local generator). Needed when some
        populations are below the numerical precision of a density matrix.

    Returns
    -------
    freqs : numpy.ndarray
        Mode frequencies :math:`\Delta_k \ge` ``static_gap``, shape ``(K,)``.
    couplings : numpy.ndarray
        Complex couplings :math:`g_{\mu k}`, shape ``(len(basis), K)``.
    """
    if spectrum is None:
        populations, eigenvectors = np.linalg.eigh(sigma_loc)
        spectrum = (populations, eigenvectors)
    p, vecs = _ordered_spectrum(spectrum)
    ops = np.array([vecs.conj().T @ np.asarray(e) @ vecs for e in basis])
    freqs: List[float] = []
    couplings: List[np.ndarray] = []
    for i, j, gap in _boson_pairs(p):
        if gap >= static_gap:
            weight = p[i] - p[j]
        else:
            gap = static_gap
            weight = 0.5 * (p[i] + p[j]) * static_gap
        freqs.append(gap)
        couplings.append(np.sqrt(weight) * ops[:, i, j])
    diag = np.real(np.array([np.diag(o) for o in ops]))
    mean = diag @ p
    cov = (diag * p[None, :]) @ diag.T - np.outer(mean, mean)
    evals, evecs = np.linalg.eigh(cov)
    for val, vec in zip(evals, evecs.T):
        if val > 1e-14:
            freqs.append(static_gap)
            couplings.append(vec * np.sqrt(val * static_gap / 2.0))
    if not freqs:
        return np.zeros(0), np.zeros((len(basis), 0), dtype=complex)
    return np.array(freqs), np.array(couplings).T.astype(complex)
