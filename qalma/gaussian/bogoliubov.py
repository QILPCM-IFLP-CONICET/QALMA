r"""Bogoliubov transformations of quadratic bosonic forms
:cite:p:`RingSchuck1980`.

The ``n`` bosonic modes are collected in
:math:`\Phi = (b_1,\dots,b_n,b_1^\dagger,\dots,b_n^\dagger)^T` and a quadratic
form is :math:`K = \frac12\Phi^\dagger \mathcal{H}\Phi`, with
:math:`\mathcal{H} = \begin{pmatrix}A & B\\ B^* & A^*\end{pmatrix}`. The real
quadratures are :math:`R = (x_1,\dots,x_n,p_1,\dots,p_n)^T`, with
:math:`x = (b+b^\dagger)/\sqrt2` and :math:`p = (b-b^\dagger)/(i\sqrt2)`.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.linalg

__all__ = [
    "ZERO_MODE_RTOL",
    "ZeroModeData",
    "bogoliubov_transformation",
    "bogoliubov_reduced",
    "quadrature_map",
    "symplectic_spectrum",
]

ZERO_MODE_RTOL = 1e-5
"""Default relative tolerance to identify the zero modes of a quadratic form."""

LOGGER = logging.getLogger(__name__)


def bogoliubov_transformation(h_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Diagonalize a bosonic quadratic Hamiltonian.

    Given :math:`H = \frac12\Phi^\dagger \mathcal{H}\Phi` with
    :math:`\Phi = (a, a^\dagger)^T` and :math:`\mathcal{H}` positive
    definite, find :math:`T` such that :math:`\Phi = T\tilde\Psi`, with
    :math:`[\tilde\Psi, \tilde\Psi^\dagger] = \tilde\eta` and
    :math:`H = \frac12 \tilde\Psi^\dagger |\Lambda| \tilde\Psi`
    (Colpa's algorithm).

    Parameters
    ----------
    h_mat : numpy.ndarray
        Hermitian positive definite matrix :math:`\mathcal{H}`.

    Returns
    -------
    lam : numpy.ndarray
        Signed mode frequencies :math:`\lambda_m`; the sign is the
        commutator sign :math:`\tilde\eta_m` of the mode.
    t_mat : numpy.ndarray
        Transformation :math:`T`.

    Raises
    ------
    ValueError
        If ``h_mat`` is not positive definite (unstable quadratic form).
    """
    n_half = h_mat.shape[0] // 2
    try:
        chol = np.linalg.cholesky(h_mat)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "Unstable quadratic form: the bosonic Hamiltonian is not "
            "positive definite."
        ) from err
    k_mat = chol.conj().T
    eta = np.diag(np.concatenate([np.ones(n_half), -np.ones(n_half)]))
    lam, vecs = np.linalg.eigh(k_mat @ eta @ k_mat.conj().T)
    t_mat = np.linalg.solve(k_mat, vecs) * np.sqrt(np.abs(lam))[None, :]
    return lam, t_mat


def quadrature_map(n_modes: int) -> np.ndarray:
    """Matrix ``W`` such that ``(b, b^dag) = W (x, p)``, with unit commutator."""
    eye = np.eye(n_modes)
    return np.block([[eye, 1j * eye], [eye, -1j * eye]]) / np.sqrt(2.0)


def symplectic_spectrum(sigma: np.ndarray) -> np.ndarray:
    r"""Return the symplectic eigenvalues :math:`\nu_\alpha` of a covariance matrix."""
    n_sub = sigma.shape[0] // 2
    if n_sub == 0:
        return np.zeros(0)
    eye, zero = np.eye(n_sub), np.zeros((n_sub, n_sub))
    omega = np.block([[zero, eye], [-eye, zero]])
    values = np.sort(np.abs(np.linalg.eigvals(1j * omega @ sigma)))
    return values[::2]


@dataclass
class ZeroModeData:
    """Conserved charges conjugate to the zero modes of the bosonic problem.

    In real quadratures, ``H = 1/2 R^T M R``. Each zero mode ``Q`` (null
    direction of ``M``) has a conjugate conserved charge ``P`` along the
    direction ``s0p``. Writing ``R = Z R_r + N Q + s0p P`` (``Z`` the basis of
    the symplectic quotient), the Hamiltonian is
    ``H = H_r(R_r) + P (drive . R_r) + 1/2 P^T kappa P``.

    Attributes
    ----------
    s0p : numpy.ndarray
        Real ``(2 n, m)`` matrix with the directions of the charges.
    kappa : numpy.ndarray
        ``(m, m)`` matrix ``s0p^T M s0p``.
    kappa_eff : numpy.ndarray
        ``kappa`` minus the response of the rest of the modes to the drive:
        the inverse covariance of the charges.
    drive_rows : numpy.ndarray
        ``(m, 2 n)`` real rows ``(M s0p)^T``: the drive of the charges on the
        quadratures ``R`` (the zero modes do not contribute).
    """

    s0p: np.ndarray
    kappa: np.ndarray
    kappa_eff: np.ndarray
    drive_rows: np.ndarray


def bogoliubov_reduced(
    h_mat: np.ndarray,
    zero_rtol: float = ZERO_MODE_RTOL,
    static: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, Optional[ZeroModeData]]:
    r"""Diagonalize a bosonic quadratic Hamiltonian, removing its zero modes.

    If :math:`\mathcal{H}` is positive definite this is :func:`bogoliubov_transformation`.
    If it is positive semidefinite, its null vectors are zero modes: the
    Hamiltonian does not depend on a collective coordinate :math:`Q` (a
    broken continuous symmetry, as the Goldstone mode of an ordered state),
    whose conjugate momentum :math:`P` is conserved. In the basis of real
    quadratures :math:`R=(x,p)`, with :math:`H=\frac12 R^T M R`, the zero
    modes span an isotropic subspace :math:`N=\ker M`. The reduced problem
    lives in the symplectic quotient :math:`N^\Omega/N`, where :math:`M` is
    positive definite and Colpa's algorithm applies: the orientation
    :math:`Q` is fixed and the sector :math:`P=0` is taken. The symmetry
    must be restored afterwards, averaging over the orbit of the broken
    symmetry (for instance with the ``symmetry_projections`` of the state).

    Parameters
    ----------
    h_mat : numpy.ndarray
        Hermitian positive semidefinite matrix :math:`\mathcal{H}`.
    zero_rtol : float, optional
        Eigenvalues of :math:`\mathcal{H}` with absolute value below
        ``zero_rtol`` times its norm are considered zero. If ``h_mat``
        comes from an approximate self-consistent solution (as the RPA
        around a mean-field state), a soft mode below this threshold cannot
        be distinguished from a zero mode. The default is 1e-5.
    static : numpy.ndarray, optional
        Boolean array with the bosons that are static (classical) modes.
        Their frequencies are tiny by construction, so the eigenvectors of
        ``h_mat`` that live mostly on them are not zero modes.

    Returns
    -------
    lam : numpy.ndarray
        Signed frequencies of the normal modes of the reduced problem
        (``2 (n - m)`` values for ``m`` zero modes).
    t_mat : numpy.ndarray
        Matrix of shape ``(2 n, 2 (n - m))`` with
        :math:`\Phi \simeq T\tilde\Psi`: the contribution of the zero modes
        is dropped.
    n_zero : int
        Number of zero modes removed.
    zero_data : ZeroModeData or None
        Data of the conserved charges conjugate to the zero modes.

    Raises
    ------
    ValueError
        If ``h_mat`` has negative eigenvalues (unstable quadratic form), or if the zero
        modes do not form an isotropic subspace.
    """
    n_modes = h_mat.shape[0] // 2
    if n_modes == 0:
        lam, t_mat = bogoliubov_transformation(h_mat)
        return lam, t_mat, 0, None
    scale = max(float(np.linalg.norm(h_mat, 2)), 1e-300)
    tol = zero_rtol * scale
    eigenvalues, eigenvectors = np.linalg.eigh(h_mat)
    if eigenvalues[0] < -tol:
        raise ValueError(
            "Unstable quadratic form: the bosonic Hamiltonian is not "
            "positive semidefinite."
        )
    static_both = (
        np.zeros(2 * n_modes, dtype=bool)
        if static is None
        else np.concatenate([static, static])
    )
    candidates = eigenvalues <= tol
    static_weight = np.sum(
        np.abs(eigenvectors[static_both][:, candidates]) ** 2, axis=0
    )
    n_zero = int(np.sum(static_weight < 0.5))
    if n_zero == 0:
        lam, t_mat = bogoliubov_transformation(h_mat)
        return lam, t_mat, 0, None

    w_map = quadrature_map(n_modes)
    m_complex = w_map.conj().T @ h_mat @ w_map
    if np.max(np.abs(m_complex.imag)) > 1e-8 * scale:
        raise ValueError(
            "The bosonic Hamiltonian has not the structure of a quadratic form."
        )
    m_real = m_complex.real
    m_real = 0.5 * (m_real + m_real.T)
    values, vectors = np.linalg.eigh(m_real)
    static_quadrature = (
        np.zeros(2 * n_modes, dtype=bool)
        if static is None
        else np.concatenate([static, static])
    )
    weight = np.sum(vectors[static_quadrature] ** 2, axis=0)
    null = vectors[:, (np.abs(values) <= tol) & (weight < 0.5)]
    if null.shape[1] != n_zero:
        raise ValueError("Inconsistent number of zero modes of the quadratic form.")
    eye, zero = np.eye(n_modes), np.zeros((n_modes, n_modes))
    omega = np.block([[zero, eye], [-eye, zero]])
    if np.max(np.abs(null.T @ omega @ null)) > 1e-7:
        raise ValueError(
            "The zero modes of the quadratic form do not form an isotropic subspace "
            "(they are not associated with a broken continuous symmetry)."
        )
    # Complement of the zero modes in their symplectic-orthogonal subspace
    complement = scipy.linalg.null_space(np.vstack([null.T @ omega, null.T]))
    n_reduced = n_modes - null.shape[1]
    if complement.shape[1] != 2 * n_reduced:
        raise ValueError(
            "Unable to reduce the quadratic form to the symplectic quotient."
        )
    # Symplectic (canonical) basis of the complement
    omega_c = complement.T @ omega @ complement
    schur_t, schur_q = scipy.linalg.schur(omega_c, output="real")
    x_cols, p_cols = [], []
    for block in range(n_reduced):
        a_val = schur_t[2 * block, 2 * block + 1]
        q_x, q_p = schur_q[:, 2 * block], schur_q[:, 2 * block + 1]
        scale_q = 1.0 / np.sqrt(abs(a_val))
        if a_val > 0:
            x_cols.append(scale_q * q_x)
            p_cols.append(scale_q * q_p)
        else:
            x_cols.append(scale_q * q_p)
            p_cols.append(scale_q * q_x)
    z_mat = complement @ np.column_stack(x_cols + p_cols)
    eye_r, zero_r = np.eye(n_reduced), np.zeros((n_reduced, n_reduced))
    omega_r = np.block([[zero_r, eye_r], [-eye_r, zero_r]])
    if np.max(np.abs(z_mat.T @ omega @ z_mat - omega_r)) > 1e-7:
        raise ValueError(
            "Unable to build a canonical basis of the symplectic quotient."
        )
    m_reduced = z_mat.T @ m_real @ z_mat
    m_reduced = 0.5 * (m_reduced + m_reduced.T)
    if np.linalg.eigvalsh(m_reduced)[0] <= 0.0:
        raise ValueError(
            "Unstable quadratic form: besides the zero modes, the bosonic "
            "Hamiltonian is not positive definite."
        )
    w_reduced = quadrature_map(n_reduced)
    lam, t_reduced = bogoliubov_transformation(
        w_reduced @ m_reduced @ w_reduced.conj().T
    )
    t_mat = w_map @ z_mat @ w_reduced.conj().T @ t_reduced

    # Conserved charges conjugate to the zero modes, symplectically orthogonal
    # to the quotient basis: P = N^T Omega R.
    s0 = -omega @ null
    s0p = s0 + z_mat @ (omega_r @ (z_mat.T @ omega @ s0))
    if (
        np.max(np.abs(z_mat.T @ omega @ s0p)) > 1e-7
        or np.max(np.abs(null.T @ omega @ s0p - np.eye(n_zero))) > 1e-7
        or np.max(np.abs(s0p.T @ omega @ s0p)) > 1e-7
    ):
        raise ValueError("Unable to build the charges conjugate to the zero modes.")
    kappa = s0p.T @ m_real @ s0p
    drive = z_mat.T @ m_real @ s0p
    kappa_eff = kappa - drive.T @ np.linalg.solve(m_reduced, drive)
    zero_data = ZeroModeData(
        s0p=s0p,
        kappa=0.5 * (kappa + kappa.T),
        kappa_eff=0.5 * (kappa_eff + kappa_eff.T),
        drive_rows=(m_real @ s0p).T,
    )
    LOGGER.warning(
        "%d zero mode(s) of the bosonic Hamiltonian were removed; "
        "the associated broken symmetry must be restored.",
        n_zero,
    )
    return lam, t_mat, n_zero, zero_data
