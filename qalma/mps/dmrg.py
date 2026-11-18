r"""Two-site DMRG for a matrix-product operator, written with numpy/scipy.

This is a deliberately plain implementation: finite system, open boundaries,
no symmetries, no subspace expansion or noise. It is meant for moderate bond
dimensions and as a reference; the :class:`~qalma.mps.gs_operator.MPSGSOperator`
accepts any other MPS (for instance, one computed with a dedicated DMRG
package) through :meth:`~qalma.mps.gs_operator.MPSGSOperator.from_mps`.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from qalma.mps.mps import MPS, _truncation_rank

LOGGER = logging.getLogger(__name__)

__all__ = ["dmrg"]


def _update_left(
    env: np.ndarray | None, tensor: np.ndarray, w_op: np.ndarray
) -> np.ndarray:
    """Add one site to a left environment ``(bra, mpo, ket)``.

    ``new[b, y, c] = sum conj(A)[a, s, b] env[a, w, x] W[w, y, s, t] A[x, t, c]``,
    contracted pairwise with ``tensordot`` (``einsum`` does not use BLAS here).
    """
    if env is not None:
        tmp = np.tensordot(env, tensor, axes=([2], [0]))  # a w t c
    else:
        tmp = tensor.copy()
    tmp = np.tensordot(tmp, w_op, axes=([1, 2], [0, 3]))  # a c y s
    tmp = np.tensordot(tensor.conj(), tmp, axes=([0, 1], [0, 3]))  # b c y
    return tmp.transpose(0, 2, 1)


def _update_right(
    env: np.ndarray | None, tensor: np.ndarray, w_op: np.ndarray
) -> np.ndarray:
    """Add one site to a right environment ``(bra, mpo, ket)``.

    ``new[a, w, x] = sum conj(A)[a, s, b] env[b, y, c] W[w, y, s, t] A[x, t, c]``.
    """
    if env is not None:
        tmp = np.tensordot(env, tensor, axes=([2], [2]))  # b y x t
    else:
        tmp = tensor.copy()
    tmp = np.tensordot(tmp, w_op, axes=([1, 3], [1, 3]))  # b x w s
    tmp = np.tensordot(tensor.conj(), tmp, axes=([1, 2], [3, 0]))  # a x w
    return tmp.transpose(0, 2, 1)


def _apply_heff(
    theta: np.ndarray,
    left: np.ndarray,
    w_1: np.ndarray,
    w_2: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Apply the two-site effective Hamiltonian to ``theta[x, t, v, c]``."""
    tmp = np.tensordot(left, theta, axes=([2], [0]))  # a w t v c
    tmp = np.tensordot(tmp, w_1, axes=([1, 2], [0, 3]))  # a v c y s
    tmp = np.tensordot(tmp, w_2, axes=([1, 3], [3, 0]))  # a c s z u
    tmp = np.tensordot(tmp, right, axes=([1, 3], [2, 1]))  # a s u b
    return tmp


def _lowest_state(
    theta: np.ndarray,
    left: np.ndarray | None,
    w_1: np.ndarray,
    w_2: np.ndarray,
    right: np.ndarray | None,
    dense_threshold: int,
    tol: float,
) -> Tuple[float, np.ndarray]:
    """Lowest eigenpair of the effective Hamiltonian."""
    shape = theta.shape
    dim = theta.size

    def matvec(vec):
        return _apply_heff(vec.reshape(shape), left, w_1, w_2, right).reshape(-1)

    if dim <= dense_threshold:
        mat = np.empty((dim, dim), dtype=complex)
        basis = np.eye(dim, dtype=complex)
        for col in range(dim):
            mat[:, col] = matvec(basis[:, col])
        mat = 0.5 * (mat + mat.conj().T)
        vals, vecs = np.linalg.eigh(mat)
        return float(vals[0]), vecs[:, 0].reshape(shape)

    op = LinearOperator((dim, dim), matvec=matvec, dtype=complex)
    v_0 = theta.reshape(-1)
    nrm = np.linalg.norm(v_0)
    v_0 = v_0 / nrm if nrm > 0 else np.random.default_rng(0).normal(size=dim) + 0j
    vals, vecs = eigsh(op, k=1, which="SA", v0=v_0, tol=tol)
    return float(vals[0]), vecs[:, 0].reshape(shape)


def _dense_single_site(mpo: Sequence[np.ndarray]) -> Tuple[float, np.ndarray]:
    """Ground state of a one-site problem."""
    mat = mpo[0][0, 0]
    vals, vecs = np.linalg.eigh(0.5 * (mat + mat.conj().T))
    return float(vals[0]), vecs[:, 0]


def dmrg(
    mpo: Sequence[np.ndarray],
    chi_max: int = 64,
    max_sweeps: int = 30,
    tol: float = 1e-10,
    cutoff: float = 1e-12,
    chi_init: int = 4,
    seed: Optional[int] = None,
    mps0: Optional[MPS] = None,
    dense_threshold: int = 200,
    eig_tol: float = 1e-12,
) -> Tuple[MPS, Dict[str, Any]]:
    r"""Find the ground state of an MPO with the two-site DMRG algorithm.

    Parameters
    ----------
    mpo : sequence of ndarray
        MPO tensors ``W[p]`` of shape ``(left, right, row, column)``, see
        :func:`qalma.mps.mpo.build_mpo`. The operator must be Hermitian.
    chi_max : int, optional
        Maximum bond dimension. Default 64.
    max_sweeps : int, optional
        Maximum number of sweeps (a sweep goes from left to right and back).
    tol : float, optional
        Stop when the energy changes less than ``tol`` between sweeps.
    cutoff : float, optional
        Truncated weight allowed at every bond (before applying ``chi_max``).
    chi_init : int, optional
        Bond dimension of the random initial state (ignored with ``mps0``).
    seed : int, optional
        Seed of the random initial state.
    mps0 : MPS, optional
        Initial state. Default is a random MPS.
    dense_threshold : int, optional
        Effective problems up to this size are diagonalized densely.
    eig_tol : float, optional
        Tolerance of the iterative eigensolver.

    Returns
    -------
    mps : MPS
        Normalized approximation of the ground state, with its orthogonality
        center at position 0.
    info : dict
        ``energy``, ``energies`` (one per sweep), ``converged``, ``sweeps``,
        ``max_truncation`` (largest discarded weight at the last sweep) and
        ``bond_dims``.
    """
    n_sites = len(mpo)
    dims = [w.shape[2] for w in mpo]

    if n_sites == 1:
        energy, vec = _dense_single_site(mpo)
        state = MPS([vec.reshape(1, -1, 1)], center=0)
        return state, dict(
            energy=energy,
            energies=[energy],
            converged=True,
            sweeps=0,
            max_truncation=0.0,
            bond_dims=(),
        )

    mps = (
        mps0.copy()
        if mps0 is not None
        else MPS.random(dims, min(chi_init, chi_max), seed)
    )
    mps.move_center(0)
    tensors: List[np.ndarray] = mps.tensors

    left: List[Optional[np.ndarray]] = [None] * (n_sites + 1)
    right: List[Optional[np.ndarray]] = [None] * (n_sites + 1)
    left[0] = np.ones((1, 1, 1), dtype=complex)
    right[n_sites] = np.ones((1, 1, 1), dtype=complex)
    for pos in range(n_sites - 1, 0, -1):
        right[pos] = _update_right(right[pos + 1], tensors[pos], mpo[pos])

    def two_site(pos: int, direction: int) -> Tuple[float, float]:
        theta = np.tensordot(tensors[pos], tensors[pos + 1], axes=(2, 0))
        energy, theta = _lowest_state(
            theta,
            left[pos],
            mpo[pos],
            mpo[pos + 1],
            right[pos + 2],
            dense_threshold,
            eig_tol,
        )
        chi_l, d_1, d_2, chi_r = theta.shape
        u_mat, s_vals, vh_mat = np.linalg.svd(
            theta.reshape(chi_l * d_1, d_2 * chi_r), full_matrices=False
        )
        keep = _truncation_rank(s_vals, chi_max, cutoff)
        discarded = float(np.sum(s_vals[keep:] ** 2) / np.sum(s_vals**2))
        s_vals = s_vals[:keep] / np.linalg.norm(s_vals[:keep])
        u_mat, vh_mat = u_mat[:, :keep], vh_mat[:keep]
        if direction > 0:
            tensors[pos] = u_mat.reshape(chi_l, d_1, keep)
            tensors[pos + 1] = (s_vals[:, None] * vh_mat).reshape(keep, d_2, chi_r)
            left[pos + 1] = _update_left(left[pos], tensors[pos], mpo[pos])
        else:
            tensors[pos] = (u_mat * s_vals[None, :]).reshape(chi_l, d_1, keep)
            tensors[pos + 1] = vh_mat.reshape(keep, d_2, chi_r)
            right[pos + 1] = _update_right(
                right[pos + 2], tensors[pos + 1], mpo[pos + 1]
            )
        return energy, discarded

    energies: List[float] = []
    converged = False
    max_trunc = 0.0
    for sweep in range(max_sweeps):
        max_trunc = 0.0
        energy = 0.0
        for pos in range(n_sites - 1):
            energy, trunc = two_site(pos, +1)
            max_trunc = max(max_trunc, trunc)
        for pos in range(n_sites - 2, -1, -1):
            energy, trunc = two_site(pos, -1)
            max_trunc = max(max_trunc, trunc)
        energies.append(energy)
        LOGGER.info(
            "DMRG sweep %d: E=%.12f, truncation=%.2e, chi=%d",
            sweep,
            energy,
            max_trunc,
            max(t.shape[2] for t in tensors[:-1]),
        )
        if sweep >= 1 and abs(energies[-1] - energies[-2]) < tol:
            converged = True
            break

    state = MPS(tensors, center=0)
    state.normalize()
    return state, dict(
        energy=energies[-1],
        energies=energies,
        converged=converged,
        sweeps=len(energies),
        max_truncation=max_trunc,
        bond_dims=state.bond_dims,
    )
