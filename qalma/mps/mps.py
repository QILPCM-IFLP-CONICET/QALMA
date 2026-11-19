r"""Minimal matrix-product-state (MPS) container, written with numpy only.

An MPS over ``N`` sites is stored as a list of tensors ``A[p]`` of shape
``(chi_left, d_p, chi_right)``, with ``chi_left = 1`` for the first tensor and
``chi_right = 1`` for the last one. The state is

.. math::

    |\psi\rangle = \sum_{s_0 \dots s_{N-1}}
        A^{s_0}_{0} A^{s_1}_{1} \cdots A^{s_{N-1}}_{N-1}
        |s_0 \dots s_{N-1}\rangle .

Position ``p`` refers to the order of the sites *along the chain*, which is
unrelated to the names of the sites of a
:class:`~qalma.model.SystemDescriptor`. The mapping between both is the
business of :class:`~qalma.mps.gs_operator.MPSGSOperator`.

The attribute :attr:`MPS.center` keeps track of the orthogonality center when
the state is in mixed-canonical form: tensors to the left of the center are
left-isometries, tensors to the right are right-isometries. ``None`` means
that nothing is assumed.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["MPS"]


class MPS:
    """Matrix product state with open boundary conditions.

    Parameters
    ----------
    tensors : sequence of ndarray
        Site tensors of shape ``(chi_left, d, chi_right)``.
    center : int or None, optional
        Position of the orthogonality center, if the tensors are in
        mixed-canonical form with respect to it. Default is ``None``.

    Attributes
    ----------
    tensors : list of ndarray
        The (complex) site tensors.
    center : int or None
        The orthogonality center, or ``None`` if unknown.
    """

    def __init__(self, tensors: Sequence[np.ndarray], center: Optional[int] = None):
        self.tensors: List[np.ndarray] = [
            np.asarray(tensor, dtype=complex) for tensor in tensors
        ]
        if not self.tensors:
            raise ValueError("An MPS needs at least one site.")
        for pos, tensor in enumerate(self.tensors):
            if tensor.ndim != 3:
                raise ValueError(f"Tensor {pos} must have 3 indices.")
            if pos and self.tensors[pos - 1].shape[2] != tensor.shape[0]:
                raise ValueError(
                    f"Bond dimension mismatch between {pos - 1} and {pos}."
                )
        if self.tensors[0].shape[0] != 1 or self.tensors[-1].shape[2] != 1:
            raise ValueError("Boundary bond dimensions must be 1.")
        self.center = center

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def random(cls, dims: Sequence[int], chi: int, seed=None) -> "MPS":
        """Build a random normalized MPS, with the center at position 0.

        Parameters
        ----------
        dims : sequence of int
            Local dimensions.
        chi : int
            Maximum bond dimension.
        seed : int, optional
            Seed of the random number generator.
        """
        rng = np.random.default_rng(seed)
        dims = [int(d) for d in dims]
        n_sites = len(dims)
        bonds = [1]
        for pos in range(1, n_sites):
            left = int(np.prod(dims[:pos], dtype=float).clip(max=1e12))
            right = int(np.prod(dims[pos:], dtype=float).clip(max=1e12))
            bonds.append(max(1, min(chi, left, right)))
        bonds.append(1)
        tensors = [
            rng.normal(size=(bonds[p], dims[p], bonds[p + 1]))
            + 1j * rng.normal(size=(bonds[p], dims[p], bonds[p + 1]))
            for p in range(n_sites)
        ]
        result = cls(tensors)
        result.move_center(0)
        result.normalize()
        return result

    @classmethod
    def product(cls, local_states: Sequence[np.ndarray]) -> "MPS":
        """Build a product state from a list of local vectors."""
        tensors = [np.asarray(v, dtype=complex).reshape(1, -1, 1) for v in local_states]
        return cls(tensors)

    @classmethod
    def from_dense(
        cls,
        vector: np.ndarray,
        dims: Sequence[int],
        chi_max: Optional[int] = None,
        cutoff: float = 0.0,
    ) -> "MPS":
        """Build an MPS from a dense state vector by successive SVDs.

        The first site is the most significant factor of the tensor product
        (C order), as in ``qutip.tensor``.

        Parameters
        ----------
        vector : ndarray
            State vector with ``prod(dims)`` entries.
        dims : sequence of int
            Local dimensions.
        chi_max : int, optional
            Maximum bond dimension. Default is no limit.
        cutoff : float, optional
            Discard Schmidt values whose squared weight (relative to the
            norm) sums less than ``cutoff``. Default is 0.
        """
        dims = [int(d) for d in dims]
        rest = np.asarray(vector, dtype=complex).reshape(1, -1)
        tensors = []
        left = 1
        for dim in dims[:-1]:
            mat = rest.reshape(left * dim, -1)
            u_mat, s_vals, vh_mat = np.linalg.svd(mat, full_matrices=False)
            keep = _truncation_rank(s_vals, chi_max, cutoff)
            tensors.append(u_mat[:, :keep].reshape(left, dim, keep))
            rest = s_vals[:keep, None] * vh_mat[:keep]
            left = keep
        tensors.append(rest.reshape(left, dims[-1], 1))
        return cls(tensors, center=len(dims) - 1)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def n_sites(self) -> int:
        """Number of sites."""
        return len(self.tensors)

    @property
    def dims(self) -> Tuple[int, ...]:
        """Local dimensions."""
        return tuple(t.shape[1] for t in self.tensors)

    @property
    def bond_dims(self) -> Tuple[int, ...]:
        """Bond dimensions of the ``N - 1`` internal bonds."""
        return tuple(t.shape[2] for t in self.tensors[:-1])

    def copy(self) -> "MPS":
        """Deep copy."""
        return MPS([t.copy() for t in self.tensors], self.center)

    def _shallow_copy(self) -> "MPS":
        """Copy sharing the tensors. Safe since tensors are never modified in place."""
        return MPS(list(self.tensors), self.center)

    def __repr__(self) -> str:
        return (
            f"MPS(n_sites={self.n_sites}, max_bond={max(self.bond_dims, default=1)}, "
            f"center={self.center})"
        )

    # ------------------------------------------------------------------
    # Canonical forms
    # ------------------------------------------------------------------
    def _shift_right(self, pos: int) -> None:
        """Make ``tensors[pos]`` a left-isometry, moving the remainder right."""
        tensor = self.tensors[pos]
        chi_l, dim, chi_r = tensor.shape
        q_mat, r_mat = np.linalg.qr(tensor.reshape(chi_l * dim, chi_r))
        self.tensors[pos] = q_mat.reshape(chi_l, dim, q_mat.shape[1])
        self.tensors[pos + 1] = np.tensordot(r_mat, self.tensors[pos + 1], axes=(1, 0))

    def _shift_left(self, pos: int) -> None:
        """Make ``tensors[pos]`` a right-isometry, moving the remainder left."""
        tensor = self.tensors[pos]
        chi_l, dim, chi_r = tensor.shape
        q_mat, r_mat = np.linalg.qr(tensor.reshape(chi_l, dim * chi_r).conj().T)
        self.tensors[pos] = q_mat.conj().T.reshape(q_mat.shape[1], dim, chi_r)
        self.tensors[pos - 1] = np.tensordot(
            self.tensors[pos - 1], r_mat.conj().T, axes=(2, 0)
        )

    def move_center(self, target: int) -> "MPS":
        """Bring the state to mixed-canonical form with center at ``target``.

        The state is not changed, only its representation. Operates in place
        and returns ``self``.
        """
        if not 0 <= target < self.n_sites:
            raise IndexError("center out of range")
        if self.center is None:
            for pos in range(target):
                self._shift_right(pos)
            for pos in range(self.n_sites - 1, target, -1):
                self._shift_left(pos)
        else:
            while self.center < target:
                self._shift_right(self.center)
                self.center += 1
            while self.center > target:
                self._shift_left(self.center)
                self.center -= 1
        self.center = target
        return self

    def norm(self) -> float:
        """Norm of the state."""
        work = self._shallow_copy()
        if work.center is None:
            work.move_center(0)
        center = work.center
        assert center is not None
        return float(np.linalg.norm(work.tensors[center]))

    def normalize(self) -> "MPS":
        """Normalize in place and return ``self``."""
        if self.center is None:
            self.move_center(0)
        center = self.center
        assert center is not None
        self.tensors[center] = self.tensors[center] / np.linalg.norm(
            self.tensors[center]
        )
        return self

    # ------------------------------------------------------------------
    # Contractions
    # ------------------------------------------------------------------
    def to_dense(self) -> np.ndarray:
        """Contract the MPS into a dense vector. Exponential in ``N``."""
        psi = self.tensors[0][0]  # (d0, chi)
        for tensor in self.tensors[1:]:
            psi = np.tensordot(psi, tensor, axes=(-1, 0))
            psi = psi.reshape(-1, psi.shape[-1])
        return psi.reshape(-1)

    def overlap(self, other: "MPS") -> complex:
        r"""Return :math:`\langle self | other \rangle`."""
        env = np.ones((1, 1), dtype=complex)  # (bra, ket)
        for bra, ket in zip(self.tensors, other.tensors):
            tmp = np.tensordot(env, ket, axes=([1], [0]))  # a s c
            env = np.tensordot(bra.conj(), tmp, axes=([0, 1], [0, 1]))  # b c
        return complex(env[0, 0])

    def reduced_density_matrix(self, positions: Sequence[int]) -> np.ndarray:
        r"""Reduced density matrix of the sites at ``positions``.

        The result is the dense matrix :math:`\rho_{IJ}` with ``I`` and ``J``
        running over the tensor product of the local spaces of the given
        positions, **sorted by position** (the first position is the most
        significant factor). It is normalized to unit trace.

        The cost is ``O(L chi^3 d)`` plus the size of the output, where ``L``
        is the distance between the first and the last position, since the
        state is brought to mixed-canonical form with the center at the first
        position and everything outside the interval contracts to the identity.

        Parameters
        ----------
        positions : sequence of int
            Positions to keep. Repeated entries are not allowed.

        Returns
        -------
        ndarray
            Matrix of shape ``(D, D)`` with ``D = prod(d_p)``.
        """
        keep = sorted(int(p) for p in positions)
        if len(set(keep)) != len(keep):
            raise ValueError("positions must be unique")
        if not keep:
            return np.ones((1, 1), dtype=complex)
        if keep[0] < 0 or keep[-1] >= self.n_sites:
            raise IndexError("positions out of range")
        work = self._shallow_copy()
        work.move_center(keep[0])
        keep_set = set(keep)

        # env[ket_bond, bra_bond, I, J]
        chi = work.tensors[keep[0]].shape[0]
        env = np.eye(chi, dtype=complex).reshape(chi, chi, 1, 1)
        for pos in range(keep[0], keep[-1] + 1):
            tensor = work.tensors[pos]
            tmp = np.tensordot(env, tensor, axes=([0], [0]))  # a I J s y
            if pos in keep_set:
                tmp = np.tensordot(tmp, tensor.conj(), axes=([0], [0]))  # I J s y t b
                tmp = tmp.transpose(3, 5, 0, 2, 1, 4)  # y b I s J t
                shape = tmp.shape
                env = tmp.reshape(
                    shape[0], shape[1], shape[2] * shape[3], shape[4] * shape[5]
                )
            else:
                tmp = np.tensordot(tmp, tensor.conj(), axes=([0, 3], [0, 1]))  # I J y b
                env = tmp.transpose(2, 3, 0, 1)
        rho = np.einsum("xxIJ->IJ", env)
        return rho / np.trace(rho)


def _truncation_rank(
    s_vals: np.ndarray, chi_max: Optional[int] = None, cutoff: float = 0.0
) -> int:
    """Number of Schmidt values to keep.

    Keeps at most ``chi_max`` values, discarding the tail whose squared weight
    (relative to the total) is at most ``cutoff``. At least one value is kept.
    """
    weights = s_vals**2
    total = weights.sum()
    keep = len(s_vals)
    if cutoff > 0 and total > 0:
        tail = np.cumsum(weights[::-1])[::-1] / total  # tail[i] = weight of i..end
        # tail is decreasing: keep the values whose tail weight exceeds cutoff
        keep = int(np.sum(tail > cutoff))
    if chi_max is not None:
        keep = min(keep, chi_max)
    return max(1, keep)
