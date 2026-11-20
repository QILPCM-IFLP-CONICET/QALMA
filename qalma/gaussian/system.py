r"""Gaussian states of bosonic modes.

A Gaussian state of ``n`` bosonic modes is determined by its first and second
moments. :class:`BosonicGaussianSystem` stores states with vanishing first
moments through the matrix of contractions

.. math::

    \mathcal{G}_{ij} = \langle \Phi_i \Phi_j^\dagger\rangle,
    \qquad
    \Phi = (b_1,\dots,b_n,b_1^\dagger,\dots,b_n^\dagger)^T ,

from which the covariance matrix of any subset of modes, its entropy and the
negativity between two subsets follow. A state can be built from

* a quadratic generator :math:`K = \frac12\Phi^\dagger\mathcal{H}\Phi`, as the
  thermal state :math:`e^{-K}/Z` (:func:`build_boson_system_from_quadratic_form`);
* its matrix of contractions (:func:`build_boson_system_from_correlations`).
"""

from dataclasses import dataclass, field
from typing import Hashable, Iterable, List, Optional, Sequence

import numpy as np

from .bogoliubov import ZERO_MODE_RTOL, bogoliubov_reduced, symplectic_spectrum

__all__ = [
    "BosonicGaussianSystem",
    "build_boson_system_from_correlations",
    "build_boson_system_from_quadratic_form",
]


def _particle_hole(matrix: np.ndarray) -> np.ndarray:
    """Return ``X matrix^* X``, with ``X`` the exchange of ``b`` and ``b^dag``."""
    n_modes = matrix.shape[0] // 2
    perm = np.concatenate([np.arange(n_modes, 2 * n_modes), np.arange(n_modes)])
    return matrix.conj()[np.ix_(perm, perm)]


def _check_square_even(matrix: np.ndarray, name: str) -> None:
    """Raise if ``matrix`` is not a square matrix of even dimension."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 2:
        raise ValueError(f"{name} must be a square matrix of even dimension.")


@dataclass
class BosonicGaussianSystem:
    r"""Gaussian state of ``n`` bosonic modes with vanishing first moments.

    Attributes
    ----------
    contractions : numpy.ndarray
        Matrix :math:`\mathcal{G} = \langle\Phi\Phi^\dagger\rangle`, of shape
        ``(2 n, 2 n)``, with
        :math:`\Phi = (b_1,\dots,b_n,b_1^\dagger,\dots,b_n^\dagger)^T`.
    labels : list
        A label for each mode (for instance, the site the mode belongs to),
        used by :meth:`modes_of`. By default, the index of the mode.
    lam : numpy.ndarray or None
        Signed frequencies of the normal modes of the generator, if the state
        was built from a quadratic form.
    t_mat : numpy.ndarray or None
        Bogoliubov transformation to the normal modes, if the state was built
        from a quadratic form.
    n_zero_modes : int
        Number of zero modes removed from the generator. If it is not zero,
        the contractions describe the sector with fixed zero modes, which in
        general is not a physical state of the original modes (its reduced
        covariance matrices may violate the uncertainty relations), and the
        entropies and negativities are not defined.
    """

    contractions: np.ndarray
    labels: List[Hashable] = field(default_factory=list)
    lam: Optional[np.ndarray] = None
    t_mat: Optional[np.ndarray] = None
    n_zero_modes: int = 0

    def __post_init__(self):
        """Check the shape of the contractions and set the default labels."""
        self.contractions = np.asarray(self.contractions, dtype=complex)
        _check_square_even(self.contractions, "contractions")
        n_modes = self.contractions.shape[0] // 2
        if not self.labels:
            self.labels = list(range(n_modes))
        self.labels = list(self.labels)
        if len(self.labels) != n_modes:
            raise ValueError(
                f"{len(self.labels)} labels were given for {n_modes} modes."
            )

    @property
    def n_modes(self) -> int:
        """Number of bosonic modes."""
        return self.contractions.shape[0] // 2

    def modes_of(self, labels: Iterable[Hashable]) -> List[int]:
        """Return the indices of the modes whose label is in ``labels``."""
        labels = set(labels)
        return [k for k, label in enumerate(self.labels) if label in labels]

    def covariance_matrix(self, modes: Optional[Sequence[int]] = None) -> np.ndarray:
        r"""Return the covariance matrix of the quadratures of a subset of modes.

        The quadratures are ordered as :math:`(x_1,\dots,x_m,p_1,\dots,p_m)`,
        with :math:`x=(b+b^\dagger)/\sqrt2` and :math:`p=(b-b^\dagger)/(i\sqrt2)`,
        so that the vacuum has covariance :math:`1/2`.

        Parameters
        ----------
        modes : sequence of int, optional
            Indices of the modes. By default, all of them.

        Returns
        -------
        numpy.ndarray
            Real symmetric matrix of shape ``(2 m, 2 m)``.
        """
        n_all = self.n_modes
        modes = list(range(n_all)) if modes is None else list(modes)
        n_sub = len(modes)
        idx = np.array(modes + [n_all + k for k in modes], dtype=int)
        g_sub = self.contractions[np.ix_(idx, idx)]
        eye = np.eye(n_sub)
        u_mat = np.block([[eye, eye], [-1j * eye, 1j * eye]]) / np.sqrt(2.0)
        return np.real(u_mat @ g_sub @ u_mat.conj().T)

    def _check_no_zero_modes(self) -> None:
        """Raise if the generator had zero modes."""
        if self.n_zero_modes:
            raise ValueError(
                f"The generator has {self.n_zero_modes} zero mode(s): without "
                "them the contractions are not those of a physical state of the "
                "modes."
            )

    def entropy(self, modes: Optional[Sequence[int]] = None) -> float:
        r"""Return the von Neumann entropy of a subset of modes, in bits.

        :math:`S=\sum_\alpha[(1+f_\alpha)\log_2(1+f_\alpha)-f_\alpha\log_2
        f_\alpha]`, with :math:`f_\alpha=\nu_\alpha-1/2` and :math:`\nu_\alpha`
        the symplectic eigenvalues of the covariance matrix.

        Parameters
        ----------
        modes : sequence of int, optional
            Indices of the modes. By default, all of them.

        Returns
        -------
        float
            The entropy in bits.

        Raises
        ------
        ValueError
            If the generator had zero modes.
        """
        self._check_no_zero_modes()
        nu = symplectic_spectrum(self.covariance_matrix(modes))
        f = np.clip(nu - 0.5, 0.0, None)
        f = f[f > 1e-15]
        return float(np.sum((1 + f) * np.log2(1 + f) - f * np.log2(f)))

    def negativity(
        self, modes_b: Sequence[int], modes_c: Optional[Sequence[int]] = None
    ) -> float:
        r"""Return the negativity between two subsets of modes.

        It is computed from the symplectic spectrum :math:`\tilde\nu` of the
        partial transpose (time reversal of the momenta of ``modes_c``) of the
        reduced state of both subsets:
        :math:`\mathcal{N} = \frac12[\prod_{\tilde\nu<1/2}(2\tilde\nu)^{-1}-1]`.

        Parameters
        ----------
        modes_b, modes_c : sequence of int
            Indices of the modes of the two subsets. If ``modes_c`` is
            ``None``, the complement of ``modes_b`` is used.

        Returns
        -------
        float
            The negativity.

        Raises
        ------
        ValueError
            If the generator had zero modes, or if the subsets overlap.
        """
        self._check_no_zero_modes()
        modes_b = list(modes_b)
        if modes_c is None:
            modes_c = [k for k in range(self.n_modes) if k not in set(modes_b)]
        modes_c = list(modes_c)
        if set(modes_b) & set(modes_c):
            raise ValueError("The two subsets of modes overlap.")
        modes = modes_b + modes_c
        sigma = self.covariance_matrix(modes)
        n_sub = len(modes)
        flip = np.ones(2 * n_sub)
        flip[n_sub + len(modes_b) :] = -1.0  # time reversal of the momenta of C
        nu = symplectic_spectrum(flip[:, None] * sigma * flip[None, :])
        nu = nu[nu < 0.5 - 1e-12]
        return float(0.5 * (np.prod(1.0 / (2.0 * nu)) - 1.0))


def build_boson_system_from_quadratic_form(
    h_mat: np.ndarray,
    labels: Optional[Sequence[Hashable]] = None,
    zero_rtol: float = ZERO_MODE_RTOL,
    static: Optional[np.ndarray] = None,
) -> BosonicGaussianSystem:
    r"""Build the thermal state :math:`e^{-K}/Z` of a quadratic generator.

    The generator is :math:`K = \frac12\Phi^\dagger\mathcal{H}\Phi`, with
    :math:`\mathcal{H} = \begin{pmatrix}A & B\\ B^* & A^*\end{pmatrix}` (the
    inverse temperature is included in :math:`K`). It is diagonalized with a
    Bogoliubov transformation (see :func:`~qalma.gaussian.bogoliubov_reduced`);
    its zero modes, if any, are removed.

    Parameters
    ----------
    h_mat : numpy.ndarray
        Hermitian positive semidefinite matrix :math:`\mathcal{H}`.
    labels : sequence, optional
        A label for each mode. By default, the index of the mode.
    zero_rtol : float, optional
        Relative tolerance to identify the zero modes.
    static : numpy.ndarray, optional
        Boolean array with the modes that have tiny frequencies by
        construction, and are not zero modes (see
        :func:`~qalma.gaussian.bogoliubov_reduced`).

    Returns
    -------
    BosonicGaussianSystem
        The Gaussian state.

    Raises
    ------
    ValueError
        If ``h_mat`` has not the structure of a bosonic quadratic form, or if
        it is not positive semidefinite.
    """
    h_mat = np.asarray(h_mat, dtype=complex)
    _check_square_even(h_mat, "h_mat")
    scale = max(float(np.max(np.abs(h_mat))) if h_mat.size else 0.0, 1e-300)
    if np.max(np.abs(h_mat - h_mat.conj().T), initial=0.0) > 1e-10 * scale:
        raise ValueError("h_mat must be Hermitian.")
    if np.max(np.abs(h_mat - _particle_hole(h_mat)), initial=0.0) > 1e-10 * scale:
        raise ValueError("h_mat must have the block structure [[A, B], [B*, A*]].")
    lam, t_mat, n_zero, _ = bogoliubov_reduced(h_mat, zero_rtol, static)
    omega = np.abs(lam)
    # <Psi Psi^dag>: n + 1 for annihilation-like modes, n for creation-like ones.
    expo = np.where(lam > 0, 0.0, -omega)
    occupation = np.exp(expo) / (-np.expm1(-omega))
    contractions = (t_mat * occupation[None, :]) @ t_mat.conj().T
    return BosonicGaussianSystem(
        contractions=contractions,
        labels=list(labels) if labels is not None else [],
        lam=lam,
        t_mat=t_mat,
        n_zero_modes=n_zero,
    )


def build_boson_system_from_correlations(
    contractions: np.ndarray,
    labels: Optional[Sequence[Hashable]] = None,
    tol: float = 1e-8,
) -> BosonicGaussianSystem:
    r"""Build a Gaussian state from its matrix of contractions.

    Parameters
    ----------
    contractions : numpy.ndarray
        Matrix :math:`\mathcal{G}_{ij} = \langle\Phi_i\Phi_j^\dagger\rangle`
        of shape ``(2 n, 2 n)``. It must satisfy the canonical commutation
        relations, :math:`\mathcal{G} - X\mathcal{G}^TX = \eta`, with
        :math:`X` the exchange of :math:`b` and :math:`b^\dagger` and
        :math:`\eta = \mathrm{diag}(1,\dots,1,-1,\dots,-1)`, and the
        uncertainty relations, :math:`\mathcal{G}\ge 0` and
        :math:`\mathcal{G} - \eta \ge 0`.
    labels : sequence, optional
        A label for each mode. By default, the index of the mode.
    tol : float, optional
        Absolute tolerance of the checks.

    Returns
    -------
    BosonicGaussianSystem
        The Gaussian state.

    Raises
    ------
    ValueError
        If ``contractions`` is not the matrix of contractions of a bosonic
        state.
    """
    contractions = np.asarray(contractions, dtype=complex)
    _check_square_even(contractions, "contractions")
    n_modes = contractions.shape[0] // 2
    eta = np.diag(np.concatenate([np.ones(n_modes), -np.ones(n_modes)]))
    if np.max(np.abs(contractions - contractions.conj().T), initial=0.0) > tol:
        raise ValueError("The contractions must be a Hermitian matrix.")
    commutator = contractions - _particle_hole(contractions).conj().T
    if np.max(np.abs(commutator - eta), initial=0.0) > tol:
        raise ValueError(
            "The contractions do not satisfy the canonical commutation relations."
        )
    hermitian = 0.5 * (contractions + contractions.conj().T)
    for matrix in (hermitian, hermitian - eta):
        if matrix.size and np.linalg.eigvalsh(matrix)[0] < -tol:
            raise ValueError(
                "The contractions violate the uncertainty relations: they are "
                "not those of a physical state."
            )
    return BosonicGaussianSystem(
        contractions=contractions,
        labels=list(labels) if labels is not None else [],
    )
