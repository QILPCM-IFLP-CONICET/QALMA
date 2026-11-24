r"""Dense kernels for the quantum correlation estimators.

The functions in this module act on plain :class:`numpy.ndarray` density
matrices together with the list of local dimensions that defines their
tensor-product structure. They know nothing about QALMA operators or
systems, so they can be tested against textbook states and reused by other
backends.

Conventions (shared with :mod:`qalma.gaussian`):

* entropies are in bits by default (``base=2``);
* the negativity is :math:`\mathcal{N}=(\|\rho^{T_B}\|_1-1)/2`;
* the logarithmic negativity is :math:`E_N=\log_2\|\rho^{T_B}\|_1`.
"""

from typing import Sequence

import numpy as np

__all__ = [
    "binary_entropy",
    "concurrence_dense",
    "entanglement_of_formation_from_concurrence",
    "log_negativity_dense",
    "mutual_information_dense",
    "negativity_dense",
    "normalize_density_matrix",
    "partial_trace_dense",
    "partial_transpose_dense",
    "trace_norm_hermitian",
    "von_neumann_entropy_dense",
]

_SIGMA_Y_SIGMA_Y = np.array(
    [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=float
)


def _check_dims(rho: np.ndarray, dims: Sequence[int]) -> tuple:
    """Check that ``dims`` is compatible with the shape of ``rho``."""
    dims = tuple(int(d) for d in dims)
    total = int(np.prod(dims)) if dims else 1
    if rho.ndim != 2 or rho.shape != (total, total):
        raise ValueError(
            f"A density matrix of shape {rho.shape} is not compatible with the "
            f"local dimensions {dims}."
        )
    return dims


def normalize_density_matrix(rho: np.ndarray) -> np.ndarray:
    r"""Return the Hermitian part of ``rho`` divided by its trace.

    Parameters
    ----------
    rho : numpy.ndarray
        Square matrix representing a (possibly unnormalized) density
        operator.

    Returns
    -------
    numpy.ndarray
        :math:`(\rho+\rho^\dagger)/(2\,\mathrm{Tr}\rho)`.

    Raises
    ------
    ValueError
        If the matrix is not square or its trace is not positive.
    """
    rho = np.asarray(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"A density matrix must be square, got shape {rho.shape}.")
    rho = 0.5 * (rho + rho.conj().T)
    trace = float(np.real(np.trace(rho)))
    if not trace > 0:
        raise ValueError(f"The density matrix has a non-positive trace ({trace}).")
    return rho / trace


def partial_transpose_dense(
    rho: np.ndarray, dims: Sequence[int], transposed: Sequence[int]
) -> np.ndarray:
    r"""Partial transpose of ``rho`` on the factors listed in ``transposed``.

    Parameters
    ----------
    rho : numpy.ndarray
        Matrix acting on :math:`\bigotimes_k \mathbb{C}^{d_k}`.
    dims : sequence of int
        Local dimensions :math:`d_k`.
    transposed : sequence of int
        Positions (indices into ``dims``) of the factors to transpose.

    Returns
    -------
    numpy.ndarray
        The partially transposed matrix, with the same shape as ``rho``.
    """
    dims = _check_dims(rho, dims)
    n_factors = len(dims)
    tensor = rho.reshape(dims + dims)
    axes = list(range(2 * n_factors))
    for k in set(transposed):
        axes[k], axes[n_factors + k] = axes[n_factors + k], axes[k]
    return tensor.transpose(axes).reshape(rho.shape)


def partial_trace_dense(
    rho: np.ndarray, dims: Sequence[int], keep: Sequence[int]
) -> np.ndarray:
    r"""Trace out every factor not listed in ``keep``.

    The kept factors appear in the result in the order given by ``keep``.

    Parameters
    ----------
    rho : numpy.ndarray
        Matrix acting on :math:`\bigotimes_k \mathbb{C}^{d_k}`.
    dims : sequence of int
        Local dimensions :math:`d_k`.
    keep : sequence of int
        Positions of the factors to keep, in the desired order.

    Returns
    -------
    numpy.ndarray
        The reduced matrix.
    """
    dims = _check_dims(rho, dims)
    keep = [int(k) for k in keep]
    if len(set(keep)) != len(keep):
        raise ValueError(f"Repeated factors in {keep}.")
    n_factors = len(dims)
    traced = [k for k in range(n_factors) if k not in keep]
    tensor = rho.reshape(dims + dims)
    # Bring the factors to the order (keep, traced | keep, traced).
    order = keep + traced
    tensor = tensor.transpose(order + [n_factors + k for k in order])
    d_keep = int(np.prod([dims[k] for k in keep])) if keep else 1
    d_traced = int(np.prod([dims[k] for k in traced])) if traced else 1
    tensor = tensor.reshape(d_keep, d_traced, d_keep, d_traced)
    return np.einsum("ajbj->ab", tensor)


def trace_norm_hermitian(matrix: np.ndarray) -> float:
    r"""Trace norm :math:`\|M\|_1=\sum_k|\lambda_k|` of a Hermitian matrix."""
    return float(np.sum(np.abs(np.linalg.eigvalsh(matrix))))


def negativity_dense(
    rho: np.ndarray, dims: Sequence[int], transposed: Sequence[int]
) -> float:
    r"""Negativity :math:`(\|\rho^{T_B}\|_1-1)/2`.

    Parameters
    ----------
    rho : numpy.ndarray
        Density matrix (it is normalized internally).
    dims : sequence of int
        Local dimensions.
    transposed : sequence of int
        Positions of the factors that form the subsystem :math:`B`.

    Returns
    -------
    float
        The negativity, clipped at zero to absorb rounding errors.
    """
    rho = normalize_density_matrix(rho)
    norm = trace_norm_hermitian(partial_transpose_dense(rho, dims, transposed))
    return max(0.0, 0.5 * (norm - 1.0))


def log_negativity_dense(
    rho: np.ndarray, dims: Sequence[int], transposed: Sequence[int]
) -> float:
    r"""Logarithmic negativity :math:`\log_2\|\rho^{T_B}\|_1`.

    See :func:`negativity_dense` for the parameters. The result is clipped at
    zero to absorb rounding errors.
    """
    rho = normalize_density_matrix(rho)
    norm = trace_norm_hermitian(partial_transpose_dense(rho, dims, transposed))
    return max(0.0, float(np.log2(norm)))


def von_neumann_entropy_dense(rho: np.ndarray, base: float = 2) -> float:
    r"""Von Neumann entropy :math:`-\mathrm{Tr}\,\rho\log\rho`.

    Parameters
    ----------
    rho : numpy.ndarray
        Density matrix (it is normalized internally).
    base : float
        Base of the logarithm. ``2`` (default) gives bits.

    Returns
    -------
    float
        The entropy.
    """
    probs = np.linalg.eigvalsh(normalize_density_matrix(rho))
    probs = probs[probs > 0]
    return max(0.0, float(-np.sum(probs * np.log(probs)) / np.log(base)))


def mutual_information_dense(
    rho: np.ndarray,
    dims: Sequence[int],
    subsystem_a: Sequence[int],
    subsystem_b: Sequence[int],
    base: float = 2,
) -> float:
    """Mutual information :math:`S_A+S_B-S_{AB}` between two sets of factors.

    Parameters
    ----------
    rho : numpy.ndarray
        Density matrix (it is normalized internally).
    dims : sequence of int
        Local dimensions.
    subsystem_a, subsystem_b : sequence of int
        Disjoint positions of the factors of each subsystem. Factors in
        neither are traced out.
    base : float
        Base of the logarithm. ``2`` (default) gives bits.

    Returns
    -------
    float
        The mutual information, clipped at zero.
    """
    if set(subsystem_a) & set(subsystem_b):
        raise ValueError("The two subsystems overlap.")
    rho = normalize_density_matrix(rho)
    rho_ab = partial_trace_dense(rho, dims, list(subsystem_a) + list(subsystem_b))
    dims_ab = [dims[k] for k in subsystem_a] + [dims[k] for k in subsystem_b]
    n_a = len(subsystem_a)
    rho_a = partial_trace_dense(rho_ab, dims_ab, range(n_a))
    rho_b = partial_trace_dense(rho_ab, dims_ab, range(n_a, len(dims_ab)))
    value = (
        von_neumann_entropy_dense(rho_a, base)
        + von_neumann_entropy_dense(rho_b, base)
        - von_neumann_entropy_dense(rho_ab, base)
    )
    return max(0.0, value)


def _drop_rounding_noise(evals: np.ndarray) -> np.ndarray:
    """Set to zero the eigenvalues indistinguishable from rounding errors.

    The concurrence takes square roots of eigenvalues, which turns a rounding
    error of order :math:`10^{-16}` into an error of order :math:`10^{-8}`.
    Eigenvalues below a few machine epsilons of the largest one (and negative
    ones) are therefore replaced by zero before taking the root.
    """
    scale = max(1.0, float(np.max(np.abs(evals)))) if evals.size else 1.0
    threshold = 16 * np.finfo(float).eps * scale
    return np.where(evals > threshold, evals, 0.0)


def concurrence_dense(rho: np.ndarray) -> float:
    r"""Wootters concurrence of a two-qubit density matrix.

    :math:`C=\max(0,\lambda_1-\lambda_2-\lambda_3-\lambda_4)`, with
    :math:`\lambda_k` the decreasingly ordered square roots of the
    eigenvalues of :math:`\sqrt\rho\,\tilde\rho\,\sqrt\rho`, and
    :math:`\tilde\rho=(\sigma_y\otimes\sigma_y)\rho^*(\sigma_y\otimes
    \sigma_y)`. Using the Hermitian matrix
    :math:`\sqrt\rho\,\tilde\rho\,\sqrt\rho` (instead of the
    non-Hermitian :math:`\rho\tilde\rho`) keeps the eigenvalues real.

    Parameters
    ----------
    rho : numpy.ndarray
        :math:`4\times4` density matrix (it is normalized internally).

    Returns
    -------
    float
        The concurrence, in :math:`[0, 1]`.
    """
    rho = normalize_density_matrix(rho)
    if rho.shape != (4, 4):
        raise ValueError(f"The concurrence needs a 4x4 matrix, got {rho.shape}.")
    evals, evecs = np.linalg.eigh(rho)
    sqrt_rho = (evecs * np.sqrt(_drop_rounding_noise(evals))) @ evecs.conj().T
    rho_tilde = _SIGMA_Y_SIGMA_Y @ rho.conj() @ _SIGMA_Y_SIGMA_Y
    r_mat = sqrt_rho @ rho_tilde @ sqrt_rho
    r_mat = 0.5 * (r_mat + r_mat.conj().T)
    lambdas = np.sqrt(_drop_rounding_noise(np.linalg.eigvalsh(r_mat)))[::-1]
    return float(min(1.0, max(0.0, lambdas[0] - np.sum(lambdas[1:]))))


def binary_entropy(prob: float) -> float:
    r"""Binary entropy :math:`h(p)=-p\log_2p-(1-p)\log_2(1-p)`, in bits."""
    prob = float(min(1.0, max(0.0, prob)))
    return float(-sum(q * np.log2(q) for q in (prob, 1.0 - prob) if q > 0))


def entanglement_of_formation_from_concurrence(concurrence: float) -> float:
    r"""Two-qubit entanglement of formation (in ebits) from the concurrence.

    :math:`E_F=h\left(\frac{1+\sqrt{1-C^2}}{2}\right)`, with :math:`h`
    the binary entropy.
    """
    concurrence = float(min(1.0, max(0.0, concurrence)))
    return binary_entropy(0.5 * (1.0 + np.sqrt(1.0 - concurrence**2)))
