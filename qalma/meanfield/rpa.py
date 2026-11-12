r"""RPA reduction of a Gibbs state.

Given ``rho = exp(-K)/Z`` and a subsystem ``A``, the generator is split as

.. math::

    K = K_A + K_E + W_{AE}

and the reduced generator is approximated by

.. math::

    K_A' = K_{A,MF} + K_{A,RPA}.

Here :math:`K_{A,MF}` is the mean-field projection of the boundary terms
(see :mod:`qalma.meanfield.gibbs_partial_trace`), and :math:`K_{A,RPA}`
accounts for the Gaussian fluctuations of the environment around its
mean-field state. Let :math:`e_\mu` be a Hilbert-Schmidt orthonormal basis
of local operators and
:math:`\delta e_\mu = e_\mu - \langle e_\mu\rangle_{MF}` the centred basis.
The centred two-body coupling matrix :math:`M` is defined by
:math:`K_2 = \frac12\sum_{\mu\nu} M_{\mu\nu}\,\delta e_\mu\delta e_\nu`.

**Static approximation.**

.. math::

    K_{A,RPA} = -\frac12 \sum_{\mu\nu} \Sigma_{\mu\nu}\,
                \delta e_\mu \delta e_\nu,
    \qquad
    \Sigma = M_{AE}\,\chi_E\,M_{EA},
    \qquad
    \chi_E = (1 + \chi^0 M_{EE})^{-1}\chi^0 ,

where :math:`\chi^0` is the connected Kubo-Mori susceptibility of the
single-site mean-field states.

**Dynamical correction (RPA bosonization).** The static form replaces the
imaginary-time evolved operators :math:`\delta e_\mu(s) =
e^{sK_{A,MF}}\delta e_\mu e^{-sK_{A,MF}}` by their value at :math:`s=0`.
To keep the full imaginary-time kernel
:math:`\Sigma_{\mu\nu}(\tau) = \langle B_\mu(\tau) B_\nu\rangle`, with
:math:`B_\mu = \sum_{\nu\in E} M_{\mu\nu}\delta e_\nu`, the local
algebra of every site of the environment is bosonized in the eigenbasis of
its mean-field state :math:`\sigma = \sum_i p_i |i\rangle\langle i|`
(populations sorted as :math:`p_i \ge p_j` for :math:`i<j`):

.. math::

    |i\rangle\langle j| \to \sqrt{p_i - p_j}\; a_{ij},
    \qquad
    |j\rangle\langle i| \to \sqrt{p_i - p_j}\; a^\dagger_{ij},

so that :math:`[|i\rangle\langle j|, |j\rangle\langle i|]` has expectation
value :math:`p_i-p_j`, as required by :math:`[a,a^\dagger]=1`. Keeping only
the part of :math:`\delta e_\mu` that is linear in the bosons,

.. math::

    \delta e_\mu \simeq \sum_k \left(g_{\mu k} a_k + g^*_{\mu k}
    a_k^\dagger\right),
    \qquad
    g_{\mu,(ij)} = \sqrt{p_i-p_j}\,(e_\mu)_{ij},

the environment becomes a quadratic bosonic Hamiltonian
:math:`H_b = \sum_k \Delta_k a_k^\dagger a_k + \frac12\sum M_{\mu\nu}
\delta e_\mu\delta e_\nu` with :math:`\Delta_k = \ln(p_i/p_j)`, which is
diagonalized by a Bogoliubov transformation. This is equivalent to
integrating the Hubbard-Stratonovich fields analytically and summing over
all Matsubara frequencies: with normal modes :math:`\Psi_m` of signed
frequency :math:`\lambda_m` and :math:`B_\mu = \sum_m c_{\mu m}\Psi_m`,

.. math::

    \Sigma_{\mu\nu}(\tau) = \sum_m c_{\mu m} c^*_{\nu m}\,
    \frac{\eta_m\, e^{-\lambda_m \tau}}{1 - e^{-\lambda_m}},
    \qquad \eta_m = \operatorname{sign}\lambda_m .

The diagonal operators :math:`|i\rangle\langle i|` are bilinear in the
bosons, so they are not part of the linear truncation. Their static
(intraband) response, :math:`\sum_i p_i (e_\mu)_{ii}(e_\nu)_{ii} -
\langle e_\mu\rangle\langle e_\nu\rangle`, and the one of pairs with
:math:`p_i = p_j`, are time independent. They are included as bosons of
vanishing frequency ``static_gap`` whose couplings are scaled so that
:math:`\chi(\omega=0)` is exact.

With :math:`\Sigma(\tau)` at hand, the second-order contribution to the
reduced state is

.. math::

    Y = \int_{0<s'<s<1} ds\,ds'\; \Sigma_{\mu\nu}(s-s')\,
        e^{-K_{A,MF}}\,\delta e_\mu(s)\,\delta e_\nu(s') ,

and the generator correction follows from inverting the derivative of the
matrix exponential (Kubo-Mori) at first order in :math:`Y`:
:math:`\delta K_{mn} = -Y_{mn}/w_{mn}`, with :math:`w_{mn} =
(e^{-\epsilon_n}-e^{-\epsilon_m})/(\epsilon_m-\epsilon_n)` in the
eigenbasis of :math:`K_{A,MF}`. In the limit where :math:`\delta e_\mu`
commutes with :math:`K_{A,MF}` the static expression is recovered.

Limitations
-----------
The environment is described by the Gaussian (linear) truncation of the
bosonization, completed with the static modes. The quartic terms, which
come from the products of the diagonal operators of different sites
(hopping and pairing contributions in a Wick decoupling), are neglected.
Their inclusion matters for sensitive quantities such as entanglement, but
it is only controlled at low temperatures with not too strong couplings, or
at high temperatures. At intermediate temperatures the thermal occupations
of the bosonic modes become large and the bosons explore regions of the
Fock space that have no counterpart in the local Hilbert space, so the
result should be regarded as qualitative. If the matrix of the bosonic
Hamiltonian is not positive definite, the RPA is unstable and a
``ValueError`` is raised.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from qutip import Qobj

from qalma.meanfield.gibbs_partial_trace import gibbs_meanfield_partial_trace
from qalma.meanfield.variational import variational_quadratic_mfa
from qalma.model import SystemDescriptor
from qalma.operators import Operator, QutipOperator
from qalma.operators.basic import LocalOperator
from qalma.operators.quadratic import QuadraticFormOperator
from qalma.operators.quadratic.build import (
    build_local_basis,
    build_quadratic_form_matrix,
    classify_terms,
    zero_expectation_value_basis,
)
from qalma.operators.states.gibbs import GibbsDensityOperator

__all__ = [
    "RPAGibbsDensityOperator",
    "RPAStructure",
    "build_rpa_structure",
    "kubo_mori_susceptibility",
    "local_boson_modes",
    "rpa_coupling_kernel",
    "rpa_dynamic_kernel",
    "rpa_generator_correction",
]

_DEGENERACY_TOL = 1e-12
_STATIC_GAP = 1e-6


# ### Local (single-site) quantities ###


def _spectral_data(sigma_loc: np.ndarray, basis: List[np.ndarray]):
    """Return populations, ``-log`` populations and the basis in the eigenbasis."""
    p, vecs = np.linalg.eigh(sigma_loc)
    p = np.clip(p, 1e-300, None)
    ops = np.array([vecs.conj().T @ np.asarray(e) @ vecs for e in basis])
    return p, -np.log(p), ops


def kubo_mori_susceptibility(sigma_loc: np.ndarray, basis: List[np.ndarray]):
    r"""Connected Kubo-Mori susceptibility of a local state.

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

    Returns
    -------
    numpy.ndarray
        Real symmetric matrix :math:`\chi^0_{\mu\nu}`.
    """
    p, eps, ops = _spectral_data(sigma_loc, basis)
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


def local_boson_modes(
    sigma_loc: np.ndarray,
    basis: List[np.ndarray],
    static_gap: float = _STATIC_GAP,
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

    Returns
    -------
    freqs : numpy.ndarray
        Mode frequencies :math:`\Delta_k \ge` ``static_gap``, shape ``(K,)``.
    couplings : numpy.ndarray
        Complex couplings :math:`g_{\mu k}`, shape ``(len(basis), K)``.
    """
    p, vecs = np.linalg.eigh(sigma_loc)
    order = np.argsort(-p)
    p = np.clip(p[order], 1e-300, None)
    vecs = vecs[:, order]
    ops = np.array([vecs.conj().T @ np.asarray(e) @ vecs for e in basis])
    freqs: List[float] = []
    couplings: List[np.ndarray] = []
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            gap = np.log(p[i] / p[j])
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


# ### Coupling matrices ###


@dataclass
class RPAStructure:
    r"""Data of the RPA of a Gibbs state that do not depend on the subsystem.

    The centred coupling matrix, the local bases and the single-site
    quantities (static susceptibilities and bosonic modes) are the same for
    any subsystem ``A``. Storing them allows reusing them when many partial
    traces of the same state are required. The single-site quantities are
    computed on demand and cached.

    Attributes
    ----------
    generator : Operator
        Generator ``K`` of the Gibbs state (flat and simplified).
    sigma_mf : ProductDensityOperator
        Mean-field reference state.
    static_gap : float
        Frequency assigned to the static bosons.
    m_full : numpy.ndarray
        Centred two-body coupling matrix :math:`M` of the whole system.
    local_basis : dict
        Centred Hilbert-Schmidt orthonormal basis of local operators of each
        site.
    positions : dict
        Index of the first element of each site in ``m_full``.
    """

    generator: Operator
    sigma_mf: Any
    static_gap: float
    m_full: np.ndarray
    local_basis: Dict[str, List[Qobj]]
    positions: Dict[str, int]
    _local_data: Dict[str, Tuple[np.ndarray, List[np.ndarray]]] = field(
        default_factory=dict, repr=False
    )
    _chi0: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _modes: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False
    )

    @property
    def system(self) -> SystemDescriptor:
        """System descriptor of the generator."""
        return self.generator.system

    def local_data(self, site: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Return the dense mean-field state and local basis of ``site``."""
        if site not in self._local_data:
            sigma_loc = self.sigma_mf.site_factors_qutip[site].full()
            basis = [e.full() for e in self.local_basis[site]]
            self._local_data[site] = (sigma_loc, basis)
        return self._local_data[site]

    def static_chi0(self, site: str) -> np.ndarray:
        """Return the static susceptibility :math:`\\chi^0` of ``site``."""
        if site not in self._chi0:
            self._chi0[site] = kubo_mori_susceptibility(*self.local_data(site))
        return self._chi0[site]

    def boson_modes(self, site: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the bosonic modes of ``site`` (see :func:`local_boson_modes`)."""
        if site not in self._modes:
            self._modes[site] = local_boson_modes(
                *self.local_data(site), self.static_gap
            )
        return self._modes[site]


def build_rpa_structure(
    generator: Operator, sigma_mf, static_gap: float = _STATIC_GAP
) -> RPAStructure:
    """Build the subsystem-independent data of the RPA of a Gibbs state.

    Parameters
    ----------
    generator : Operator
        Generator ``K`` of the Gibbs state.
    sigma_mf : ProductDensityOperator
        Mean-field reference state.
    static_gap : float, optional
        Frequency assigned to the static bosons.

    Returns
    -------
    RPAStructure
        The structure.
    """
    generator = generator.flat().simplify()
    blocks, _, _ = classify_terms(generator, sigma_mf)
    local_basis = build_local_basis(blocks)
    m_full, positions = build_quadratic_form_matrix(blocks, local_basis)
    local_basis = zero_expectation_value_basis(local_basis, sigma_mf)
    return RPAStructure(
        generator=generator,
        sigma_mf=sigma_mf,
        static_gap=static_gap,
        m_full=m_full,
        local_basis=local_basis,
        positions=positions,
    )


def _split_sites(
    structure: RPAStructure, sites: frozenset
) -> Tuple[List[int], List[int], List[str], List[str]]:
    """Split the indices of the coupling matrix into subsystem and environment."""
    idx_a: List[int] = []
    idx_e: List[int] = []
    sites_a: List[str] = []
    sites_e: List[str] = []
    for site, start in structure.positions.items():
        indices = range(start, start + len(structure.local_basis[site]))
        if site in sites:
            idx_a.extend(indices)
            sites_a.append(site)
        else:
            idx_e.extend(indices)
            sites_e.append(site)
    return idx_a, idx_e, sites_a, sites_e


def _check_stability(chi0_static: np.ndarray, m_ee: np.ndarray) -> None:
    """Raise if ``1 + chi0 M_EE`` is not positive definite."""
    stab = np.eye(chi0_static.shape[0]) + chi0_static @ m_ee
    if np.min(np.linalg.eigvals(stab).real) <= 1e-10:
        raise ValueError(
            "Unstable RPA: 1 + chi0 M_EE is not positive definite "
            "(the environment is at or beyond its critical point)."
        )


def rpa_coupling_kernel(
    structure: RPAStructure, sites: frozenset
) -> Tuple[np.ndarray, Dict[str, List], List[str]]:
    r"""Compute the static kernel :math:`\Sigma` restricted to ``sites``.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites : frozenset
        Sites of the subsystem ``A``.

    Returns
    -------
    sigma : numpy.ndarray
        Real symmetric matrix indexed by the local basis elements of the
        sites in ``A``.
    local_basis : dict
        Centred local basis of each site.
    sites_a : list of str
        Sites of ``A`` in the order used to index ``sigma``.
    """
    local_basis = structure.local_basis
    idx_a, idx_e, sites_a, sites_e = _split_sites(structure, sites)
    if not idx_a or not idx_e:
        return np.zeros((len(idx_a), len(idx_a))), local_basis, sites_a
    chi0 = np.zeros((len(idx_e), len(idx_e)))
    pos = 0
    for site in sites_e:
        size = len(local_basis[site])
        chi0[pos : pos + size, pos : pos + size] = structure.static_chi0(site)
        pos += size
    m_ae = structure.m_full[np.ix_(idx_a, idx_e)]
    m_ee = structure.m_full[np.ix_(idx_e, idx_e)]
    _check_stability(chi0, m_ee)
    chi_e = np.linalg.solve(np.eye(len(idx_e)) + chi0 @ m_ee, chi0)
    chi_e = 0.5 * (chi_e + chi_e.T)
    sigma = m_ae @ chi_e @ m_ae.T
    return 0.5 * (sigma + sigma.T), local_basis, sites_a


def _bogoliubov(h_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        If ``h_mat`` is not positive definite (unstable RPA).
    """
    n_half = h_mat.shape[0] // 2
    try:
        chol = np.linalg.cholesky(h_mat)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "Unstable RPA: the bosonic Hamiltonian of the environment is not "
            "positive definite (the environment is at or beyond its "
            "critical point)."
        ) from err
    k_mat = chol.conj().T
    eta = np.diag(np.concatenate([np.ones(n_half), -np.ones(n_half)]))
    lam, vecs = np.linalg.eigh(k_mat @ eta @ k_mat.conj().T)
    t_mat = np.linalg.solve(k_mat, vecs) * np.sqrt(np.abs(lam))[None, :]
    return lam, t_mat


def rpa_dynamic_kernel(
    structure: RPAStructure,
    sites: frozenset,
    n_tau: int = 32,
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
    nodes, weights = np.polynomial.legendre.leggauss(n_tau)
    nodes, weights = 0.5 * (nodes + 1.0), 0.5 * weights
    local_basis = structure.local_basis
    m_full = structure.m_full
    idx_a, idx_e, sites_a, sites_e = _split_sites(structure, sites)
    n_a = len(idx_a)
    empty = np.zeros((n_tau, n_a, n_a), dtype=complex)
    if not idx_a or not idx_e:
        return empty, nodes, weights, local_basis, sites_a

    # Bosonize every site of the environment.
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
        return empty, nodes, weights, local_basis, sites_a
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
    lam, t_mat = _bogoliubov(h_mat)

    # B_mu = sum_m c_{mu m} Psi_m, Sigma(tau) = sum_m c c^* f_m exp(-lam_m tau)
    g_b = m_ae @ g_mat
    c_mat = np.hstack([g_b, g_b.conj()]) @ t_mat
    omega = np.abs(lam)
    expo = np.where(
        lam[None, :] > 0,
        -omega[None, :] * nodes[:, None],
        -omega[None, :] * (1.0 - nodes[:, None]),
    )
    occupation = np.exp(expo) / (-np.expm1(-omega))[None, :]
    sigma_tau = np.einsum("im,tm,jm->tij", c_mat, occupation, c_mat.conj())
    return sigma_tau, nodes, weights, local_basis, sites_a


# ### Dynamical correction of the generator ###


def dynamic_generator_correction(
    k0: np.ndarray,
    d_ops: List[np.ndarray],
    sigma_tau: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    r"""First-order generator correction from an imaginary-time kernel.

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
    delta = eps[:, None] - eps[None, :]  # eps_m - eps_n
    small = np.abs(delta) < _DEGENERACY_TOL
    safe = np.where(small, 1.0, delta)

    def km_weight(a: float) -> np.ndarray:
        """Matrix ``(e^{-a eps_n} - e^{-a eps_m})/(eps_m - eps_n)``."""
        diff = (np.exp(-a * eps)[None, :] - np.exp(-a * eps)[:, None]) / safe
        return np.where(small, a * np.exp(-a * eps)[:, None], diff)

    y_mat = np.zeros_like(k0, dtype=complex)
    for u, w_q, sigma_u in zip(nodes, weights, sigma_tau):
        c_ops = np.einsum("mn,mik->nik", sigma_u, q_ops)
        z_mat = np.einsum("nik,k,nkj->ij", c_ops, np.exp(-u * eps), q_ops)
        y_mat += w_q * z_mat * km_weight(1.0 - u)
    d_k = -y_mat / km_weight(1.0)
    d_k = 0.5 * (d_k + d_k.conj().T)
    return vecs @ d_k @ vecs.conj().T


def rpa_generator_correction(
    generator: Operator,
    k_mf: Operator,
    sites: frozenset,
    sigma_mf,
    dynamic: bool = False,
    n_tau: int = 32,
    static_gap: float = _STATIC_GAP,
    structure: Optional[RPAStructure] = None,
) -> Optional[Operator]:
    r"""RPA correction :math:`K_{A,RPA}` to the mean-field reduced generator.

    The reduced generator is :math:`K_A' = K_{A,MF} + K_{A,RPA}`.

    Parameters
    ----------
    generator : Operator
        Generator ``K`` of the whole Gibbs state.
    k_mf : Operator
        Mean-field reduced generator :math:`K_{A,MF}` of the subsystem
        (the terms of ``K`` inside ``sites`` plus the projected boundary
        terms). It is only used by the dynamical correction.
    sites : frozenset
        Sites of the subsystem ``A``.
    sigma_mf : ProductDensityOperator
        Mean-field reference state.
    dynamic : bool, optional
        If ``True``, use the full imaginary-time kernel (dynamical
        correction, computed with the bosonized RPA) instead of the static
        approximation. This builds dense matrices of the size of the Hilbert
        space of ``A``. The default is False.
    n_tau : int, optional
        Number of imaginary-time quadrature nodes (dynamic only).
    static_gap : float, optional
        Frequency of the static bosons. Ignored if ``structure`` is given.
    structure : RPAStructure, optional
        Precomputed subsystem-independent data. If ``None``, it is built
        from ``generator`` and ``sigma_mf``.

    Returns
    -------
    Operator or None
        The correction :math:`K_{A,RPA}`, or ``None`` if it vanishes
        (for instance, if ``A`` is not coupled to the environment).

    Raises
    ------
    ValueError
        If the RPA is unstable.
    """
    if structure is None:
        structure = build_rpa_structure(generator, sigma_mf, static_gap)
    system = structure.system
    if dynamic:
        return _dynamic_correction(k_mf, structure, sites, n_tau)

    sigma, local_basis, sites_a = rpa_coupling_kernel(structure, sites)
    if sigma.size == 0 or not np.any(np.abs(sigma) > 1e-14):
        return None
    d_ops = [
        LocalOperator(site, e, system) for site in sites_a for e in local_basis[site]
    ]
    lam, vec = np.linalg.eigh(sigma)
    basis, weights = [], []
    for l_a, v_a in zip(lam, vec.T):
        if abs(l_a) < 1e-14:
            continue
        q_a = sum(c * op for c, op in zip(v_a, d_ops) if abs(c) > 1e-14)
        basis.append(q_a.simplify())
        weights.append(-0.5 * l_a)
    return QuadraticFormOperator(tuple(basis), tuple(weights), system)


def _dynamic_correction(
    k_mf: Operator,
    structure: RPAStructure,
    sites: frozenset,
    n_tau: int,
) -> Optional[Operator]:
    """Dynamical RPA correction to the mean-field generator."""
    system = structure.system
    sigma_tau, nodes, weights, local_basis, sites_a = rpa_dynamic_kernel(
        structure, sites, n_tau
    )
    if sigma_tau.size == 0 or not np.any(np.abs(sigma_tau) > 1e-14):
        return None
    block = tuple(sorted(sites))
    k0_qobj: Qobj = k_mf.to_qutip(block)
    d_ops = [
        LocalOperator(site, e, system).to_qutip(block).full()
        for site in sites_a
        for e in local_basis[site]
    ]
    d_k = dynamic_generator_correction(k0_qobj.full(), d_ops, sigma_tau, nodes, weights)
    return QutipOperator(
        Qobj(d_k, dims=k0_qobj.dims),
        system,
        {site: pos for pos, site in enumerate(block)},
    )


# ### The RPA Gibbs state ###


class RPAGibbsDensityOperator(GibbsDensityOperator):
    r"""Gibbs state whose reductions include the RPA correction.

    It behaves as a :class:`~qalma.operators.states.gibbs.GibbsDensityOperator`,
    but :meth:`partial_trace` and :meth:`reduce` return by default the
    mean-field plus RPA approximation of the reduced state
    (:math:`K_A' = K_{A,MF} + K_{A,RPA}`, see :mod:`qalma.meanfield.rpa`).

    The data used by the RPA that do not depend on the subsystem (the
    mean-field state, the centred coupling matrix, the local bases and the
    single-site susceptibilities and bosonic modes) are computed once, on
    demand, and stored in :attr:`rpa_structure`, so they are reused by
    successive partial traces.

    Multiplying or dividing the state by a positive scalar returns an
    ``RPAGibbsDensityOperator`` that shares the stored structure. Other
    arithmetic operations give the standard operator types.

    Parameters
    ----------
    k : Operator
        The generator of the state.
    system : SystemDescriptor, optional
        The system on which the state acts.
    prefactor : float, optional
        Normalization prefactor. The default is 1.
    normalized : bool, optional
        Whether ``k`` is already normalized.
    meanfield : ProductDensityOperator, optional
        Mean-field state. If ``None``, it is computed on demand.
    symmetry_projections : tuple, optional
        Symmetry projections applied to the reduced states.
    dynamic : bool, optional
        Use by default the dynamical RPA correction instead of the static
        one. The default is False.
    n_tau : int, optional
        Number of imaginary-time quadrature nodes of the dynamical
        correction. The default is 32.
    static_gap : float, optional
        Frequency of the static bosons. The default is 1e-6.
    """

    def __init__(
        self,
        k: Operator,
        system: Optional[SystemDescriptor] = None,
        prefactor: float = 1.0,
        normalized: bool = False,
        meanfield=None,
        symmetry_projections: tuple = (),
        dynamic: bool = False,
        n_tau: int = 32,
        static_gap: float = _STATIC_GAP,
    ):
        super().__init__(
            k,
            system=system,
            prefactor=prefactor,
            normalized=normalized,
            meanfield=meanfield,
            symmetry_projections=symmetry_projections,
        )
        self.dynamic = dynamic
        self.n_tau = n_tau
        self.static_gap = static_gap
        self._rpa_structure: Optional[RPAStructure] = None

    @property
    def meanfield(self):
        """The self-consistent mean-field state (computed on demand)."""
        if self._meanfield is None:
            self._meanfield = variational_quadratic_mfa(self.k.flat())
        return self._meanfield

    @property
    def rpa_structure(self) -> RPAStructure:
        """The subsystem-independent data of the RPA (computed on demand)."""
        if self._rpa_structure is None:
            self._rpa_structure = build_rpa_structure(
                self.k, self.meanfield, self.static_gap
            )
        return self._rpa_structure

    def partial_trace(
        self,
        sites: frozenset | SystemDescriptor,
        rpa: bool = True,
        dynamic: Optional[bool] = None,
        exact: bool = True,
    ):
        """Compute the partial trace over the complement of ``sites``.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*. All other sites are traced out.
        rpa : bool, optional
            If ``True`` (default), add the RPA correction to the mean-field
            generator of the reduced state.
        dynamic : bool, optional
            Use the dynamical RPA correction. If ``None`` (default), use
            the value given when the state was built.
        exact : bool, optional
            If ``True`` (default), small systems are reduced exactly, as in
            :class:`~qalma.operators.states.gibbs.GibbsDensityOperator`.
            If ``False``, the mean-field plus RPA reduction is always used.

        Returns
        -------
        Operator
            The reduced density operator.
        """
        if isinstance(sites, SystemDescriptor):
            sites = frozenset(sites.sites)
        return gibbs_meanfield_partial_trace(
            self,
            sites,
            rpa=rpa,
            dynamic=self.dynamic if dynamic is None else dynamic,
            n_tau=self.n_tau,
            exact=exact,
        )

    def reduce(self, sites, state=None, **kwargs):
        """Alias of :meth:`partial_trace`. ``state`` is ignored."""
        return self.partial_trace(sites, **kwargs)

    def _with_prefactor(self, prefactor: complex) -> "RPAGibbsDensityOperator":
        """Copy of the state with a different prefactor, sharing the structure."""
        new_state = type(self)(
            self.k,
            system=self.system,
            prefactor=prefactor,
            normalized=self.normalized,
            meanfield=self._meanfield,
            symmetry_projections=self.symmetry_projections,
            dynamic=self.dynamic,
            n_tau=self.n_tau,
            static_gap=self.static_gap,
        )
        new_state._rpa_structure = self._rpa_structure
        return new_state

    def __truediv__(self, operand):
        if isinstance(operand, (int, float)) and operand > 0:
            return self._with_prefactor(self.prefactor / operand)
        return super().__truediv__(operand)


def _inherit_gibbs_arithmetic(subclass: type) -> None:
    """Register for ``subclass`` the arithmetic handlers of the Gibbs state.

    The dispatch of the arithmetic of the operators is done on the exact
    types of the operands, so a subclass must be registered explicitly.
    Products by positive real scalars are overridden to preserve the type.
    """
    # pylint: disable=import-outside-toplevel
    from qalma.operators.states.register_ops import _gibbs  # noqa: F401

    base = GibbsDensityOperator
    for table in (Operator.__add__dispatch__, Operator.__mul__dispatch__):
        for (left, right), func in list(table.items()):
            if left is base or right is base:
                new_key = (
                    subclass if left is base else left,
                    subclass if right is base else right,
                )
                table.setdefault(new_key, func)
                if left is base and right is base:
                    table.setdefault((subclass, base), func)
                    table.setdefault((base, subclass), func)

    mul_table = Operator.__mul__dispatch__

    def _scaled(fallback_key: Tuple[type, type]) -> Callable:
        fallback = mul_table.get(fallback_key)

        def handler(x_op, y_op):
            state, factor = (x_op, y_op) if isinstance(x_op, subclass) else (y_op, x_op)
            if isinstance(factor, (int, float)) and factor > 0:
                return state._with_prefactor(  # pylint: disable=protected-access
                    state.prefactor * factor
                )
            return fallback(x_op, y_op)

        return handler

    for key in [k for k in mul_table if k[0] is base or k[1] is base]:
        other = key[1] if key[0] is base else key[0]
        if other in (int, float, np.float64):
            new_key = (subclass, other) if key[0] is base else (other, subclass)
            mul_table[new_key] = _scaled(key)


_inherit_gibbs_arithmetic(RPAGibbsDensityOperator)
