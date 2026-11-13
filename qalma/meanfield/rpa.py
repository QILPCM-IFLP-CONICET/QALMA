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

**Hybrid reduction (subsystem exact, RPA on the environment).** Instead of
correcting the generator at first order, the environment is bosonized with
the subsystem removed, its normal modes are compressed to a few effective
bosons (block Lanczos, a Gauss quadrature of the spectral density), the
modes with large thermal occupation are taken as static classical Gaussian
fields, and the resulting hybrid Hamiltonian (subsystem exact, truncated
bosons) is diagonalized exactly (:func:`rpa_hybrid_state`). It is non
perturbative in the coupling between the subsystem and the environment, so
it includes the back-reaction of the subsystem, and it is valid at low
temperatures, where the generator corrections fail. The symmetry
restoration is the one of the Gibbs state (``symmetry_projections``), which
for the parity is the mixture of the two broken-symmetry branches.

**Full bosonization.** If the subsystem is also bosonized, the whole system
is a quadratic bosonic Hamiltonian whose thermal state is Gaussian. The
reduced covariance matrix of any subset of sites gives the entropy and the
negativity of the bosonic RPA state (see :func:`rpa_entanglement_entropy`
and :func:`rpa_negativity`), as in the RPA of Matera, Rossignoli and Canosa
(Phys. Rev. A 82, 052332). The populations of the local mean-field states
are obtained from the local mean-field generators, so that the zero
temperature limit can be reached.

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
import scipy.linalg
import scipy.sparse as sparse
from qutip import Qobj

from qalma.meanfield.gibbs_partial_trace import (
    gibbs_meanfield_partial_trace,
    project_boundary_term,
)
from qalma.meanfield.variational import variational_quadratic_mfa
from qalma.model import SystemDescriptor
from qalma.operators import Operator, QutipOperator
from qalma.operators.arithmetic import SumOperator, iterable_to_operator
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
    "BosonSystem",
    "RPAGibbsDensityOperator",
    "RPAStructure",
    "build_boson_system",
    "build_rpa_structure",
    "effective_bath_modes",
    "environment_modes",
    "rpa_hybrid_state",
    "rpa_entanglement_entropy",
    "rpa_negativity",
    "kubo_mori_susceptibility",
    "local_boson_modes",
    "rpa_coupling_kernel",
    "rpa_dynamic_kernel",
    "rpa_generator_correction",
]

_DEGENERACY_TOL = 1e-12
_STATIC_GAP = 1e-6


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
        p, vecs = np.linalg.eigh(sigma_loc)
    else:
        p, vecs = spectrum
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
    _boson_system: Optional["BosonSystem"] = field(default=None, repr=False)
    _spectra: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = field(
        default=None, repr=False
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

    def local_spectrum(self, site: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the populations and eigenvectors of the local mean-field state.

        They are obtained from the local mean-field generator of the site
        (the terms of ``K`` over the site plus the boundary terms projected
        with the mean-field state), so very small populations keep their
        accuracy even if they cannot be resolved in a density matrix.
        """
        if self._spectra is None:
            self._spectra = _local_spectra(self.generator, self.sigma_mf)
        return self._spectra[site]

    def static_chi0(self, site: str) -> np.ndarray:
        """Return the static susceptibility :math:`\\chi^0` of ``site``."""
        if site not in self._chi0:
            self._chi0[site] = kubo_mori_susceptibility(
                *self.local_data(site), spectrum=self.local_spectrum(site)
            )
        return self._chi0[site]

    def boson_modes(self, site: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the bosonic modes of ``site`` (see :func:`local_boson_modes`)."""
        if site not in self._modes:
            self._modes[site] = local_boson_modes(
                *self.local_data(site),
                self.static_gap,
                spectrum=self.local_spectrum(site),
            )
        return self._modes[site]

    def boson_system(self) -> "BosonSystem":
        """Return the bosonic system of the whole state (computed on demand)."""
        if self._boson_system is None:
            self._boson_system = build_boson_system(self)
        return self._boson_system


def _local_spectra(
    generator: Operator, sigma_mf
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Populations and eigenvectors of the local mean-field generators."""
    system = generator.system
    terms = generator.terms if isinstance(generator, SumOperator) else [generator]
    contributions: Dict[str, List[Operator]] = {}
    for term in terms:
        acts_over = term.acts_over()
        for site in acts_over:
            if len(acts_over) == 1:
                projected = term
            else:
                projected = project_boundary_term(term, sigma_mf, frozenset([site]))
            if projected:
                contributions.setdefault(site, []).append(projected)
    spectra = {}
    for site, local_terms in contributions.items():
        k_loc = iterable_to_operator(local_terms, system, isherm=True)
        k_mat = k_loc.to_qutip((site,)).full()
        eps, vecs = np.linalg.eigh(0.5 * (k_mat + k_mat.conj().T))
        weights = np.exp(-(eps - eps.min()))
        spectra[site] = (weights / weights.sum(), vecs)
    return spectra


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
    lam, t_mat = _bogoliubov(h_mat)
    g_b = m_ae @ g_mat
    c_mat = np.hstack([g_b, g_b.conj()]) @ t_mat
    return _EnvironmentBogoliubov(sites_a, local_basis, c_mat, lam)


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
    environment = _environment_bogoliubov(structure, sites)
    if environment is None:
        _, _, sites_a, _ = _split_sites(structure, sites)
        n_a = sum(len(structure.local_basis[site]) for site in sites_a)
        empty = np.zeros((n_tau, n_a, n_a), dtype=complex)
        return empty, nodes, weights, structure.local_basis, sites_a

    # B_mu = sum_m c_{mu m} Psi_m, Sigma(tau) = sum_m c c^* f_m exp(-lam_m tau)
    c_mat, lam = environment.c_mat, environment.lam
    omega = np.abs(lam)
    expo = np.where(
        lam[None, :] > 0,
        -omega[None, :] * nodes[:, None],
        -omega[None, :] * (1.0 - nodes[:, None]),
    )
    occupation = np.exp(expo) / (-np.expm1(-omega))[None, :]
    sigma_tau = np.einsum("im,tm,jm->tij", c_mat, occupation, c_mat.conj())
    return sigma_tau, nodes, weights, environment.local_basis, environment.sites_a


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


# ### Hybrid reduction: the subsystem exact, RPA on the environment ###


def environment_modes(
    structure: RPAStructure, sites: frozenset
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[Qobj]]]]:
    r"""Normal modes of the bosonized environment, coupled to ``sites``.

    With the subsystem removed, the bosonized environment is a set of
    independent normal modes of frequencies :math:`\omega_m>0` (in the
    units of the generator), and the boundary operators of the subsystem are
    :math:`B_\mu = \sum_m (C_{\mu m} b_m + C^*_{\mu m} b_m^\dagger)`.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites : frozenset
        Sites of the subsystem ``A``.

    Returns
    -------
    omega : numpy.ndarray
        Frequencies :math:`\omega_m` of the normal modes.
    coupling : numpy.ndarray
        Complex matrix :math:`C_{\mu m}` of shape ``(nA, n_modes)``.
    sites_a : list of str
        Sites of ``A`` in the order used to index ``mu``.
    local_basis : dict
        Centred local basis of each site.

    ``None`` is returned if ``A`` is not coupled to the environment.

    Raises
    ------
    ValueError
        If the RPA is unstable.
    """
    environment = _environment_bogoliubov(structure, sites)
    if environment is None:
        return None
    positive = environment.lam > 0
    return (
        environment.lam[positive],
        environment.c_mat[:, positive],
        environment.sites_a,
        environment.local_basis,
    )


def effective_bath_modes(
    omega: np.ndarray,
    coupling: np.ndarray,
    n_blocks: int = 1,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compress a bosonic bath to a few effective modes (block Lanczos).

    The bath is :math:`\sum_m \omega_m b_m^\dagger b_m` and it couples to the
    subsystem through :math:`B_\mu = \sum_m (C_{\mu m} b_m + h.c.)`. Only the
    Krylov subspace generated by the coupling vectors is relevant: the
    effective modes are the eigenmodes of the Hamiltonian projected on that
    subspace. This is a Gauss quadrature of the spectral density
    :math:`J_{\mu\nu}(\omega)=\sum_m C_{\mu m}C^*_{\nu m}\delta(\omega-\omega_m)`
    that reproduces its first :math:`2 n_{blocks}` moments, and is exact if the
    subspace is the whole space of modes.

    Parameters
    ----------
    omega : numpy.ndarray
        Frequencies of the bath modes.
    coupling : numpy.ndarray
        Complex matrix :math:`C_{\mu m}` of shape ``(nA, n_modes)``.
    n_blocks : int, optional
        Number of Lanczos blocks. Each block adds as many effective modes as
        the rank of the coupling. The default is 1.
    tol : float, optional
        Relative tolerance to discard linearly dependent directions.

    Returns
    -------
    omega_eff : numpy.ndarray
        Frequencies of the effective modes.
    coupling_eff : numpy.ndarray
        Couplings of the effective modes, shape ``(nA, n_effective)``.
    """
    n_modes = len(omega)
    if n_modes == 0:
        return omega, coupling

    def orthonormal_columns(vectors: np.ndarray) -> np.ndarray:
        u_mat, sing, _ = np.linalg.svd(vectors, full_matrices=False)
        if sing.size == 0 or sing[0] <= 1e-300:
            return u_mat[:, :0]
        return u_mat[:, sing > tol * sing[0]]

    basis = orthonormal_columns(np.conj(coupling).T)
    if basis.shape[1] == 0:
        return omega[:0], coupling[:, :0]
    last = basis
    for _ in range(n_blocks - 1):
        if basis.shape[1] >= n_modes:
            break
        vectors = omega[:, None] * last
        for _ in range(2):  # full reorthogonalization
            vectors = vectors - basis @ (basis.conj().T @ vectors)
        last = orthonormal_columns(vectors)
        if last.shape[1] == 0:
            break
        basis = np.hstack([basis, last])
    projected = basis.conj().T @ (omega[:, None] * basis)
    omega_eff, eigenvectors = np.linalg.eigh(0.5 * (projected + projected.conj().T))
    return omega_eff, coupling @ (basis @ eigenvectors)


def _bath_operators(n_modes: int, n_fock: int) -> List[sparse.csr_matrix]:
    """Annihilation operators of ``n_modes`` bosons truncated at ``n_fock`` quanta."""
    dim = n_fock + 1
    boson = sparse.diags(np.sqrt(np.arange(1, dim)), 1, format="csr")
    operators = []
    for j in range(n_modes):
        factors = [sparse.identity(dim, format="csr")] * n_modes
        factors[j] = boson
        full = factors[0]
        for factor in factors[1:]:
            full = sparse.kron(full, factor, format="csr")
        operators.append(full)
    return operators


def rpa_hybrid_state(
    structure: RPAStructure,
    sites: frozenset,
    k_mf: Operator,
    n_blocks: int = 1,
    n_fock: int = 6,
    n_cut: float = 3.0,
    n_gauss: int = 6,
    max_dim: int = 6000,
    max_nodes: int = 20000,
) -> Qobj:
    r"""Reduced state of the subsystem, exact in ``A`` and RPA in the environment.

    The environment is bosonized (RPA) and the subsystem is kept exact. The
    reduced state is the partial trace over the bosons of the Gibbs state of

    .. math::

        K = K_{A,MF} + \sum_j \omega_j b_j^\dagger b_j
            + \sum_\mu \delta e_\mu \otimes (B_\mu^{q} + x_\mu),

    where the quantum modes :math:`B^q_\mu` are the normal modes of the
    environment with thermal occupation not larger than ``n_cut``, compressed
    to a few effective modes with :func:`effective_bath_modes` and truncated
    in Fock space, and the modes with larger occupations are classical static
    Gaussian fields :math:`x_\mu` averaged with a Gauss-Hermite quadrature.
    Unlike the generator corrections of
    :func:`rpa_generator_correction`, it is non perturbative in the coupling
    between the subsystem and the environment. The back-reaction of the
    subsystem on the environment is therefore included.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites : frozenset
        Sites of the subsystem ``A``.
    k_mf : Operator
        Mean-field reduced generator :math:`K_{A,MF}` of the subsystem.
    n_blocks : int, optional
        Lanczos blocks of the effective bath. The default is 1.
    n_fock : int, optional
        Maximum number of quanta of each effective boson. The default is 6.
    n_cut : float, optional
        Modes with thermal occupation larger than ``n_cut`` are treated as
        classical. The default is 3.
    n_gauss : int, optional
        Gauss-Hermite nodes per classical direction. The default is 6.
    max_dim : int, optional
        Maximum dimension of the Hilbert space of the hybrid Hamiltonian.
    max_nodes : int, optional
        Maximum number of quadrature nodes of the classical fields.

    Returns
    -------
    qutip.Qobj
        The normalized reduced density matrix, in the tensor-product order of
        ``tuple(sorted(sites))``.

    Raises
    ------
    ValueError
        If the RPA is unstable, or if the hybrid problem is too large.
    """
    block = tuple(sorted(sites))
    system = structure.system
    k0_qobj: Qobj = k_mf.to_qutip(block)
    k0 = k0_qobj.full()
    k0 = 0.5 * (k0 + k0.conj().T)
    dim_a = k0.shape[0]

    def finish(rho: np.ndarray) -> Qobj:
        rho = 0.5 * (rho + rho.conj().T)
        return Qobj(rho / np.real(np.trace(rho)), dims=k0_qobj.dims)

    environment = environment_modes(structure, sites)
    if environment is None:
        eps, vecs = np.linalg.eigh(k0)
        return finish((vecs * np.exp(-(eps - eps.min()))) @ vecs.conj().T)
    omega, coupling, sites_a, local_basis = environment
    d_ops = [
        LocalOperator(site, e, system).to_qutip(block).full()
        for site in sites_a
        for e in local_basis[site]
    ]

    # Classical (static) and quantum modes
    occupation = 1.0 / np.expm1(omega)
    classical = occupation > n_cut
    n_a = len(d_ops)
    covariance = np.zeros((n_a, n_a))
    if np.any(classical):
        c_cl = coupling[:, classical]
        covariance = np.real(
            (c_cl * (2.0 * occupation[classical] + 1.0)[None, :]) @ c_cl.conj().T
        )
        covariance = 0.5 * (covariance + covariance.T)
    cov_vals, cov_vecs = np.linalg.eigh(covariance)
    keep = cov_vals > 1e-12 * max(1.0, cov_vals.max())
    fields = [
        np.sqrt(cov_vals[a]) * sum(cov_vecs[mu, a] * d_ops[mu] for mu in range(n_a))
        for a in np.nonzero(keep)[0]
    ]
    omega_eff, coupling_eff = effective_bath_modes(
        omega[~classical], coupling[:, ~classical], n_blocks
    )
    n_eff = len(omega_eff)
    bath_dim = (n_fock + 1) ** n_eff
    if dim_a * bath_dim > max_dim:
        raise ValueError(
            f"The hybrid problem has dimension {dim_a * bath_dim} > {max_dim}: "
            "reduce n_fock or n_blocks, or increase max_dim."
        )
    nodes_1d, weights_1d = np.polynomial.hermite_e.hermegauss(n_gauss)
    weights_1d = weights_1d / np.sqrt(2.0 * np.pi)
    n_nodes = n_gauss ** len(fields)
    if n_nodes > max_nodes:
        raise ValueError(
            f"{len(fields)} classical directions need {n_nodes} > {max_nodes} "
            "quadrature nodes: reduce n_gauss or increase max_nodes."
        )

    # Hamiltonian of the bath and coupling to the subsystem
    bosons = _bath_operators(n_eff, n_fock)
    identity_bath = sparse.identity(bath_dim, format="csr")
    identity_a = sparse.identity(dim_a, format="csr")
    bath_h = sparse.csr_matrix((bath_dim, bath_dim), dtype=complex)
    for w_j, boson in zip(omega_eff, bosons):
        bath_h = bath_h + w_j * (boson.T @ boson)
    h_fixed = sparse.kron(identity_a, bath_h, format="csr")
    for mu, d_op in enumerate(d_ops):
        b_mu = sparse.csr_matrix((bath_dim, bath_dim), dtype=complex)
        for j, boson in enumerate(bosons):
            b_mu = (
                b_mu
                + coupling_eff[mu, j] * boson
                + np.conj(coupling_eff[mu, j]) * boson.T
            )
        h_fixed = h_fixed + sparse.kron(sparse.csr_matrix(d_op), b_mu, format="csr")

    energies, states = [], []
    for index in np.ndindex(*([n_gauss] * len(fields))):
        xi = [nodes_1d[i] for i in index]
        weight = float(np.prod([weights_1d[i] for i in index]))
        k_a = k0 + sum(x * f for x, f in zip(xi, fields)) if fields else k0
        h_total = (
            h_fixed + sparse.kron(sparse.csr_matrix(k_a), identity_bath)
        ).toarray()
        eps, vecs = scipy.linalg.eigh(0.5 * (h_total + h_total.conj().T))
        energies.append((weight, eps[0]))
        boltz = np.exp(-(eps - eps[0]))
        psi = vecs.reshape(dim_a, bath_dim, -1)
        rho = np.einsum("anr,r,bnr->ab", psi, boltz, psi.conj())
        states.append(rho)
    e_min = min(e for _, e in energies)
    total = sum(w * np.exp(-(e - e_min)) * rho for (w, e), rho in zip(energies, states))
    return finish(total)


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


# ### Full bosonization: Gaussian state of the whole system ###


@dataclass
class BosonSystem:
    r"""Bosonized RPA of the whole system, and its Gaussian (thermal) state.

    The local algebra of every site is bosonized (see
    :func:`local_boson_modes`) and the quadratic bosonic Hamiltonian of the
    whole system is diagonalized with a Bogoliubov transformation. The
    thermal state of the quasi-particles is a Gaussian state, determined by
    the matrix of contractions
    :math:`\mathcal{G} = \langle\Phi\Phi^\dagger\rangle`, with
    :math:`\Phi = (b_1,\dots,b_n,b_1^\dagger,\dots,b_n^\dagger)^T`.

    Only the bosons with a non-vanishing frequency are included: the static
    (zero-frequency) modes are classical fluctuations, and do not contribute
    to the entanglement.

    Attributes
    ----------
    sites : list of str
        Sites of the system.
    mode_site : list of str
        Site to which each boson belongs.
    freqs : numpy.ndarray
        Local frequencies :math:`\Delta_k` of the bosons.
    couplings : numpy.ndarray
        Couplings :math:`g_{\mu k}` of the local operators to the bosons.
    lam : numpy.ndarray
        Signed frequencies of the normal modes.
    t_mat : numpy.ndarray
        Bogoliubov transformation.
    contractions : numpy.ndarray
        Matrix :math:`\langle\Phi\Phi^\dagger\rangle` of the thermal state.
    """

    sites: List[str]
    mode_site: List[str]
    freqs: np.ndarray
    couplings: np.ndarray
    lam: np.ndarray
    t_mat: np.ndarray
    contractions: np.ndarray

    @property
    def n_modes(self) -> int:
        """Number of bosons."""
        return len(self.freqs)

    def modes_of(self, sites: frozenset) -> List[int]:
        """Indices of the bosons of the given sites."""
        return [k for k, site in enumerate(self.mode_site) if site in sites]


def build_boson_system(structure: RPAStructure) -> BosonSystem:
    """Build the bosonized RPA of the whole system.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.

    Returns
    -------
    BosonSystem
        The bosonic system, with the contractions of its thermal state.

    Raises
    ------
    ValueError
        If the RPA is unstable.
    """
    sites = list(structure.positions)
    freqs_list, mode_site, blocks = [], [], []
    row = 0
    for site in sites:
        size = len(structure.local_basis[site])
        freqs, coup = structure.boson_modes(site)
        dynamic = freqs > structure.static_gap * (1.0 + 1e-6)
        freqs_list.append(freqs[dynamic])
        mode_site.extend([site] * int(np.sum(dynamic)))
        blocks.append((row, size, coup[:, dynamic]))
        row += size
    freqs_all = np.concatenate(freqs_list) if freqs_list else np.zeros(0)
    n_modes = len(freqs_all)
    g_mat = np.zeros((row, n_modes), dtype=complex)
    col = 0
    for r_0, size, coup in blocks:
        g_mat[r_0 : r_0 + size, col : col + coup.shape[1]] = coup
        col += coup.shape[1]
    m_full = structure.m_full
    a_mat = np.diag(freqs_all) + g_mat.conj().T @ m_full @ g_mat
    b_mat = g_mat.conj().T @ m_full @ g_mat.conj()
    a_mat = 0.5 * (a_mat + a_mat.conj().T)
    b_mat = 0.5 * (b_mat + b_mat.T)
    h_mat = np.block([[a_mat, b_mat], [b_mat.conj(), a_mat.conj()]])
    lam, t_mat = _bogoliubov(h_mat)
    omega = np.abs(lam)
    expo = np.where(lam > 0, 0.0, -omega)  # equal-time occupations
    occupation = np.exp(expo) / (-np.expm1(-omega))
    contractions = (t_mat * occupation[None, :]) @ t_mat.conj().T
    return BosonSystem(
        sites=sites,
        mode_site=mode_site,
        freqs=freqs_all,
        couplings=g_mat,
        lam=lam,
        t_mat=t_mat,
        contractions=contractions,
    )


def _covariance_matrix(bosons: BosonSystem, modes: List[int]) -> np.ndarray:
    r"""Covariance matrix of the quadratures of a subset of bosons.

    The quadratures are ordered as :math:`(x_1,\dots,x_n,p_1,\dots,p_n)` with
    :math:`x=(b+b^\dagger)/\sqrt2`, :math:`p=(b-b^\dagger)/(i\sqrt2)`, so that
    the vacuum has covariance :math:`1/2`.
    """
    n_all, n_sub = bosons.n_modes, len(modes)
    idx = np.array(list(modes) + [n_all + k for k in modes], dtype=int)
    g_sub = bosons.contractions[np.ix_(idx, idx)]
    eye = np.eye(n_sub)
    u_mat = np.block([[eye, eye], [-1j * eye, 1j * eye]]) / np.sqrt(2.0)
    return np.real(u_mat @ g_sub @ u_mat.conj().T)


def _symplectic_spectrum(sigma: np.ndarray) -> np.ndarray:
    """Symplectic eigenvalues :math:`\\nu_\\alpha` of a covariance matrix."""
    n_sub = sigma.shape[0] // 2
    if n_sub == 0:
        return np.zeros(0)
    eye, zero = np.eye(n_sub), np.zeros((n_sub, n_sub))
    omega = np.block([[zero, eye], [-eye, zero]])
    values = np.sort(np.abs(np.linalg.eigvals(1j * omega @ sigma)))
    return values[::2]


def rpa_entanglement_entropy(structure: RPAStructure, sites: frozenset) -> float:
    r"""Bosonic RPA entropy of a subsystem, in bits.

    It is the von Neumann entropy of the reduced Gaussian state of the
    bosons of ``sites``, :math:`S=\sum_\alpha[(1+f_\alpha)\log_2(1+f_\alpha)
    - f_\alpha\log_2 f_\alpha]` with :math:`f_\alpha=\nu_\alpha-1/2`.
    For a pure state (zero temperature) it is the entanglement entropy of
    the bipartition (``sites``, complement); otherwise it also includes the
    thermal entropy.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites : frozenset
        Sites of the subsystem.

    Returns
    -------
    float
        The entropy in bits.
    """
    bosons = structure.boson_system()
    modes = bosons.modes_of(sites)
    nu = _symplectic_spectrum(_covariance_matrix(bosons, modes))
    f = np.clip(nu - 0.5, 0.0, None)
    f = f[f > 1e-15]
    return float(np.sum((1 + f) * np.log2(1 + f) - f * np.log2(f)))


def rpa_negativity(
    structure: RPAStructure, sites_b: frozenset, sites_c: Optional[frozenset] = None
) -> float:
    r"""Bosonic RPA negativity between two subsystems.

    The negativity is computed from the symplectic spectrum of the partial
    transpose of the reduced Gaussian state of ``sites_b`` and ``sites_c``
    (:math:`\mathcal{N} = \frac12[\prod_{\tilde\nu<1/2}(2\tilde\nu)^{-1}-1]`).

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites_b, sites_c : frozenset
        Sites of the two subsystems. If ``sites_c`` is ``None``, the
        complement of ``sites_b`` is used.

    Returns
    -------
    float
        The negativity.
    """
    bosons = structure.boson_system()
    if sites_c is None:
        sites_c = frozenset(bosons.sites) - sites_b
    modes_b = bosons.modes_of(sites_b)
    modes_c = bosons.modes_of(sites_c)
    modes = modes_b + modes_c
    sigma = _covariance_matrix(bosons, modes)
    n_sub = len(modes)
    flip = np.ones(2 * n_sub)
    flip[n_sub + len(modes_b) :] = -1.0  # time reversal of the momenta of C
    sigma_pt = flip[:, None] * sigma * flip[None, :]
    nu = _symplectic_spectrum(sigma_pt)
    nu = nu[nu < 0.5 - 1e-12]
    return float(0.5 * (np.prod(1.0 / (2.0 * nu)) - 1.0))


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
    hybrid : bool, optional
        Use by default the non perturbative hybrid reduction, with the
        subsystem exact and the RPA only on the environment (see
        :func:`rpa_hybrid_state`). The default is False.
    hybrid_options : dict, optional
        Options of :func:`rpa_hybrid_state`.
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
        hybrid: bool = False,
        hybrid_options: Optional[dict] = None,
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
        self.hybrid = hybrid
        self.hybrid_options = hybrid_options
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
        hybrid: Optional[bool] = None,
        hybrid_options: Optional[dict] = None,
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
        hybrid : bool, optional
            Use the non perturbative hybrid reduction. If ``None``
            (default), use the value given when the state was built.
        hybrid_options : dict, optional
            Options of :func:`rpa_hybrid_state`. If ``None``, use the ones
            given when the state was built.

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
            hybrid=self.hybrid if hybrid is None else hybrid,
            hybrid_options=(
                self.hybrid_options if hybrid_options is None else hybrid_options
            ),
        )

    def reduce(self, sites, state=None, **kwargs):
        """Alias of :meth:`partial_trace`. ``state`` is ignored."""
        return self.partial_trace(sites, **kwargs)

    def rpa_entropy(self, sites: frozenset) -> float:
        """Bosonic RPA entropy of the subsystem ``sites``, in bits.

        See :func:`rpa_entanglement_entropy`.
        """
        return rpa_entanglement_entropy(self.rpa_structure, frozenset(sites))

    def rpa_negativity(
        self, sites_b: frozenset, sites_c: Optional[frozenset] = None
    ) -> float:
        """Bosonic RPA negativity between ``sites_b`` and ``sites_c``.

        See :func:`rpa_negativity`.
        """
        return rpa_negativity(
            self.rpa_structure,
            frozenset(sites_b),
            None if sites_c is None else frozenset(sites_c),
        )

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
            hybrid=self.hybrid,
            hybrid_options=self.hybrid_options,
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
