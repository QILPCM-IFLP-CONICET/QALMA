"""
RPAGibbsDensityOperator.

A sub-class of a GibbsDensityOperator that estimates partial traces
using RPA-like approximations.
"""

import logging
from typing import Optional

import numpy as np

from qalma.meanfield.gibbs_partial_trace import (
    gibbs_exact_partial_trace,
    gibbs_meanfield_generator,
)
from qalma.meanfield.variational import variational_quadratic_mfa
from qalma.model import SystemDescriptor
from qalma.operators import Operator
from qalma.operators.basic import LocalOperator
from qalma.operators.quadratic import QuadraticFormOperator
from qalma.operators.states.basic import DensityOperatorProtocol
from qalma.operators.states.gibbs import GibbsDensityOperator
from qalma.operators.states.qutip import BlockDensityOperator
from qalma.operators.states.register_ops._gibbs import _inherit_gibbs_arithmetic

from .bosonized import rpa_bosonized_state
from .dynamic import _dynamic_correction
from .environment import ENVIRONMENTS, environment_modes, rpa_zero_modes, static_kernel
from .functions import rpa_entanglement_entropy, rpa_negativity
from .hybrid import rpa_hybrid_state
from .rpa_structure import RPAStructure, build_rpa_structure, rpa_coupling_kernel
from .utils import _STATIC_GAP, _ZERO_MODE_RTOL

__all__ = ["RPAGibbsDensityOperator", "rpa_generator_correction"]

LOGGER = logging.getLogger(__name__)


def rpa_generator_correction(
    structure: RPAStructure,
    k_mf: Operator,
    sites: frozenset,
    dynamic: bool = False,
    n_tau: int = 32,
    environment: str = "isolated",
) -> Optional[Operator]:
    r"""RPA correction :math:`K_{A,RPA}` to the mean-field reduced generator.

    The reduced generator is :math:`K_A' = K_{A,MF} + K_{A,RPA}`.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA of the Gibbs state (see
        :func:`~qalma.meanfield.rpa.build_rpa_structure`).
    k_mf : Operator
        Mean-field reduced generator :math:`K_{A,MF}` of the subsystem
        (the terms of ``K`` inside ``sites`` plus the projected boundary
        terms). It is only used by the dynamical correction.
    sites : frozenset
        Sites of the subsystem ``A``.
    dynamic : bool, optional
        If ``True``, use the full imaginary-time kernel (dynamical
        correction, computed with the bosonized RPA) instead of the static
        approximation. This builds dense matrices of the size of the Hilbert
        space of ``A``. The default is False.
    n_tau : int, optional
        Initial number of imaginary-time quadrature nodes (dynamic only).
        They are doubled until the correction converges.
    environment : str, optional
        Origin of the correlations of the environment: ``"isolated"``
        (default) or ``"global"`` (see
        :func:`~qalma.meanfield.rpa.environment.environment_modes`).

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
    system = structure.system
    if dynamic:
        return _dynamic_correction(k_mf, structure, sites, n_tau, environment)

    if environment == "isolated":
        sigma, local_basis, sites_a = rpa_coupling_kernel(structure, sites)
    else:
        modes = environment_modes(structure, sites, environment)
        if modes is None:
            return None
        sigma = static_kernel(modes)
        local_basis, sites_a = modes.local_basis, modes.sites_a
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
        Initial number of imaginary-time quadrature nodes of the dynamical
        correction, which are doubled until it converges. The default is 32.
    static_gap : float, optional
        Frequency of the static bosons. The default is 1e-6.
    hybrid : bool, optional
        Use by default the non perturbative hybrid reduction, with the
        subsystem exact and the RPA only on the environment (see
        :func:`rpa_hybrid_state`). The default is False.
    hybrid_options : dict, optional
        Options of :func:`rpa_hybrid_state`.
    zero_rtol : float, optional
        Relative tolerance to identify the zero modes of the bosonic RPA
        problems, which are removed (see :func:`qalma.gaussian.bogoliubov_reduced`). The
        broken symmetry associated with them must be restored with the
        ``symmetry_projections`` of the state. The default is 1e-5.
    environment : str, optional
        Origin of the correlations of the environment in the static and
        dynamical corrections. ``"isolated"`` (default) bosonizes, for each
        subsystem, the environment with the subsystem removed. ``"global"``
        takes them from the bosonization of the whole system (J. M. Matera's
        thesis, chapter 6), which is computed once and shared by all the
        subsystems: it is cheaper for large systems, but it includes the
        fluctuations of the subsystem itself fed back through the
        environment, which makes it less accurate close to instabilities.
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
        zero_rtol: float = _ZERO_MODE_RTOL,
        environment: str = "isolated",
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
        self.zero_rtol = zero_rtol
        if environment not in ENVIRONMENTS:
            raise ValueError(
                f"environment must be one of {ENVIRONMENTS}, not {environment!r}."
            )
        self.environment = environment
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
                self.k, self.meanfield, self.static_gap, self.zero_rtol
            )
        return self._rpa_structure

    def reduced_generator(
        self,
        sites: frozenset,
        rpa: bool = True,
        dynamic: Optional[bool] = None,
        environment: Optional[str] = None,
    ) -> Operator:
        r"""Approximate generator :math:`K_A'` of the reduced state of ``sites``.

        It is the mean-field generator :math:`K_{A,MF}` plus, if ``rpa``, the
        RPA correction :math:`K_{A,RPA}`. No exact shortcut is taken.

        Parameters
        ----------
        sites : frozenset
            Sites of the subsystem ``A``.
        rpa : bool, optional
            Add the RPA correction. The default is True.
        dynamic : bool, optional
            Use the dynamical correction. If ``None`` (default), use the value
            given when the state was built.
        environment : str, optional
            Origin of the correlations of the environment (``"isolated"`` or
            ``"global"``). If ``None`` (default), use the value given when
            the state was built.

        Returns
        -------
        Operator
            The generator, acting on ``sites``.
        """
        sites = frozenset(sites)
        k_mf, has_boundary = gibbs_meanfield_generator(self, sites)
        if rpa and has_boundary:
            return self._add_rpa_correction(k_mf, sites, dynamic, environment)
        return k_mf

    def _add_rpa_correction(
        self,
        k_mf: Operator,
        sites: frozenset,
        dynamic: Optional[bool],
        environment: Optional[str] = None,
    ) -> Operator:
        """Add the RPA correction to the mean-field generator ``k_mf``."""
        correction = rpa_generator_correction(
            self.rpa_structure,
            k_mf,
            sites,
            dynamic=self.dynamic if dynamic is None else dynamic,
            n_tau=self.n_tau,
            environment=self.environment if environment is None else environment,
        )
        if correction is None:
            return k_mf
        return (k_mf + correction).simplify()

    def partial_trace(
        self,
        sites: frozenset | SystemDescriptor,
        max_exact_sites: Optional[int] = None,
        rpa: bool = True,
        dynamic: Optional[bool] = None,
        hybrid: Optional[bool] = None,
        hybrid_options: Optional[dict] = None,
        environment: Optional[str] = None,
    ) -> DensityOperatorProtocol:
        """Compute the partial trace over the complement of ``sites``.

        Reductions involving at most ``max_exact_sites`` sites are computed
        exactly, as in
        :meth:`~qalma.operators.states.gibbs.GibbsDensityOperator.partial_trace`,
        and the RPA options are then irrelevant. Otherwise, the reduced state
        is the mean-field approximation plus the RPA correction selected by
        ``rpa``, ``dynamic`` and ``hybrid``.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*. All other sites are traced out.
        max_exact_sites : int, optional
            Maximum number of sites of an exact reduction. If ``None``
            (default), use ``qalma.settings.MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE``.
            ``0`` always uses the approximation, which is needed to evaluate
            the RPA in small systems.
        rpa : bool, optional
            If ``True`` (default), include the RPA. If ``False``, return the
            mean-field reduction.
        dynamic : bool, optional
            Use the dynamical RPA correction. If ``None`` (default), use
            the value given when the state was built.
        hybrid : bool, optional
            Use the non perturbative hybrid reduction. If ``None``
            (default), use the value given when the state was built.
        hybrid_options : dict, optional
            Options of :func:`rpa_hybrid_state`. If ``None``, use the ones
            given when the state was built.
        environment : str, optional
            Origin of the correlations of the environment in the static and
            dynamical corrections: ``"isolated"`` (the environment with the
            subsystem removed, one Bogoliubov transformation per subsystem)
            or ``"global"`` (the bosonization of the whole system, computed
            once). If ``None`` (default), use the value given when the state
            was built. The hybrid reduction always uses ``"isolated"``.

        Returns
        -------
        DensityOperatorProtocol
            The reduced density operator.

        Raises
        ------
        ValueError
            If the hybrid reduction is requested with ``environment="global"``.
        """
        if isinstance(sites, SystemDescriptor):
            sites = frozenset(sites.sites)
        sites = frozenset(sites)
        result = gibbs_exact_partial_trace(self, sites, max_exact_sites)
        if result is not None:
            if rpa:
                LOGGER.debug(
                    "Exact reduction of %s: the RPA options are not used.",
                    sorted(sites),
                )
            return result

        hybrid = self.hybrid if hybrid is None else hybrid
        environment = self.environment if environment is None else environment
        if rpa and hybrid and environment != "isolated":
            raise ValueError(
                "The hybrid reduction needs the environment with the subsystem "
                'removed: use environment="isolated".'
            )
        subsystem = self.system.subsystem(sites)
        k_mf, has_boundary = gibbs_meanfield_generator(self, sites)
        if not (rpa and has_boundary):
            result = GibbsDensityOperator(
                k_mf, subsystem, prefactor=self.prefactor
            ).to_qutip_operator()
        elif hybrid:
            options = self.hybrid_options if hybrid_options is None else hybrid_options
            rho = rpa_hybrid_state(self.rpa_structure, sites, k_mf, **(options or {}))
            block = tuple(sorted(sites))
            result = BlockDensityOperator(
                rho,
                names={name: pos for pos, name in enumerate(block)},
                system=subsystem,
                prefactor=self.prefactor,
            )
        else:
            k_reduced = self._add_rpa_correction(k_mf, sites, dynamic, environment)
            result = GibbsDensityOperator(
                k_reduced, subsystem, prefactor=self.prefactor
            ).to_qutip_operator()
        # If there were non-trivial terms in the boundary, restore symmetries.
        if has_boundary:
            for symm in self.symmetry_projections:
                result = symm(result)
        return result

    def bosonized_state(
        self, sites: frozenset | SystemDescriptor
    ) -> DensityOperatorProtocol:
        """Reduced state of one or two sites from the bosonized RPA.

        It is computed from the Gaussian state of the bosonization of the
        whole system, which is stored in ``rpa_structure`` and shared by all
        the subsystems, so it is much cheaper than :meth:`partial_trace`
        with ``dynamic`` or ``hybrid``. See
        :func:`~qalma.meanfield.rpa.bosonized.rpa_bosonized_state`. The
        ``symmetry_projections`` of the state are applied to the result.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            One or two sites.

        Returns
        -------
        DensityOperatorProtocol
            The reduced density operator. It is Hermitian and normalized,
            but not necessarily positive.
        """
        if isinstance(sites, SystemDescriptor):
            sites = frozenset(sites.sites)
        block = tuple(sorted(sites))
        result: DensityOperatorProtocol = BlockDensityOperator(
            rpa_bosonized_state(self.rpa_structure, frozenset(block)),
            names={name: pos for pos, name in enumerate(block)},
            system=self.system.subsystem(frozenset(block)),
            prefactor=self.prefactor,
        )
        for symm in self.symmetry_projections:
            result = symm(result)
        return result

    def reduce(self, sites, state=None, **kwargs):
        """Alias of :meth:`partial_trace`. ``state`` is ignored."""
        return self.partial_trace(sites, **kwargs)

    def rpa_zero_modes(self, sites: Optional[frozenset] = None) -> int:
        """Return the number of zero modes removed from the bosonic RPA problem.

        See :func:`~qalma.meanfield.rpa.environment.rpa_zero_modes`.
        """
        return rpa_zero_modes(
            self.rpa_structure, None if sites is None else frozenset(sites)
        )

    def rpa_entropy(self, sites: frozenset) -> float:
        """Bosonic RPA entropy of the subsystem ``sites``, in bits.

        See :func:`~qalma.meanfield.rpa.functions.rpa_entanglement_entropy`.
        """
        return rpa_entanglement_entropy(self.rpa_structure, frozenset(sites))

    def rpa_negativity(
        self, sites_b: frozenset, sites_c: Optional[frozenset] = None
    ) -> float:
        """Bosonic RPA negativity between ``sites_b`` and ``sites_c``.

        See :func:`~qalma.meanfield.rpa.functions.rpa_negativity`.
        """
        return rpa_negativity(
            self.rpa_structure,
            frozenset(sites_b),
            None if sites_c is None else frozenset(sites_c),
        )

    def _with_prefactor(self, prefactor: complex) -> "RPAGibbsDensityOperator":
        """Copy of the state with a different prefactor, sharing the structure."""
        assert isinstance(prefactor, float), "prefactor must be a float."
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
            zero_rtol=self.zero_rtol,
            environment=self.environment,
        )
        new_state._rpa_structure = self._rpa_structure
        return new_state

    def __truediv__(self, operand):
        if isinstance(operand, (int, float)) and operand > 0:
            return self._with_prefactor(self.prefactor / operand)
        return super().__truediv__(operand)


_inherit_gibbs_arithmetic(RPAGibbsDensityOperator)
