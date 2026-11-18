r"""Ground state of a Hamiltonian, stored as a matrix product state.

:class:`MPSGSOperator` is a density operator :math:`\rho = |\psi_0\rangle
\langle\psi_0|` whose internal representation is an MPS, computed with a DMRG
sweep (see :mod:`qalma.mps.dmrg`) or supplied by the user. Reduced states of
small subsystems are obtained by contracting the MPS, so that expectation
values of few-body observables, which is what the rest of QALMA needs from a
state, are available for systems far beyond exact diagonalization.

The ground state is assumed to be non-degenerate. For a degenerate manifold
the limit :math:`\beta \to \infty` of a Gibbs state is a mixture of the
states of the manifold, which this class does not represent.
"""

import logging
import pickle
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from qutip import Qobj

from qalma.model import SystemDescriptor
from qalma.mps.dmrg import dmrg
from qalma.mps.mpo import build_mpo
from qalma.mps.mps import MPS
from qalma.operators.arithmetic import SumOperator
from qalma.operators.basic import Operator
from qalma.operators.states.basic import DensityOperatorMixin
from qalma.operators.states.product import ProductDensityOperator
from qalma.operators.states.qutip import QutipDensityOperator
from qalma.operators.states.register_ops._gibbs import _inherit_gibbs_arithmetic
from qalma.operators.states.register_ops._types import COMPLEX_NUMERIC_TYPES

LOGGER = logging.getLogger(__name__)

__all__ = ["MPSGSOperator"]


def _resolve_order(
    sites: Sequence[str], permutation: Optional[Sequence[Union[int, str]]]
) -> Tuple[str, ...]:
    """Order of the sites along the chain.

    ``permutation[p] = i`` means that position ``p`` of the chain holds the
    ``i``-th site in lexicographic order. As a convenience, a sequence with the
    site names themselves is also accepted.
    """
    lexicographic = tuple(sorted(sites))
    if permutation is None:
        return lexicographic
    permutation = list(permutation)
    if len(permutation) != len(lexicographic):
        raise ValueError(
            f"permutation has {len(permutation)} entries for "
            f"{len(lexicographic)} sites."
        )
    if all(isinstance(item, str) for item in permutation):
        if sorted(permutation) != list(lexicographic):
            raise ValueError("The site names do not match the system.")
        return tuple(permutation)
    if sorted(int(item) for item in permutation) != list(range(len(lexicographic))):
        raise ValueError("permutation must be a permutation of range(n_sites).")
    return tuple(lexicographic[int(i)] for i in permutation)


def hamiltonian_terms(
    hamiltonian: Operator, order: Sequence[str]
) -> Tuple[list, complex]:
    """Split a Hamiltonian into dense few-body terms along a chain.

    Parameters
    ----------
    hamiltonian : Operator
        The Hamiltonian.
    order : sequence of str
        Site names, in the order they have along the chain.

    Returns
    -------
    terms : list of (positions, matrix)
        Ascending positions and Hermitian dense matrix of every term.
    offset : complex
        Sum of the terms that act on no site.
    """
    position = {site: pos for pos, site in enumerate(order)}
    flat = hamiltonian.flat()
    pieces = flat.terms if isinstance(flat, SumOperator) else (flat,)
    terms = []
    offset: complex = 0.0
    for piece in pieces:
        acts_over = piece.acts_over()
        if not acts_over:
            value = piece.to_qutip(tuple())
            if isinstance(value, Qobj):
                raise ValueError("Unexpected non-scalar term acting on no site.")
            offset += value
            continue
        positions = tuple(sorted(position[site] for site in acts_over))
        block = tuple(order[pos] for pos in positions)
        matrix = np.asarray(piece.to_qutip(block).full())
        terms.append((positions, 0.5 * (matrix + matrix.conj().T)))
    return terms, offset


class MPSGSOperator(DensityOperatorMixin, Operator):
    r"""Pure density operator :math:`|\psi_0\rangle\langle\psi_0|` stored as an MPS.

    The state is the ground state of ``hamiltonian``, found on demand with a
    two-site DMRG, or an MPS provided with :meth:`from_mps`.

    It is a *sibling* of
    :class:`~qalma.operators.states.gibbs.GibbsDensityOperator`: it has the
    same arithmetic and the same interface for ``expect`` and
    ``partial_trace``, but it has no finite generator ``K`` (the attribute
    ``k`` is not defined), since :math:`\log\rho` is unbounded for a pure
    state.

    Parameters
    ----------
    hamiltonian : Operator, optional
        The Hamiltonian. Required unless ``mps`` is given.
    system : SystemDescriptor, optional
        Defaults to ``hamiltonian.system``.
    permutation : sequence of int, optional
        Order of the sites along the MPS. ``permutation[p] = i`` puts the
        ``i``-th site (in lexicographic order of the names) at position ``p``
        of the chain. Notice that lexicographic order is not the natural
        one for names such as ``"1[10]"`` and ``"1[2]"``. A sequence with the
        site names in the desired order is also accepted. Default: the
        lexicographic order.
    mps : MPS, optional
        An already computed state, whose positions follow the same order.
    chi_max : int, optional
        Maximum bond dimension of the DMRG. Default 64.
    prefactor : float, optional
        Positive weight, relevant only when mixing states. Default 1.
    dmrg_options : dict, optional
        Extra keyword arguments for :func:`qalma.mps.dmrg.dmrg`.

    Attributes
    ----------
    order : tuple of str
        Names of the sites in the order they have in the MPS.
    energy : float or None
        Ground state energy found by the DMRG (computed on demand; ``None``
        when the state was provided).
    dmrg_info : dict or None
        Diagnostics of the DMRG run.
    """

    #: Largest number of sites for which the dense matrix can be requested.
    MAX_DENSE_SITES = 10

    def __init__(
        self,
        hamiltonian: Optional[Operator] = None,
        system: Optional[SystemDescriptor] = None,
        permutation: Optional[Sequence[Union[int, str]]] = None,
        *,
        mps: Optional[MPS] = None,
        chi_max: int = 64,
        prefactor: float = 1.0,
        dmrg_options: Optional[Dict[str, Any]] = None,
        energy: Optional[float] = None,
    ):
        if hamiltonian is None and mps is None:
            raise ValueError("Either a Hamiltonian or an MPS is required.")
        if system is None:
            if hamiltonian is None:
                raise ValueError("A system is required when only an MPS is given.")
            system = hamiltonian.system
        assert np.imag(prefactor) == 0 and np.real(prefactor) > 0
        self.system = system
        self.prefactor = prefactor
        self.hamiltonian = hamiltonian
        self.chi_max = chi_max
        self.dmrg_options = dict(dmrg_options or {})
        self.order = _resolve_order(tuple(system.sites), permutation)
        self._position = {site: pos for pos, site in enumerate(self.order)}
        # Results of the DMRG. The dict is shared with the copies made by the
        # arithmetic (``2 * rho``), so the DMRG runs at most once for all of them.
        self._solution: Dict[str, Any] = {"mps": None, "energy": energy, "info": None}
        if mps is not None:
            dims = tuple(system.dimensions[site] for site in self.order)
            if mps.dims != dims:
                raise ValueError(
                    f"The MPS has local dimensions {mps.dims}, expected {dims} "
                    "for the given site order."
                )
            self._solution["mps"] = mps

    @classmethod
    def from_mps(
        cls,
        mps: MPS,
        system: SystemDescriptor,
        permutation: Optional[Sequence[Union[int, str]]] = None,
        hamiltonian: Optional[Operator] = None,
        prefactor: float = 1.0,
    ) -> "MPSGSOperator":
        """Wrap an MPS (for instance, computed by another DMRG code)."""
        return cls(
            hamiltonian,
            system,
            permutation,
            mps=mps,
            prefactor=prefactor,
        )

    # ------------------------------------------------------------------
    # The MPS
    # ------------------------------------------------------------------
    def solve(self, **options) -> "MPSGSOperator":
        """Run the DMRG (again), optionally overriding its options.

        Keyword arguments are passed to :func:`qalma.mps.dmrg.dmrg`; ``chi_max``
        also updates the stored value. Returns ``self``.
        """
        if self.hamiltonian is None:
            raise ValueError("The state was built without a Hamiltonian.")
        if "chi_max" in options:
            self.chi_max = options["chi_max"]
        settings = {**self.dmrg_options, **options}
        settings["chi_max"] = self.chi_max
        terms, offset = hamiltonian_terms(self.hamiltonian, self.order)
        dims = [self.system.dimensions[site] for site in self.order]
        mpo = build_mpo(terms, dims)
        mps, info = dmrg(mpo, **settings)
        if not info["converged"]:
            LOGGER.warning(
                "DMRG did not converge after %d sweeps (last energy %.10f).",
                info["sweeps"],
                info["energy"],
            )
        self._solution.update(
            mps=mps, energy=float(info["energy"] + np.real(offset)), info=info
        )
        return self

    @property
    def solved(self) -> bool:
        """Whether the MPS is available (computed or provided)."""
        return self._solution["mps"] is not None

    @property
    def mps(self) -> MPS:
        """The MPS. The DMRG runs the first time it is needed."""
        if not self.solved:
            self.solve()
        return self._solution["mps"]

    @property
    def dmrg_info(self) -> Optional[Dict[str, Any]]:
        """Diagnostics of the DMRG run (``None`` if it did not run)."""
        return self._solution["info"]

    @property
    def energy(self) -> Optional[float]:
        """Ground state energy found by the DMRG.

        Triggers the DMRG if the state has not been computed yet. It is
        ``None`` for states provided with :meth:`from_mps`, unless the
        energy was given explicitly.
        """
        if (
            self._solution["energy"] is None
            and not self.solved
            and self.hamiltonian is not None
        ):
            self.solve()
        return self._solution["energy"]

    @property
    def bond_dims(self) -> Tuple[int, ...]:
        """Bond dimensions of the MPS."""
        return self.mps.bond_dims

    @property
    def k(self):
        """Not defined: a pure state has no finite generator."""
        raise AttributeError(
            "MPSGSOperator is a pure state and has no finite generator `k`."
        )

    def __repr__(self) -> str:
        if not self.solved:
            return f"MPSGSOperator(unsolved, {len(self.order)} sites)"
        energy = self._solution["energy"]
        energy_str = "?" if energy is None else f"{energy:.8g}"
        return (
            f"MPSGSOperator({len(self.order)} sites, "
            f"max bond={max(self.bond_dims, default=1)}, E0={energy_str})"
        )

    # ------------------------------------------------------------------
    # Interface of the density operators
    # ------------------------------------------------------------------
    def acts_over(self) -> frozenset:
        """Return the set of sites of the system."""
        return frozenset(self.system.sites)

    def normalize(self) -> "MPSGSOperator":
        """No-op: the MPS is normalized by construction."""
        return self

    def logm(self) -> Operator:
        """Not available: :math:`\\log\\rho` is unbounded for a pure state."""
        raise NotImplementedError("The logarithm of a pure state is not defined.")

    def variational_free_energy(self, ham: Operator) -> float:
        r"""Return :math:`\mathrm{Tr}[\rho (H + \log\rho)] = \langle H\rangle`.

        The entropy of a pure state vanishes.
        """
        return float(np.real(self.expect(ham)))

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor], **_kwargs):
        """Reduced state of ``sites`` as a :class:`QutipDensityOperator`.

        Extra keyword arguments (``rpa``, ``dynamic``, ``exact``, ...) of the
        interface of :class:`GibbsDensityOperator` are accepted and ignored.
        The cost grows exponentially with the number of sites kept, and
        linearly with the distance between the first and last of them along
        the chain.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*.
        """
        if isinstance(sites, SystemDescriptor):
            subsystem = sites
            sites = frozenset(s for s in subsystem.sites if s in self._position)
        else:
            sites = frozenset(sites)
            unknown = sites.difference(self._position)
            if unknown:
                raise ValueError(f"sites {sorted(unknown)} are not in the system.")
            subsystem = self.system.subsystem(sites)
        if not sites:
            return ProductDensityOperator({}, self.prefactor, subsystem)
        positions = sorted(self._position[site] for site in sites)
        block = tuple(self.order[pos] for pos in positions)
        rho = self.mps.reduced_density_matrix(positions)
        dims = [self.system.dimensions[site] for site in block]
        rho_qobj = Qobj(rho, dims=[dims, dims])
        names = {site: pos for pos, site in enumerate(block)}
        return QutipDensityOperator(
            rho_qobj, names=names, system=subsystem, prefactor=self.prefactor
        )

    def reduce(self, sites, state=None, **kwargs):
        """Alias of :meth:`partial_trace`. ``state`` is ignored."""
        return self.partial_trace(sites, **kwargs)

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """Return the dense matrix of the state restricted to ``block``.

        As for :class:`GibbsDensityOperator`, ``block`` lists the sites that
        are kept, in the order of the tensor factors of the result; the
        default is all the sites in lexicographic order. The prefactor is not
        included.

        Raises
        ------
        ValueError
            If more than :attr:`MAX_DENSE_SITES` sites are requested.
        """
        if block is None:
            block = tuple(sorted(self.system.sites))
        block = tuple(block)
        if len(set(block)) != len(block) or not set(block).issubset(self._position):
            raise ValueError("block must contain distinct sites of the system.")
        if len(block) > self.MAX_DENSE_SITES:
            raise ValueError(
                f"A dense matrix of {len(block)} sites was requested; the limit "
                f"is {self.MAX_DENSE_SITES} (see MPSGSOperator.MAX_DENSE_SITES). "
                "Use partial_trace and expect on small subsystems instead."
            )
        if not block:
            return 1.0
        by_position = sorted(block, key=self._position.__getitem__)
        rho = self.mps.reduced_density_matrix([self._position[s] for s in by_position])
        dims = [self.system.dimensions[site] for site in by_position]
        n_kept = len(dims)
        if by_position != list(block):
            axes = [by_position.index(site) for site in block]
            rho = rho.reshape(dims + dims).transpose(axes + [n_kept + a for a in axes])
            dims = [dims[a] for a in axes]
            rho = rho.reshape(int(np.prod(dims)), -1)
        return Qobj(rho, dims=[dims, dims])

    # ------------------------------------------------------------------
    # Arithmetic helpers
    # ------------------------------------------------------------------
    def _with_prefactor(self, prefactor: complex) -> "MPSGSOperator":
        """Copy of the state with a different prefactor, sharing the MPS."""
        new_state = object.__new__(type(self))
        new_state.__dict__.update(self.__dict__)
        new_state.prefactor = prefactor
        return new_state

    def __getstate__(self):
        """State for pickling.

        Unlike :meth:`DensityOperatorMixin.__getstate__` nothing is cached,
        since the MPS can be computed after the first pickling.
        """
        return pickle.dumps(dict(self.__dict__))

    def __truediv__(self, operand):
        """Divide by a positive scalar (rescales the prefactor) or an operator."""
        if isinstance(operand, (int, float)) and operand > 0:
            return self._with_prefactor(self.prefactor / operand)
        if isinstance(operand, Operator):
            return self * operand.inv()
        raise ValueError("Division of an operator by ", type(operand), " not defined.")


def _register_arithmetic() -> None:
    """Register the arithmetic of :class:`MPSGSOperator`.

    The handlers of :class:`GibbsDensityOperator` are inherited (they convert
    the state to a dense operator, so they are meant for small systems). The
    ones that would rebuild a ``GibbsDensityOperator`` from the generator ``k``
    are replaced by handlers that preserve the type.
    """
    cls = MPSGSOperator
    _inherit_gibbs_arithmetic(cls)

    add_table = Operator.__add__dispatch__
    mul_table = Operator.__mul__dispatch__
    add_fallback = add_table[(cls, cls)]

    def add_same(x_op, y_op):
        if x_op is y_op:
            return x_op._with_prefactor(2 * x_op.prefactor)  # pylint: disable=W0212
        return add_fallback(x_op, y_op)

    add_table[(cls, cls)] = add_same

    def scaled(fallback):
        def handler(x_op, y_op):
            state, factor = (x_op, y_op) if isinstance(x_op, cls) else (y_op, x_op)
            value = complex(factor)
            if value.imag == 0 and value.real > 0:
                return state._with_prefactor(  # pylint: disable=protected-access
                    state.prefactor * value.real
                )
            return fallback(x_op, y_op)

        return handler

    for other in COMPLEX_NUMERIC_TYPES:
        for key in ((cls, other), (other, cls)):
            if key in mul_table:
                mul_table[key] = scaled(mul_table[key])


_register_arithmetic()
