# -*- coding: utf-8 -*-
"""Qutip representation of an operator."""

import logging
from functools import reduce
from typing import Dict, Iterable, List, Optional, Tuple, Union

from numpy import imag, log as np_log, real
from qutip import Qobj as _Qobj, tensor as _tensor  # type: ignore[import-untyped]

from qalma.model import SystemDescriptor, build_system_from_dims
from qalma.operators.basic import (
    LocalOperator,
    Operator,
)
from qalma.operators.product import (
    ProductOperator,
    ScalarOperator,
)
from qalma.qutip_tools.tools import (
    decompose_qutip_operator,
    decompose_qutip_operator_hermitian,
    empty_op,
    is_diagonal_op,
    scalar_value,
)


class QutipOperator(Operator):
    """Represents a Qutip operator that acts over a block of sites of a system.

    If two QutipOperator are combined in an arithmetic
    operation, the result is QutipOperator acting on
    the union of both blocks.


    """

    prefactor: complex
    system: SystemDescriptor
    operator: _Qobj
    site_names: dict

    def __init__(
        self,
        qoperator,
        system: Optional[SystemDescriptor] = None,
        names: Optional[Dict[str, int]] = None,
        prefactor=1,
    ):
        # If build from a scalar:
        if not isinstance(qoperator, _Qobj):
            prefactor = prefactor * qoperator
            qoperator = None
            names = {}

        dims = [] if qoperator is None else qoperator.dims[0]
        if system is None:
            if names is None:
                names = {f"qutip_{i}": i for i in range(len(dims))}
            dims_names = {name: dims[pos] for name, pos in names.items()}
            system = build_system_from_dims(dims_names)
        else:
            # Check that names is correct, and compatible
            if names is None:
                names = {s: i for i, s in enumerate(system.sites)}
            elif len(names) != len(dims):
                raise ValueError(
                    f"dimensions {qoperator.dims[0]} and name dictionary {names} do not match."
                )
            elif any(pos >= len(dims) for pos in names.values()):
                raise ValueError(f"names {names} points out of dims {dims}")

        self.system = system
        self.operator = qoperator
        self.site_names = names
        self.prefactor = prefactor

    def __neg__(self):
        """Multiply by -1."""
        return QutipOperator(
            self.operator,
            self.system,
            names=self.site_names,
            prefactor=-self.prefactor,
        )

    def __pow__(self, exponent):
        """Exponentiate the operator at a given power."""
        operator = self.operator
        if exponent < 0:
            operator = operator.inv()
            exponent = -exponent

        return QutipOperator(
            operator**exponent,
            system=self.system,
            names=self.site_names,
            prefactor=1 / self.prefactor**exponent,
        )

    def __repr__(self) -> str:
        """Built the repr str."""
        return (
            f"qutip interface operator over sites {self.site_names} for {self.prefactor} x  \n"
            + repr(self.operator)
        )

    def acts_over(self) -> frozenset:
        """List the sites where the operator acts over."""
        return frozenset(self.site_names.keys())

    def as_sum_of_products(self):
        """Decompose the operator as a sum of product operators."""
        from qalma.operators.arithmetic import SumOperator

        isherm = self.operator.isherm
        site_names = self.site_names
        sites = sorted(site_names, key=lambda x: site_names[x])
        if isherm:
            decomposition = decompose_qutip_operator_hermitian(
                self.prefactor * self.operator.tidyup()
            )
        else:
            decomposition = decompose_qutip_operator(
                self.prefactor * self.operator.tidyup()
            )
        terms = tuple(
            (
                ProductOperator(
                    dict(zip(sites, term)),
                    prefactor=1.0,
                    system=self.system,
                ).simplify()
                for term in decomposition
            )
        )
        if isherm:
            assert all(
                term.isherm for term in terms
            ), f"{[(type(term), term.isherm) for term in terms]}"
        if len(terms) == 0:
            terms = tuple((ScalarOperator(0, self.system),))
        return SumOperator(terms, self.system, isherm=isherm)

    def dag(self):
        """Build the hermitian adjoint operator."""
        prefactor = self.prefactor
        operator = self.operator
        if isinstance(prefactor, complex):
            prefactor = prefactor.conjugate()
        else:
            if operator.isherm:
                return self
        return QutipOperator(
            operator.dag(),
            system=self.system,
            names=self.site_names,
            prefactor=prefactor,
        )

    def eigenenergies(self):
        """Compute the spectrum of the operator."""
        return self.operator.eigenenergies() * self.prefactor

    def eigenstates(self):
        """Compute the eigendecomposition of the operator."""
        evals, evecs = self.operator.eigenstates()
        return evals * self.prefactor, evecs

    def hermitian_part(self):
        """Compute the hermitian part of the operator."""
        if self.isherm:
            return self
        qop = self.operator
        prefactor = self.prefactor
        if qop.isherm:
            prefactor = real(prefactor)
        else:
            qop = qop * (0.5 * prefactor)
            qop = qop + qop.dag()
            prefactor = 1.0
        return QutipOperator(qop, self.system, self.site_names, prefactor)

    def inv(self):
        """Compute the inverse of the operator."""
        operator = self.operator
        return QutipOperator(
            operator.inv(),
            system=self.system,
            names=self.site_names,
            prefactor=1 / self.prefactor,
        )

    @property
    def isherm(self) -> bool:
        """
        Check if the operator is hermitian.

        Return True if operator is hermitian.
        """
        isherm = self.operator.isherm
        if imag(self.prefactor) == 0.0:
            return isherm
        # herm operator with complex prefactor
        if isherm:
            return False
        # should this be cached?
        return (self.operator * self.prefactor).isherm

    @property
    def isdiagonal(self) -> bool:
        """Check if the operator is diagonal."""
        return is_diagonal_op(self.operator)

    @property
    def is_zero(self) -> bool:
        """Check if the matrix is zero."""
        return not (self.prefactor) or empty_op(self.operator)

    def logm(self):
        """Compute the logarithm of the operator."""
        operator = self.operator
        evals, evecs = operator.eigenstates()
        evals = evals * self.prefactor
        evals[abs(evals) < 1.0e-50] = 1.0e-50
        if any(value < 0 for value in evals):
            evals = (1.0 + 0j) * evals
        log_op = sum(
            np_log(e_val) * e_vec * e_vec.dag() for e_val, e_vec in zip(evals, evecs)
        )
        return QutipOperator(log_op, self.system, self.site_names)

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        """
        Compute the partial trace.

        Parameters
        ----------
        sites: Union[frozenset :

        SystemDescriptor] :


        Returns
        -------
        Operator
        The partial trace of the operator.

        """
        if isinstance(sites, SystemDescriptor):
            subsystem = sites
            sites = frozenset(site for site in subsystem.sites)
        else:
            subsystem = self.system.subsystem(sites)
            sites = frozenset(sites)

        if len(sites) == 0:
            return ScalarOperator(self.tr(), subsystem)

        prefactor = self.prefactor
        system = self.system
        dimensions = system.dimensions
        site_names = self.site_names
        partial_site_names = {
            site: pos for site, pos in site_names.items() if site in sites
        }
        keep = tuple(partial_site_names.values())
        if len(keep) == 0:
            # compute the trace of the block,
            # and multiply by the prefactor
            prefactor *= self.operator.tr()
            # Now, multiply by the dimensions not included in
            # sites or site_names
            dims_other = (
                dim
                for site, dim in dimensions.items()
                if site not in site_names and site not in sites
            )
            prefactor = reduce(lambda x, y: x * y, dims_other, prefactor)
            return ScalarOperator(prefactor, subsystem)

        new_qutip_op = self.operator.ptrace(keep)
        new_site_names = {
            site: i
            for i, site in enumerate(
                sorted(partial_site_names, key=lambda x: partial_site_names[x])
            )
        }
        other_dims = (
            dim
            for site, dim in dimensions.items()
            if (site not in sites and site not in site_names)
        )
        new_prefactor = reduce(lambda x, y: x * y, other_dims, self.prefactor)
        return QutipOperator(
            new_qutip_op,
            subsystem,
            names=new_site_names,
            prefactor=new_prefactor,
        )

    def reduce(self, sites: Iterable, state=None) -> Operator:
        """Compute the reduced operator.

        Partial trace of the product of the operator and the density
        operator acting on the subsystem which is traced out. If the state is
        not provided, the result is the partial trace, divided by the dimension
        of the subsystem traced out.

        Parameters
        ----------
        sites: Iterable

        state: Optional[DensityOperatorProtocol]
               The state relative to which make the reduction.

        Return
        ------

        The reduced operator.

        """
        system = self.system
        prefactor = self.prefactor
        if prefactor == 0:
            return ScalarOperator(0, system)

        acts_over = self.acts_over()
        sites = acts_over.intersection(sites)
        environment = acts_over - sites
        if not environment:
            return self
        if not sites:
            if state is None:
                prefactor = self.tr()
                dimensions = system.dimensions
                prefactor /= reduce(
                    lambda x, y: x * y, (dimensions[site] for site in environment), 1.0
                )
                return ScalarOperator(prefactor, system)
            return ScalarOperator(state.expect(self), system)

        if state is None:
            # Is state is not provided, just compute the partial trace
            # on the block, and divide by the dimension of the environment.
            dims = system.dimensions
            site_list = sorted(sites)
            site_names = self.site_names
            qop = self.operator.ptrace([site_names[site] for site in site_list])
            for site in environment:
                prefactor /= dims[site]

            return QutipOperator(
                qop,
                system,
                names={site: i for i, site in enumerate(site_list)},
                prefactor=prefactor,
            )

        env_tuple = tuple(environment)
        sites_tuple = tuple(sites)
        qop = self.to_qutip(sites_tuple + env_tuple)
        state_qutip = state.partial_trace(environment).to_qutip(env_tuple)
        state_qutip = _tensor(
            *(system.site_identity(site) for site in sites_tuple), state_qutip
        )
        qop = (qop * state_qutip).ptrace(list(range(len(sites_tuple))))
        return QutipOperator(
            qop,
            system,
            names={site: i for i, site in enumerate(sites_tuple)},
        )

    def simplify(self):
        """Simplify the operator."""
        names = self.site_names
        prefactor = self.prefactor
        qt_operator = self.operator
        system = self.system
        if prefactor == 0:
            return ScalarOperator(0.0, system)
        assert len(names) > 0

        # If is an empty op, return a ScalarOperator
        if empty_op(qt_operator):
            return ScalarOperator(0.0, self.system)

        if len(names) > 1:
            return self

        # The operator acts on a single site. Check if is an scalar
        s_val = scalar_value(qt_operator.data)
        if s_val is not None:
            return ScalarOperator(s_val * self.prefactor, self.system)
        # Otherwise, return a local operator:
        (site,) = names.keys()
        operator = self.operator * self.prefactor
        return LocalOperator(site, operator, self.system)

    def tidyup(self, atol=None):
        """Remove small elements from the quantum object.

        Parameters
        ----------
        atol : float
            the threshold
            (Default value = None)

        Return
        ------
        Operator
            An operator with all their matrix elements having absolute values
            larger than the threshold.

        """
        return QutipOperator(
            self.operator.tidyup(atol),
            system=self.system,
            names=self.site_names,
            prefactor=self.prefactor,
        )

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """
        Return a Qobj representation for the operator.

        Parameters
        ----------
        block: Optional[Tuple[str]] :
             The sites defining the block on which we want the
             prepresentation. If ``None``, returns a Qobj acting
             on all the sites.

        Returns
        -------
        Operator
            sites in block.
            By default (`block`=`None`), returns an operator
            acting over the full system, with sites sorted in
            lexicographical order.
            If `block`=`(,)` (the empty tuple), returns
            `self.operator`.

        """
        site_names_dict = self.site_names
        site_names = sorted(site_names_dict, key=lambda x: site_names_dict[x])
        system = self.system
        sites = system.sites
        operator_qutip: _Qobj = self.operator * self.prefactor
        if block is None:
            if len(sites) > 8:
                logging.warning(
                    (
                        "to_qutip does not received a block. "
                        "Return an operator over the full system"
                    )
                )
            block = tuple(sorted(self.system.sites.keys()))

        if len(block) == 0 or list(block) == site_names:
            return operator_qutip

        # Look for sites in block that are not in site_names
        out_sites = tuple(
            (site for site in block if site not in site_names_dict and site in sites)
        )
        # Add identities and operators in block but not in site_names
        if out_sites:
            next_index: int = len(site_names)
            site_names_dict = site_names_dict.copy()
            site_names_dict.update(
                {site: next_index + i for i, site in enumerate(out_sites)}
            )
            extra_identities = (sites[site]["identity"] for site in out_sites)
            operator_qutip = (
                _tensor(operator_qutip, *extra_identities)
                if site_names
                else _tensor(*extra_identities)
            )

        # Add sites which are in site_names, but not in block
        block = block + tuple((site for site in site_names if site not in block))
        shuffle: List[int] = list(site_names_dict[site] for site in block)
        assert len(shuffle) == len(
            operator_qutip.dims[0]
        ), f"len({shuffle})!=len({operator_qutip.dims[0]})"
        if shuffle == sorted(shuffle):
            return operator_qutip
        return operator_qutip.permute(shuffle)

    def tr(self) -> complex:
        """Compute the trace."""
        prefactor = self.prefactor
        if prefactor == 0:
            return prefactor

        site_names: Dict[str, int] = self.site_names
        op_tr = self.operator.tr() if site_names else 1.0
        if op_tr == 0.0:
            return 0.0

        system: SystemDescriptor = self.system
        dimensions: Dict[str, int] = system.dimensions
        if len(site_names) < len(dimensions):
            names = set(site_names)
            dims_other = (dim for site, dim in dimensions.items() if site not in names)
            prefactor = reduce(lambda x, y: x * y, dims_other, prefactor)
        result = op_tr * prefactor
        return result
