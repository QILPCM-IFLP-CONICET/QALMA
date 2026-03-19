"""
Different representations for operators
"""

import logging
from functools import cached_property, reduce
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import qutip  # type: ignore[import-untyped]
from qutip import Qobj

from qalma.model import SystemDescriptor
from qalma.qutip_tools.tools import (
    _to_array,
    empty_op,
    is_diagonal_op,
    is_scalar_op,
    ishermitian,
    norm,
)
from qalma.settings import (
    QALMA_ALLOW_OVERWRITE_BINDINGS,
)

from .utils import find_arithmetic_implementation


class Operator:  # pylint: disable=too-many-public-methods
    """Base class for operators"""

    system: SystemDescriptor
    prefactor: complex = 1.0

    # TODO check if it is possible implementing this
    # with multimethods
    __add__dispatch__: Dict[Tuple, Callable] = {}
    __mul__dispatch__: Dict[Tuple, Callable] = {}

    @staticmethod
    def register_add_handler(key: Tuple | List[Tuple]):
        """Register a function to implement add"""

        def register_func(func):
            if isinstance(key[0], (list, tuple)):
                keys = key
            else:
                keys = (key,)

            for curr_key in keys:
                if curr_key in Operator.__add__dispatch__:
                    if not QALMA_ALLOW_OVERWRITE_BINDINGS:
                        assert curr_key not in Operator.__add__dispatch__, (
                            f"{curr_key} already registered in "
                            f"{Operator.__add__dispatch__[curr_key].__code__}."
                        )
                # print(f"registering add operation for {curr_key} with {func} {func.__code__}")
                Operator.__add__dispatch__[curr_key] = func
            return func

        return register_func

    @staticmethod
    def register_mul_handler(key: Tuple | List[Tuple]):
        """Register a function to implement mul"""

        def register_func(func):
            if isinstance(key[0], (list, tuple)):
                keys = key
            else:
                keys = (key,)

            for curr_key in keys:
                if curr_key in Operator.__mul__dispatch__:
                    if not QALMA_ALLOW_OVERWRITE_BINDINGS:
                        assert curr_key not in Operator.__mul__dispatch__, (
                            f"{curr_key} already registered in "
                            f"{Operator.__mul__dispatch__[curr_key].__code__}."
                        )
                Operator.__mul__dispatch__[curr_key] = func
            return func

        return register_func

    def __bool__(self):
        return not self.is_zero

    def __add__(self, term):
        # Use multiple dispatch to determine how to add
        dispatch_table = Operator.__add__dispatch__
        # First try with the cases stored in the dispatch table:
        func = dispatch_table.get((type(self), type(term)), None)
        if func is not None:
            return func(self, term)

        func = dispatch_table.get((type(term), type(self)), None)
        if func is not None:
            return func(term, self)

        # Now, look for cases associated to the class hierarchy
        func = find_arithmetic_implementation(self, term, dispatch_table)
        if func:
            return func(self, term)
        func = find_arithmetic_implementation(term, self, dispatch_table)
        if func:
            return func(term, self)
        try:
            return term.__radd__(self)
        except TypeError as exc:
            raise TypeError(f"{type(self)} cannot be added with  {type(term)}") from exc

    def __mul__(self, factor):
        # Use multiple dispatch to determine how to multiply
        dispatch_table = Operator.__mul__dispatch__
        # First try with the cases stored in the dispatch table:
        func = dispatch_table.get((type(self), type(factor)), None)
        if func is not None:
            return func(self, factor)
        # Now, look for cases associated to the class hierarchy
        func = find_arithmetic_implementation(self, factor, dispatch_table)
        if func:
            return func(self, factor)

        try:
            return factor.__rmul__(self)
        except TypeError as exc:
            raise TypeError(
                f"{type(self)} cannot be multiplied with  {type(factor)}"
            ) from exc

    def __neg__(self):
        return -(self.to_qutip_operator())

    def __sub__(self, operand):
        if operand is None:
            raise ValueError("None can not be an operand")
        neg_op = -operand
        return self + neg_op

    def __radd__(self, term):
        # Use multiple dispatch to determine how to add
        dispatch_table = Operator.__add__dispatch__
        # First try with the cases stored in the dispatch table:
        func = dispatch_table.get(
            (
                type(term),
                type(self),
            ),
            None,
        )
        if func is not None:
            return func(term, self)
        # Now, look for cases associated to the class hierarchy
        func = find_arithmetic_implementation(term, self, dispatch_table)
        if func:
            return func(term, self)

        # Last chance: try in the opposite direction
        func = dispatch_table.get(
            (
                type(self),
                type(term),
            ),
            None,
        )
        if func is not None:
            return func(self, term)
        func = find_arithmetic_implementation(self, term, dispatch_table)
        if func:
            return func(self, term)

        raise TypeError(f"{type(self)} cannot be added with  {type(term)}")

    def __rmul__(self, factor):
        # Use __mul__dispatch__ to determine how to evaluate the product

        dispatch_table = Operator.__mul__dispatch__

        # First try with the cases stored in the dispatch table:
        func = dispatch_table.get((type(factor), type(self)), None)
        if func is not None:
            return func(factor, self)
        # Now, look for cases associated to the class hierarchy
        func = find_arithmetic_implementation(factor, self, dispatch_table)
        if func:
            return func(factor, self)

        raise TypeError(f"{type(factor)} cannot be multiplied with  {type(self)}")

    def __rsub__(self, operand):
        if operand is None:
            raise ValueError("None can not be an operand")

        neg_self = -self
        return operand + neg_self

    def __pow__(self, exponent):
        if exponent is None:
            raise ValueError("None can not be an operand")

        return self.to_qutip_operator() ** exponent

    def __truediv__(self, operand):
        if isinstance(operand, (int, float, complex)):
            return self * (1.0 / operand)
        if isinstance(operand, Operator):
            return self * operand.inv()
        raise ValueError("Division of an operator by ", type(operand), " not defined.")

    def _repr_latex_(self):
        """LaTeX Representation"""
        acts_over = sorted(self.acts_over())
        if len(acts_over) > 4:
            return repr(self)
        qutip_repr = self.to_qutip(tuple(acts_over))
        if isinstance(qutip_repr, Qobj):
            # pylint: disable=protected-access
            parts = qutip_repr._repr_latex_().replace("$$", "$").split("$")
            if len(parts) != 3:
                tex = "-?-"
            else:
                tex = parts[1]
        else:
            tex = str(qutip_repr)
        result = f"${tex}_" + "{" + ",".join(acts_over) + "}$"
        return result

    def acts_over(self) -> frozenset:
        """
        Return the list of sites over which the operator acts nontrivially.
        If this cannot be determined, return None.
        """
        raise NotImplementedError

    def as_sum_of_products(self):
        """Decompose an operator as a sum of product operators"""
        return self

    def dag(self):
        """Adjoint operator of quantum object"""
        return self.to_qutip_operator().dag()

    def flat(self):
        """simplifies sums and products"""
        return self

    def hermitician_part(self):
        """The hermitician part of the operator"""
        if self.isherm:
            return self
        return (self + self.dag()) * 0.5

    @property
    def isherm(self) -> bool:
        """Check if the operator is hermitician"""
        return self.to_qutip(tuple()).tidyup().isherm

    @property
    def isdiagonal(self) -> bool:
        """Check if the operator is diagonal"""
        return False

    @property
    def is_zero(self) -> bool:
        """True if self is a null operator"""
        return empty_op(self)

    def eigenenergies(self):
        """List of eigenstates of the operator"""
        return self.to_qutip_operator().eigenenergies()

    def eigenstates(self):
        """List of eigenstates of the operator"""
        return self.to_qutip_operator().eigenstates()

    def expm(self) -> "Operator":
        """
        Compute the exponential of the Qutip representation of the operator
        """

        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        # type: ignore[import-untyped]
        from scipy.sparse.linalg import ArpackError

        from qalma.operators.functions import eigenvalues
        from qalma.operators.qutip import QutipOperator

        op_qutip = self.to_qutip()
        try:
            max_eval = eigenvalues(op_qutip, sort="high", sparse=True, eigvals=3)[0]
        except ArpackError:
            max_eval = max(op_qutip.diag())

        op_qutip = (op_qutip - max_eval).expm()
        return QutipOperator(op_qutip, self.system, prefactor=np.exp(max_eval))

    def inv(self) -> "Operator":
        """the inverse of the operator"""
        return self.to_qutip_operator().inv()

    def logm(self) -> "Operator":
        """Logarithm of the operator"""
        return self.to_qutip_operator().logm()

    def n_body_sector(self) -> int:
        """
        The maximum number of factors of any term in
        a product state decomposition.
        """
        return len(self.acts_over())

    def num_terms(self) -> int:
        """Number of terms that spans the operator"""
        return 1

    def norm(self, ord: Optional[int | str | float] = None):
        """The norm of the operator"""

        return norm(self.to_qutip(), ord)

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        """Partial trace over sites not listed in `sites`"""
        raise NotImplementedError

    def reduce(self, sites: Iterable, state=None):
        """
        Partial trace of the product of the operator and the density operator
        acting on the subsystem which is traced out.
        If the state is not provided, the result is the partial trace, divided
        by the dimension of the subsystem traced out.

        Parameters
        ==========
        sites: Iterable

        state: Optional[DensityOperatorProtocol]
               The state relative to which make the reduction.

        Return
        ======

        The reduced operator.

        """
        raise NotImplementedError

    def _set_system_(self, system=None):
        """
        Change the system associated to the operator,
        and references of other operators inside.

        In a multiprocess context, the `system` attribute of
        the objects generated by the children process lost
        their identity regarding the `system` attribute
        of the committed object.
        To get the right reference on the returned objects,
        call this method without parameters in the worker,
        before returning the objects.
        Then, in the main process, set back the original system
        object.
        """
        self.system = system
        return self

    def simplify(self) -> "Operator":
        """Returns a more efficient representation"""
        return self

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """Convert to a Qutip object"""
        raise NotImplementedError

    def to_qutip_operator(self):
        """Produce a Qutip representation of the operator"""
        # pylint: disable=import-outside-toplevel

        block = tuple(sorted(self.acts_over()))
        if len(block) == 0:
            return self
        site_names = {site: i for i, site in enumerate(block)}
        qobj = self.to_qutip(block)
        if isinstance(qobj, Qobj):
            from .qutip import QutipOperator

            assert qobj.type != "scalar"
            return QutipOperator(qobj, system=self.system, names=site_names)

        from .product import ScalarOperator

        return ScalarOperator(qobj, self.system)

    # pylint: disable=invalid-name
    def tr(self) -> complex:
        """The trace of the operator"""
        return self.partial_trace(frozenset()).prefactor

    def tidyup(self, _atol=None):
        """remove tiny elements of the operator"""
        return self


class LocalOperator(Operator):
    """
    Operator acting over a single site.
    """

    operator: np.ndarray
    site: str

    def __init__(
        self,
        site: str,
        local_operator,
        system: Optional[SystemDescriptor] = None,
    ):
        assert isinstance(site, str)
        assert system is not None
        self.site = site
        if isinstance(local_operator, (int, float, complex)):
            local_operator = system.site_identity(site) * local_operator

        self.operator = _to_array(local_operator)
        self.system = system

    def __bool__(self):
        return bool(self.operator.any())

    def __neg__(self):
        return LocalOperator(self.site, -self.operator, self.system)

    def __pow__(self, exp):
        operator = self.operator_qutip
        if exp < 0 and hasattr(operator, "inv"):
            operator = operator.inv()
            exp = -exp

        return LocalOperator(self.site, operator**exp, self.system)

    def __repr__(self):
        return f"Local Operator on site {self.site}:" f"\n {repr(self.operator_qutip)}"

    @cached_property
    def operator_qutip(self) -> Qobj:
        """Return a Qutip representation of the local operator"""
        op = self.operator.copy()
        op[np.abs(op) < 1e-12] = 0
        return Qobj(op, copy=False).to("CSR")

    def acts_over(self) -> frozenset:
        return frozenset((self.site,))

    def dag(self):
        """
        Return the adjoint operator
        """
        operator = self.operator
        if self.isherm:
            return self
        return LocalOperator(self.site, operator.T.conj(), self.system)

    def expm(self):
        return LocalOperator(self.site, self.operator_qutip.expm(), self.system)

    def hermitician_part(self):
        """The hermitician part of the operator"""
        op = self.operator
        if self.isherm:
            return self
        op = (op + op.T.conj()) * 0.5
        return LocalOperator(self.site, op, self.system)

    def inv(self):
        operator = self.operator_qutip
        system = self.system
        site = self.site
        return LocalOperator(
            site,
            operator.inv() if hasattr(operator, "inv") else 1 / operator,
            system,
        )

    @cached_property
    def isherm(self) -> bool:
        return ishermitian(self.operator)

    @cached_property
    def isdiagonal(self) -> bool:
        return is_diagonal_op(self.operator)

    def logm(self):
        def log_qutip(loc_op):
            evals, evecs = loc_op.eigenstates()
            evals[abs(evals) < 1.0e-50] = 1.0e-50
            return sum(
                np.log(e_val) * e_vec * e_vec.dag()
                for e_val, e_vec in zip(evals, evecs)
            )

        return LocalOperator(self.site, log_qutip(self.operator_qutip), self.system)

    def norm(self, ord=None):
        """The norm of the operator"""

        result = norm(self.operator, ord)
        if ord in ("fro", "nuc"):
            dim_factor = 1.0
            for dim in (
                dim for site, dim in self.system.dimensions.items() if site != self.site
            ):
                dim_factor *= dim
            if ord == "fro":
                result *= dim_factor**0.5
            else:
                result *= dim_factor

        return result

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        # pylint: disable=import-outside-toplevel

        system = self.system
        dimensions = system.dimensions
        subsystem = (
            sites if isinstance(sites, SystemDescriptor) else system.subsystem(sites)
        )
        local_sites = subsystem.sites
        site = self.site
        prefactors = [
            d for s, d in dimensions.items() if s != site and s not in local_sites
        ]

        if len(prefactors) > 0:
            prefactor = reduce(lambda x, y: x * y, prefactors)
        else:
            prefactor = 1

        local_op = self.operator
        if site not in local_sites:
            from .product import ScalarOperator

            return ScalarOperator(local_op.trace() * prefactor, subsystem)
        return LocalOperator(site, local_op * prefactor, subsystem)

    def reduce(self, sites: Iterable, state=None) -> Operator:
        """
        Partial trace of the product of the operator and the density operator
        acting on the subsystem which is traced out.
        If the state is not provided, the result is the partial trace, divided
        by the dimension of the subsystem traced out.

        Parameters
        ==========
        sites: Iterable

        state: Optional[DensityOperatorProtocol]
               The state relative to which make the reduction.

        Return
        ======

        The reduced operator.

        """
        # pylint: disable=import-outside-toplevel

        scalar_val: complex
        site = self.site
        if site in sites:
            return self
        system = self.system
        if state is not None:
            scalar_val = state.expect(self)
        else:
            scalar_val = self.operator.trace() / system.dimensions[site]

        from .product import ScalarOperator

        return ScalarOperator(scalar_val, system)

    def simplify(self):
        # TODO: reduce multiples of the identity to ScalarOperators
        # pylint: disable=import-outside-toplevel
        operator = self.operator
        if not is_scalar_op(operator):
            return self
        value = operator[0, 0] * self.prefactor

        from .product import ScalarOperator

        return ScalarOperator(value, self.system)

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """Convert to a Qutip object"""
        site = self.site
        system = self.system
        sites = system.sites
        dimensions = system.dimensions
        operator = self.operator
        # Ensure that block at least contains site
        if block is None:
            block = tuple(sorted(sites))
            if len(block) > 8:
                logging.warning(
                    "Asking for a qutip representation of an operator over the full system"
                )
        elif site not in block:
            block = block + (site,)
        # Ensure that operator is a qutip operator
        if isinstance(operator, (int, float, complex)):
            operator = qutip.qeye(dimensions[site]) * operator
        elif isinstance(operator, Operator):
            operator = operator.to_qutip((site,))
        else:
            operator = self.operator_qutip
        # Build factors
        factors_dict = (operator if s == site else sites[s]["identity"] for s in block)
        return qutip.tensor(*factors_dict)

    def tr(self):
        result = self.partial_trace(frozenset())
        return result.prefactor

    def tidyup(self, atol=None):
        """remove tiny elements of the operator"""
        return LocalOperator(self.site, self.operator_qutip.tidyup(atol), self.system)
