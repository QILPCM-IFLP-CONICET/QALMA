"""
Different representations for operators
"""

import logging
from functools import cached_property, reduce

# from types import MappingProxyType
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
from qutip import Qobj

from qalma.model import SystemDescriptor
from qalma.qutip_tools.tools import (
    _to_array,
    empty_op,
    fast_tensor,
    is_diagonal_op,
    is_scalar_op,
    ishermitian,
    norm,
    to_qobj,
)
from qalma.settings import (
    QALMA_TOLERANCE,
)

from .basic import LocalOperator, Operator

# from scipy.linalg import ishermitian


class ProductOperator(Operator):
    """Product of operators acting over different sites"""

    _to_qutip_cache: Dict[Optional[Tuple[str, ...]], Qobj]
    prefactor: complex
    site_factors: Dict[str, np.ndarray]
    system: SystemDescriptor

    def __init__(
        self,
        sites_operators: dict,
        prefactor: complex = 1.0,
        system: Optional[SystemDescriptor] = None,
    ):
        assert system is not None
        remove_numbers = False
        for site, local_op in sites_operators.items():
            if isinstance(local_op, (int, float, complex)):
                prefactor *= local_op
                remove_numbers = True

        if remove_numbers:
            sites_operators = {
                s: local_op
                for s, local_op in sites_operators.items()
                if not isinstance(local_op, (int, float, complex))
            }

        if all(isinstance(value, Qobj) for value in sites_operators.values()):
            self.__dict__["site_factors_qutip"] = sites_operators

        sites_operators = {key: _to_array(op) for key, op in sites_operators.items()}
        self.site_factors = sites_operators
        if any(empty_op(op) for op in sites_operators.values()):
            prefactor = 0
            self.site_factors = {}
        self.prefactor = prefactor
        assert isinstance(prefactor, (int, float, complex)), f"{type(prefactor)}"
        self.system = system
        if system is not None:
            self.size = len(system.sites)
            self.dimensions = {
                name: site["dimension"] for name, site in system.sites.items()
            }
        self._to_qutip_cache = {}

    @cached_property
    def site_factors_qutip(self) -> Dict[str, Qobj]:
        return {key: to_qobj(op.copy()) for key, op in self.site_factors.items()}

    @cached_property
    def _dense_tensor(self):
        """
        Stacked dense representation for *homogeneous* systems, i.e. where
        every site has the same local Hilbert-space dimension d.

        Returns ``(sites, tensor)`` where:
          - ``sites``  is a sorted tuple of site names matching axis-0, and
          - ``tensor`` is a complex128 ndarray of shape ``(N, d, d)``.

        Raises ``ValueError`` for heterogeneous systems; callers should
        catch it and fall back to iterating over ``_dense``.
        """
        if not self.site_factors:
            return (), np.empty((0,), dtype=np.complex128)
        dense = self.site_factors
        sites = tuple(sorted(dense))
        shapes = {dense[s].shape for s in sites}
        if len(shapes) > 1:
            raise ValueError(
                "ProductOperator._dense_tensor: heterogeneous site dimensions "
                f"{shapes}; use _dense instead."
            )
        return sites, np.stack([dense[s] for s in sites])  # (N, d, d)

    @staticmethod
    def _trace2(a: np.ndarray, b: np.ndarray) -> complex:
        """
        Tr(a @ b) without allocating an intermediate matrix.

        Equivalent to ``np.einsum('ij,ji->', a, b)`` but avoids einsum's
        fixed Python overhead, which dominates for the small matrices (d=2,3,4)
        typical in spin/boson lattice models.
        """
        return complex((a * b.T).sum())

    def __bool__(self):
        return bool(self.prefactor) and all(
            factor.any() for factor in self.site_factors.values()
        )

    def __neg__(self):
        return ProductOperator(self.site_factors, -self.prefactor, self.system)

    def __pow__(self, exp):
        return ProductOperator(
            {s: op**exp for s, op in self.site_factors_qutip.items()},
            self.prefactor**exp,
            self.system,
        )

    def __repr__(self):
        result = "  " + str(self.prefactor) + " * (\n  "
        result += "  (x)\n  ".join(
            f"({item[1].full()} <-  {item[0]})"
            for item in sorted(self.site_factors_qutip.items(), key=lambda x: x[0])
        )
        result += "\n   )"
        return result

    def _repr_latex(self):
        """latex representation"""
        factors_latex = []
        for site, qutip_op in self.site_factors_qutip.items():
            # pylint: disable=protected-access
            tex = qutip_op._repr_latex_().replace("$$", "$")
            parts = tex.split("$")
            if len(parts) == 3:
                tex = parts[1]
            else:
                tex = "-?-"

            prefactor = self.prefactor
            if prefactor == 1:
                factors_latex.append(tex + "_{" + site + "}")
            elif prefactor < 0:
                factors_latex.append(f"({prefactor}) *" + tex + "_{" + site + "}")
            else:
                factors_latex.append(f"{prefactor} *" + tex + "_{" + site + "}")
        return "$" + "\\otimes".join(factors_latex) + "$"

    def acts_over(self) -> frozenset:
        return frozenset(site for site in self.site_factors)

    def dag(self):
        """
        Return the adjoint operator
        """
        sites_op_dag = {key: op.T.conj() for key, op in self.site_factors.items()}
        prefactor = self.prefactor
        if isinstance(prefactor, complex):
            prefactor = prefactor.conjugate()
        return ProductOperator(sites_op_dag, prefactor, self.system)

    def expm(self):
        sites_op = self.site_factors_qutip
        n_ops = len(sites_op)
        if n_ops == 0:
            return ScalarOperator(np.exp(self.prefactor), self.system)
        if n_ops == 1:
            site, operator = next(iter(sites_op.items()))
            result = LocalOperator(
                site, (self.prefactor * operator).expm(), self.system
            )
            return result
        result = super().expm()
        return result

    def flat(self):
        nfactors = len(self.site_factors)
        if nfactors == 0:
            return ScalarOperator(self.prefactor, self.system)
        if nfactors == 1:
            name, op_factor = list(self.site_factors_qutip.items())[0]
            return LocalOperator(name, self.prefactor * op_factor, self.system)
        return self

    def hermitician_part(self):
        from qalma.operators import SumOperator

        if self.isherm:
            return self
        if all(ishermitian(op) for op in self.site_factors.values()):
            return ProductOperator(
                self.site_factors, np.real(self.prefactor), self.system
            )
        half_self = self * 0.5
        return SumOperator(
            (half_self, half_self.dag()), system=self.system, isherm=True
        )

    def inv(self):
        sites_op = self.site_factors_qutip
        system = self.system
        prefactor = self.prefactor

        n_ops = len(sites_op)
        sites_op = {site: op_local.inv() for site, op_local in sites_op.items()}
        if n_ops == 1:
            site, op_local = next(iter(sites_op.items()))
            return LocalOperator(site, op_local / prefactor, system)
        return ProductOperator(sites_op, 1 / prefactor, system)

    @cached_property
    def isherm(self) -> bool:
        # TODO: check if it worth to check that factors are not hermitician
        # up to a phase factor.
        if not all(ishermitian(loc_op) for loc_op in self.site_factors.values()):
            return False
        prefactor = self.prefactor
        if isinstance(prefactor, (int, float, np.float64)):
            return True
        if isinstance(prefactor, (complex, np.complex128)):
            return abs(prefactor.imag) < QALMA_TOLERANCE
        return False

    @cached_property
    def isdiagonal(self) -> bool:
        for factor_op in self.site_factors.values():
            if not is_diagonal_op(factor_op):
                return False
        return True

    def logm(self):
        # pylint: disable=import-outside-toplevel
        from qalma.operators.arithmetic import OneBodyOperator

        def log_qutip(loc_op):
            evals, evecs = loc_op.eigenstates()
            evals[abs(evals) < 1.0e-30] = 1.0e-30
            return sum(
                np.log(e_val) * e_vec * e_vec.dag()
                for e_val, e_vec in zip(evals, evecs)
            )

        system = self.system
        terms = tuple(
            LocalOperator(site, log_qutip(loc_op), system)
            for site, loc_op in self.site_factors_qutip.items()
        )
        result = OneBodyOperator(terms, system, False)
        result = result + ScalarOperator(np.log(self.prefactor), system)
        return result

    def norm(self, ord=None):
        """The norm of the operator"""

        result = self.prefactor
        for op_loc in self.site_factors_qutip.values():
            result *= norm(op_loc, ord)

        if ord in ("fro", "nuc"):
            dim_factor = 1.0
            for dim in (
                dim
                for site, dim in self.system.dimensions.items()
                if site not in self.site_factors
            ):
                dim_factor *= dim
            if ord == "fro":
                result *= dim_factor**0.5
            else:
                result *= dim_factor

        return result

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        full_system_sites = self.system.sites
        dimensions = self.dimensions
        if isinstance(sites, SystemDescriptor):
            subsystem = sites
            sites = frozenset(sites.sites.keys())
        else:
            subsystem = self.system.subsystem(sites)

        sites_out = tuple(s for s in full_system_sites if s not in sites)
        sites_op = self.site_factors
        prefactors = [
            sites_op[s].trace() if s in sites_op else dimensions[s] for s in sites_out
        ]
        sites_op = {s: o for s, o in sites_op.items() if s in sites}
        prefactor = self.prefactor
        for factor in prefactors:
            if factor == 0:
                return ScalarOperator(factor, subsystem)
            prefactor *= factor

        if len(sites_op) == 0:
            return ScalarOperator(prefactor, subsystem)
        return ProductOperator(sites_op, prefactor, subsystem)

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
        acts_over = self.acts_over()
        prefactor = self.prefactor
        sites = acts_over.intersection(sites)
        environment = acts_over - sites
        if not environment:
            return self
        system = self.system
        if not sites:
            if state is None:
                value = self.tr()
                dimensions = system.dimensions
                value /= reduce(
                    lambda x, y: x * y, (dimensions[site] for site in acts_over)
                )
                return ScalarOperator(value, system)
            return ScalarOperator(state.expect(self), system)

        system = self.system
        # Special cases:
        if state is None:
            dimensions = self.system.dimensions
            sites_op = self.site_factors

            for site in environment:
                prefactor *= sites_op[site].trace() / dimensions[site]
            return ProductOperator(
                {site: sites_op[site] for site in sites}, prefactor, system
            )
        # ProductDensityOperator:
        if hasattr(state, "terms"):
            return self.to_qutip_operator().reduce(sites, state)

        if hasattr(state, "to_product_state"):
            state = state.to_product_state()
        if isinstance(state, ProductOperator):
            state_by_site = state.site_factors
            sites_op = self.site_factors
            for site in environment:
                prefactor *= (sites_op[site] @ state_by_site[site]).trace()
            result = ProductOperator(
                {site: sites_op[site] for site in sites}, prefactor, system
            )
        else:
            # General case:
            env_tuple = tuple(environment)
            state = state.partial_trace(environment).to_qutip(env_tuple)
            sites_ops = self.site_factors
            # TODO: check if we can do more using numpy
            prefactor *= (
                state
                * fast_tensor(*(self.site_factors_qutip[site] for site in env_tuple))
            ).tr()
            sites_op = {site: op_q for site, op_q in sites_ops.items() if site in sites}
            result = ProductOperator(sites_op, prefactor, system)
        return result

    def simplify(self) -> Operator:
        """
        Simplifies a product operator
           - first, collect all the scalar factors and
             absorbe them in the prefactor.
           - If the prefactor vanishes, or all the factors are scalars,
             return a ScalarOperator.
           - If there is just one nontrivial factor, return a LocalOperator.
           - If no reduction is possible, return self.
        """
        # Remove multiples of the identity
        nontrivial_factors = {}
        prefactor = self.prefactor
        if prefactor == 0:
            return ScalarOperator(0, self.system)
        for site, op_factor in self.site_factors.items():
            if is_scalar_op(op_factor):
                prefactor *= op_factor[0, 0]
                assert isinstance(
                    prefactor, (int, float, complex)
                ), f"{type(prefactor)}:{prefactor}"
                if not prefactor:
                    return ScalarOperator(0, self.system)
            else:
                nontrivial_factors[site] = op_factor
        nops = len(nontrivial_factors)
        if nops == 0:
            return ScalarOperator(prefactor, self.system)
        if nops == 1:
            site, op_local = next(iter(nontrivial_factors.items()))
            return LocalOperator(site, to_qobj(op_local * prefactor), self.system)
        if nops != len(self.site_factors):
            return ProductOperator(nontrivial_factors, prefactor, self.system)
        return self

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """
        return a qutip object acting over the sites listed in
        `block`.
        By default (`block=None`) returns a qutip object
        acting over all the sites, in lexicographical order.
        """
        cached = self._to_qutip_cache.get(block, None)
        if cached is not None:
            return cached

        sites_op = self.site_factors_qutip
        system = self.system
        sites = system.sites if system else {}
        # Ensure that block has the sites in the operator.
        orig_block = block
        if block is None:
            if system is not None:
                block = tuple(sorted(sites))
            else:
                block = tuple(sorted(self.acts_over()))

            if len(block) > 8:
                logging.warning(
                    "Asking for a qutip representation of an operator over the full system"
                )

        else:
            block = tuple((site for site in block if site in sites)) + tuple(
                sorted(site for site in sites_op if site not in block)
            )
        if len(block) == 0:
            return self.prefactor

        factors = (
            (sites_op[site] if site in sites_op else sites[site]["identity"])
            for site in block
        )
        self._to_qutip_cache[orig_block] = result = self.prefactor * fast_tensor(
            *factors
        )
        return result

    def to_qutip_operator(self) -> Operator:
        """
        Return a QutipOperator representation.
        If the operator is scalar, returns a ScalarOperator.
        Otherwise, returns a QutipOperator.
        """
        prefactor = self.prefactor
        if not (prefactor and self.site_factors_qutip):
            return ScalarOperator(prefactor, self.system)
        return super().to_qutip_operator()

    def tr(self):
        result = self.partial_trace(frozenset())
        return result.prefactor

    def tidyup(self, atol=None):
        """remove tiny elements of the operator"""
        tidy_site_operators = {
            name: op_s.tidyup(atol) for name, op_s in self.site_factors_qutip.items()
        }
        return ProductOperator(tidy_site_operators, self.prefactor, self.system)


class ScalarOperator(ProductOperator):
    """A product operator that acts trivially on every subsystem"""

    def __init__(self, prefactor, system):
        assert system is not None
        super().__init__({}, prefactor, system)

    def __bool__(self):
        return bool(self.prefactor)

    def __neg__(self):
        return ScalarOperator(-self.prefactor, self.system)

    def __repr__(self):
        result = (
            str(self.prefactor) + " * Identity_{" + ",".join(self.system.sites) + "} "
        )

        return result

    def _repr_latex_(self):

        return (
            "$\\left("
            + str(self.prefactor)
            + " \\times \\mathbb{I}\\right)_{"
            + ",".join(self.system.sites)
            + "}$"
        )

    def acts_over(self) -> frozenset:
        return frozenset()

    def dag(self):
        if isinstance(self.prefactor, complex):
            return ScalarOperator(self.prefactor.conjugate(), self.system)
        return self

    def hermitician_part(self):
        if self.isherm:
            return self
        return ScalarOperator(np.real(self.prefactor), self.system)

    @property
    def isherm(self):
        prefactor = self.prefactor
        return not (
            isinstance(prefactor, complex) and abs(prefactor.imag) > QALMA_TOLERANCE
        )

    @property
    def isdiagonal(self) -> bool:
        return True

    def logm(self):
        return ScalarOperator(np.log(self.prefactor), self.system)

    def norm(self, ord=None):
        """The norm of the operator"""

        result = self.prefactor
        if ord in ("fro", "nuc"):
            dim_factor = 1.0
            for dim in (dim for site, dim in self.system.dimensions.items()):
                dim_factor *= dim
            if ord == "fro":
                result *= dim_factor**0.5
            else:
                result *= dim_factor

        return result

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
        return self

    def simplify(self):
        """simplify a scalar operator"""
        return self

    def tidyup(self, atol=None):
        if atol is None:
            atol = QALMA_TOLERANCE
        if abs(self.prefactor) < atol:
            return ScalarOperator(0, self.system)
        return self

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """
        return a qutip object acting over the sites listed in
        `block`.
        By default (`block=None`) returns a qutip object
        acting over all the sites, in lexicographical order.
        """
        system = self.system
        sites = system.sites
        if block is None:
            block = tuple(sorted(sites))
        elif len(block) == 0:
            return self.prefactor

        factors = (sites[site]["identity"] for site in block)
        return self.prefactor * fast_tensor(*factors)

    def to_qutip_operator(self):
        """
        Produce a Qutip representation of the operator.
        For ScalarOperators, just return self.
        """
        return self
