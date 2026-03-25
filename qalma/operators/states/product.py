"""
Density operator classes.
"""

import logging
from typing import Any, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from qutip import (  # type: ignore[import-untyped]
    qeye as qutip_qeye,
    tensor as qutip_tensor,
)

from qalma.model import SystemDescriptor
from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import (
    LocalOperator,
)
from qalma.operators.product import (
    ProductOperator,
    ScalarOperator,
)
from qalma.operators.states.basic import DensityOperatorMixin
from qalma.qutip_tools.tools import (
    _to_array,
)


class ProductDensityOperator(DensityOperatorMixin, ProductOperator):
    """An uncorrelated density operator."""

    prefactor: complex  # must be float

    def __init__(
        self,
        local_states: dict,
        weight: float = 1.0,
        system: Optional[SystemDescriptor] = None,
        normalized: bool = False,
    ):
        assert weight >= 0

        # Build the local partition functions and normalize
        # if required
        if weight == 0:
            local_states = {}
            local_zs = {}
        else:
            local_states = {key: _to_array(val) for key, val in local_states.items()}
            local_zs = {site: state.trace() for site, state in local_states.items()}
            if not normalized:
                assert (z > 0 for z in local_zs.values())
                local_states = {
                    site: sigma / local_zs[site] for site, sigma in local_states.items()
                }

        # Complete the scalar factors using the system
        if system is None:
            dimensions = {
                site: operator.data.shape[0] for site, operator in local_states.items()
            }
            # TODO: build a system
        else:
            dimensions = system.dimensions
            local_identities: dict = {}
            for site, dimension in dimensions.items():
                if site not in local_states:
                    local_id = local_identities.get(dimension, None)
                    local_zs[site] = dimension
                    if local_id is None:
                        local_id = qutip_qeye(dimension) / dimension
                        local_identities[dimension] = local_id
                    local_states[site] = local_id

        super().__init__(local_states, prefactor=weight, system=system)
        self.local_fs = {site: -np.log(z) for site, z in local_zs.items()}

    def __mul__(self, a):
        if isinstance(a, (float, np.float64)):
            if a >= 0:
                return ProductDensityOperator(
                    self.site_factors, self.prefactor * a, self.system, False
                )
            logging.warning(
                (
                    "Multiplication of a non positive number by a "
                    "density operator returns a regular operator."
                )
            )
            return ProductOperator(self.site_factors, 1, self.system) * a
        return ProductOperator(self.site_factors, 1, self.system) * a

    def __neg__(self):
        logging.warning("Negate a DensityOperator leads to a regular operator.")
        return ProductOperator(self.site_factors, -1, self.system)

    def __rmul__(self, a):
        if isinstance(a, (float, np.float64)):
            if a >= 0:
                return ProductDensityOperator(
                    self.site_factors, self.prefactor * a, self.system, False
                )
            logging.warning(
                (
                    "Multiplication of a non positive number by "
                    "a density operator returns a regular operator."
                )
            )
            return ProductOperator(self.site_factors, 1, self.system) * a
        return a * ProductOperator(self.site_factors, 1, self.system)

    def expect(self: Any, obs_objs: Any) -> Any:
        """
        Compute the expectation value of an operator or a sequence of
        operators.

        Hot paths use dense numpy arithmetic and bypass Qobj overhead:

        * LocalOperator  -> single _trace2 call (no Qobj allocation)
        * ProductOperator, homogeneous system -> batched einsum over a
          stacked (N, d, d) tensor
        * ProductOperator, heterogeneous system -> per-site _trace2 loop
        * Everything else -> delegate to the parent DensityOperatorMixin
        """
        if isinstance(obs_objs, LocalOperator):
            site = obs_objs.site
            op_dense = obs_objs.operator
            return self._trace2(self.site_factors[site], op_dense)

        if isinstance(obs_objs, ProductOperator):
            obs_prod = cast(ProductOperator, obs_objs)
            result: complex = complex(obs_prod.prefactor)
            if not result:
                return complex(0)

            rhos = self.site_factors  # dict[site -> (d,d)]

            # --- Fast path: homogeneous system, batched einsum -----------
            try:
                obs_sites, obs_tensor = obs_prod._dense_tensor  # (N, d, d)
                rho_tensor = np.stack([rhos[s] for s in obs_sites])  # (N, d, d)
                # traces[i] = Tr(rho_i @ obs_i), no intermediate matrices
                traces = np.einsum("nij,nji->n", rho_tensor, obs_tensor)
                result *= complex(traces.prod())
            except (ValueError, KeyError):
                # Heterogeneous dims or a site not in rhos: fall back to
                # a per-site loop that is still numpy-only (no Qobj).
                for site, obs_op in obs_prod.site_factors.items():
                    if not result:
                        break
                    result *= self._trace2(rhos[site], obs_op)

            return result

        if isinstance(obs_objs, SumOperator):
            obs_sum = cast(SumOperator, obs_objs)
            return cast(
                NDArray,
                sum(cast(NDArray, self.expect(term)) for term in obs_sum.terms),
            )

        if isinstance(obs_objs, (tuple, list)):
            return np.array([self.expect(elem) for elem in obs_objs])

        if isinstance(obs_objs, dict):
            return {key: self.expect(val) for key, val in obs_objs.items()}

        # Fallback for QuadraticFormOperator and any other type
        return super().expect(obs_objs)

    def logm(self):
        def log_qutip(loc_op):
            evals, evecs = loc_op.eigenstates()
            evals[abs(evals) < 1.0e-30] = 1.0e-30
            return sum(
                np.log(e_val) * e_vec * e_vec.dag()
                for e_val, e_vec in zip(evals, evecs)
            )

        system = self.system
        sites_op = self.site_factors_qutip
        terms = tuple(
            LocalOperator(site, log_qutip(loc_op), system)
            for site, loc_op in sites_op.items()
        )
        if system:
            norm = -sum(
                np.log(dim)
                for site, dim in system.dimensions.items()
                if site not in sites_op
            )
            return OneBodyOperator(terms, system, False) + ScalarOperator(norm, system)
        return OneBodyOperator(terms, system, False)

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        sites_op = self.site_factors_qutip
        if isinstance(sites, SystemDescriptor):
            subsystem = sites
            sites = frozenset(sites.sites.keys())
        else:
            subsystem = self.system.subsystem(sites)

        local_states = {site: sites_op[site] for site in sites}

        return ProductDensityOperator(
            local_states, np.real(self.prefactor), subsystem, normalized=True
        )

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        prefactor = self.prefactor
        if prefactor == 0 or len(self.system.dimensions) == 0:
            return np.exp(-sum(np.log(dim) for dim in self.system.dimensions.values()))

        sites_op = self.site_factors_qutip
        dimensions = self.system.dimensions
        if block is None:
            block = tuple(sorted(self.system.sites))
        else:
            block = block + tuple(
                (site for site in sorted(sites_op) if site not in block)
            )

        return qutip_tensor(
            [
                (
                    sites_op[site]
                    if site in sites_op
                    else qutip_qeye(dimensions[site]) / dimensions[site]
                )
                for site in block
            ]
        )
