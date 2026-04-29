"""Wrappers for converting DensityOperators into basic operators."""

from typing import Any, Callable, Dict

from qalma.operators.arithmetic import SumOperator
from qalma.operators.basic import Operator
from qalma.operators.product import ProductOperator
from qalma.operators.qutip import QutipOperator
from qalma.operators.states.arithmetic import MixtureDensityOperator
from qalma.operators.states.gibbs import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
)
from qalma.operators.states.product import ProductDensityOperator
from qalma.operators.states.qutip import QutipDensityOperator

WRAPPERS_BY_TYPE: Dict[Any, Callable] = {}


def _wrapper_generic(rho, prefactor: complex = 1) -> Operator:
    """Generic wrapper."""
    try:
        return WRAPPERS_BY_TYPE[type(rho)](rho, prefactor)
    except KeyError:
        if hasattr(rho, "to_product_state"):
            return _wrapper_gibbs_product(rho, prefactor)
        return _wrapper_gibbs(rho, prefactor)


def _wrapper_sum(op, prefactor: complex = 1) -> SumOperator:
    """Convert a MixtureDensityOperator into a SumOperator."""
    return SumOperator(
        tuple(
            _wrapper_generic(term, term.prefactor * prefactor)
            for term in op.terms
            if term.prefactor
        ),
        system=op.system,
        isherm=True,
    )


def _wrapper_qutip(
    qutip_density: QutipDensityOperator, prefactor: complex = 1
) -> QutipOperator:
    """Discard the prefactor and build a new qutip operator."""
    result = QutipOperator(
        qutip_density.operator,
        qutip_density.system,
        qutip_density.site_names,
        prefactor=prefactor,
    )
    return result


def _wrapper_product(
    y_op: ProductDensityOperator, prefactor: complex = 1
) -> ProductOperator:
    """Convert a ProductDensityOperator into a ProductOperator Missing factors
    in  ProductDensityOperator are not treated as the identity operator, but as
    a prefactor 1/dim_local.
    """
    return ProductOperator(y_op.site_factors, prefactor=prefactor, system=y_op.system)


def _wrapper_gibbs(
    operator: GibbsDensityOperator, prefactor: complex = 1
) -> QutipOperator:
    """Convert a GibbsOperator into a basic QutioOperator."""
    return _wrapper_qutip(operator.to_qutip_operator(), prefactor)


def _wrapper_gibbs_product(
    operator: GibbsProductDensityOperator, prefactor: complex = 1
) -> ProductOperator:
    """Convert a GibbsOperator into a basic ProductOperator."""
    return _wrapper_product(operator.to_product_state(), prefactor)


WRAPPERS_BY_TYPE.update(
    {
        QutipDensityOperator: _wrapper_qutip,
        MixtureDensityOperator: _wrapper_sum,
        GibbsProductDensityOperator: _wrapper_gibbs_product,
        GibbsDensityOperator: _wrapper_gibbs,
        ProductDensityOperator: _wrapper_product,
    }
)
