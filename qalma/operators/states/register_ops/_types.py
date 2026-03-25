"""
Shared operator type tuples used across all register_ops submodules.
"""

from numbers import Number

import numpy as np

from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import LocalOperator, Operator
from qalma.operators.product import ProductOperator, ScalarOperator
from qalma.operators.quadratic import QuadraticFormOperator
from qalma.operators.qutip import QutipOperator
from qalma.operators.states.basic import (
    DensityOperatorMixin,
)
from qalma.operators.states.gibbs import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
)
from qalma.operators.states.product import (
    ProductDensityOperator,
)
from qalma.operators.states.qutip import QutipDensityOperator

BASIC_OPERATOR_TYPES = (
    Operator,
    ScalarOperator,
    LocalOperator,
    ProductOperator,
    QutipOperator,
    OneBodyOperator,
    QuadraticFormOperator,
)

NON_PRODUCT_BASIC_OPERATOR_TYPES = (
    Operator,
    QutipOperator,
    OneBodyOperator,
    QuadraticFormOperator,
)

DENSITY_OPERATOR_BASIC_TYPES = (
    DensityOperatorMixin,
    ProductDensityOperator,
    QutipDensityOperator,
)

NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES = (
    DensityOperatorMixin,
    QutipDensityOperator,
    GibbsDensityOperator,
)

ANY_OPERATOR_TYPES = (
    DensityOperatorMixin,
    GibbsDensityOperator,
    GibbsProductDensityOperator,
    LocalOperator,
    OneBodyOperator,
    Operator,
    ProductDensityOperator,
    ProductOperator,
    QuadraticFormOperator,
    QutipDensityOperator,
    QutipOperator,
    ScalarOperator,
    SumOperator,
)

REAL_NUMERIC_TYPES = tuple((int, float, np.float64))
COMPLEX_NUMERIC_TYPES = tuple((Number, complex, np.complex128))
TYPES_WITH_PREFACTOR = (ScalarOperator, ProductOperator, QutipOperator)
SUM_TYPES = (SumOperator, OneBodyOperator)
