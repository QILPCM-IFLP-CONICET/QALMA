"""Shared operator type tuples used across all register_ops submodules."""

from numbers import Number

import numpy as np

from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import LocalOperator, Operator
from qalma.operators.block import BlockOperator
from qalma.operators.product import ProductOperator, ScalarOperator
from qalma.operators.quadratic import QuadraticFormOperator
from qalma.operators.states.basic import (
    DensityOperatorMixin,
)
from qalma.operators.states.block import BlockDensityOperator
from qalma.operators.states.gibbs import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
)
from qalma.operators.states.product import (
    ProductDensityOperator,
)

BASIC_OPERATOR_TYPES = (
    Operator,
    ScalarOperator,
    LocalOperator,
    ProductOperator,
    BlockOperator,
    QuadraticFormOperator,
)

NON_PRODUCT_BASIC_OPERATOR_TYPES = (
    Operator,
    BlockOperator,
    OneBodyOperator,
    QuadraticFormOperator,
)

DENSITY_OPERATOR_BASIC_TYPES = (
    DensityOperatorMixin,
    ProductDensityOperator,
    BlockDensityOperator,
)

NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES = (
    DensityOperatorMixin,
    BlockDensityOperator,
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
    BlockDensityOperator,
    BlockOperator,
    ScalarOperator,
    SumOperator,
)

REAL_NUMERIC_TYPES = tuple((int, float, np.float64))
COMPLEX_NUMERIC_TYPES = tuple((Number, complex, np.complex128))
TYPES_WITH_PREFACTOR = (ScalarOperator, ProductOperator, BlockOperator)
SUM_TYPES = (SumOperator, OneBodyOperator)
