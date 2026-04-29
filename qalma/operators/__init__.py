# -*- coding: utf-8 -*-
""":no-index:

Operators
=========

Classes representing different kind of operators.

"""

from qalma.operators import register_ops
from qalma.operators.arithmetic import (
    OneBodyOperator,
    SumOperator,
    iterable_to_operator,
)
from qalma.operators.basic import (
    LocalOperator,
    Operator,
)
from qalma.operators.product import (
    ProductOperator,
    ScalarOperator,
)
from qalma.operators.quadratic import QuadraticFormOperator
from qalma.operators.qutip import QutipOperator

__all__ = [
    "LocalOperator",
    "OneBodyOperator",
    "Operator",
    "ProductOperator",
    "QuadraticFormOperator",
    "QutipOperator",
    "ScalarOperator",
    "SumOperator",
    "iterable_to_operator",
    "register_ops",
]
