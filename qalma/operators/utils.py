"""Utility functions used in Operator classes."""

import logging
from typing import Callable, Optional

from qalma.settings import (
    QALMA_INFER_ARITHMETICS,
)


def find_arithmetic_implementation(
    op1, op2, dispatch_table: dict
) -> Optional[Callable]:
    """Find the function that implements the operation
    op1 [operation] op2 in the dispatch table
    dispatch.
    If the combination of types is not already in the dispatch table,
    store it.
    """
    type_op1, type_op2 = type(op1), type(op2)
    op1_parent_classes = type_op1.__mro__
    op2_parent_classes = type_op2.__mro__
    # Go over the combinations of parent classes
    for lhf in op1_parent_classes:
        for rhf in op2_parent_classes:
            key = (lhf, rhf)
            if key in dispatch_table:
                func = dispatch_table[key]
                if QALMA_INFER_ARITHMETICS:
                    dispatch_table[(type_op1, type_op2)] = func
                    return func
                logging.warning("try with %s", func.__code__)
                return None

    # Last resource: try if the operands are instances of one of the keys
    # in the dispatch table.
    # Required for example for keys of the form (Operator, complex).

    for key, func in dispatch_table.items():
        if isinstance(op1, key[0]) and isinstance(op2, key[1]):
            func = dispatch_table[key]
            if QALMA_INFER_ARITHMETICS:
                dispatch_table[(type_op1, type_op2)] = func
                return func
            logging.warning("try with %s", func.__code__)
            return None
    return None
