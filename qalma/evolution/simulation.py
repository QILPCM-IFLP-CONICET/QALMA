"""
This module defines the `Simulation` dataclass containing the
state and the result of a simulation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from qalma.operators.basic import Operator


@dataclass
class Simulation:
    """
    Hold the state and result of a simulation.
    """

    parameters: Dict[Any, Any]
    stats: Dict[Any, Any]
    time_span: List[float]
    expect_ops: Dict[Any, Any]
    states: List[Operator]
