from dataclasses import dataclass
from typing import Any, Dict, List

from qalma.operators.basic import Operator


@dataclass
class Simulation:
    parameters: Dict[Any, Any]
    stats: Dict[Any, Any]
    time_span: List[float]
    expect_ops: Dict[Any, Any]
    states: List[Operator]
