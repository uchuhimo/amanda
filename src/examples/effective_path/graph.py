from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Op:
    raw_op: Any
    input_ops: List[Any]
    inputs: List[Any] = field(default_factory=list)
    outputs: List[Any] = field(default_factory=list)


@dataclass
class Graph:
    ops: List[Op] = field(default_factory=list)
