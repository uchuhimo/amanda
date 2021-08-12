from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Tensor:
    value: Any
    op: "Op"
    outputs: List["Op"] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.value}{self.attrs}"


@dataclass
class Op:
    id: Any
    raw_op: Any
    inputs: List[Tensor] = field(default_factory=list)
    outputs: List[Tensor] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.raw_op}{self.attrs}"

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class Graph:
    ops: Dict[Any, Op] = field(default_factory=dict)
    tensors: List[Tensor] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
