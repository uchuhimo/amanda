from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set


class Op:
    def __init__(
        self,
        attrs=None,
        input_tensors=None,
        control_dependencies=None,
        output_num: int = 1,
    ):
        self.attrs: Dict[str, Any] = dict(attrs or {})
        self.input_tensors: List[Tensor] = list(input_tensors or [])
        self.control_dependencies: List[Op] = list(control_dependencies or [])
        self._output_tensors: List[Tensor] = [
            Tensor(self, i) for i in range(output_num)
        ]

    def update_attr(self, name: str, value: Any):
        self.attrs[name] = value

    def update_input_tensor(self, index: int, tensor: "Tensor"):
        self.input_tensors[index] = tensor

    def add_control_dependency(self, op: "Op"):
        assert op not in self.control_dependencies
        self.control_dependencies.append(op)

    def remove_control_dependency(self, op: "Op"):
        assert op in self.control_dependencies
        self.control_dependencies.remove(op)


@dataclass
class Tensor:
    op: Op
    output_index: int


class Graph:
    def __init__(self, ops: Iterable[Op] = None, attrs=None):
        self.ops: Set[Op] = set()
        self.attrs: Dict[str, Any] = dict(attrs or {})
        if ops is not None:
            for op in ops:
                self.add_op(op)

    def update_attr(self, name: str, value: Any):
        self.attrs[name] = value

    def add_op(self, op: Op) -> None:
        assert op not in self.ops
        self.ops.add(op)

    def remove_op(self, op: Op) -> None:
        assert op in self.ops
        self.ops.remove(op)
