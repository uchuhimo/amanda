from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterable, List, Set, TypeVar

T = TypeVar("T", bound="Op")


class Op(Generic[T]):
    def __init__(
        self,
        attrs=None,
        input_tensors=None,
        control_dependencies=None,
        output_num: int = 1,
    ):
        self.attrs: Dict[str, Any] = attrs or {}
        self.input_tensors: List["Tensor[T]"] = list(input_tensors or [])
        self.control_dependencies: Set[T] = set(control_dependencies or set())
        self.output_num: int = output_num

    def add_control_dependency(self, op: T):
        assert op not in self.control_dependencies
        self.control_dependencies.add(op)

    def remove_control_dependency(self, op: T):
        assert op in self.control_dependencies
        self.control_dependencies.remove(op)


@dataclass
class Tensor(Generic[T]):
    op: T
    output_index: int


class Graph(Generic[T]):
    def __init__(self, ops: Iterable[T] = None, attrs=None):
        self.ops: Set[T] = set()
        self.attrs: Dict[str, Any] = attrs or {}
        if ops is not None:
            for op in ops:
                self.add_op(op)

    def add_op(self, op: T) -> None:
        assert op not in self.ops
        self.ops.add(op)

    def remove_op(self, op: T) -> None:
        assert op in self.ops
        self.ops.remove(op)
