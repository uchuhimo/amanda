from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterable, List, Set, TypeVar

from mmx.exception import IrremovableOpError

T = TypeVar("T", bound="Op")


class Op(Generic[T]):
    def __init__(self, attrs=None, input_tensors=None, control_dependencies=None):
        self.attrs: Dict[str, Any] = attrs or {}
        self.input_tensors: List["Tensor[T]"] = list(input_tensors or [])
        self.control_dependencies: Set[T] = set(control_dependencies or set())

    def add_control_dependency(self, op: T):
        assert op not in self.control_dependencies
        self.control_dependencies.add(op)

    def remove_control_dependency(self, op: T):
        assert op in self.control_dependencies
        self.control_dependencies.remove(op)

    @property
    def input_ops(self) -> List[T]:
        return list(map(lambda port: port.op, self.input_tensors))


@dataclass
class Tensor(Generic[T]):
    op: T
    output_index: int

    def __hash__(self) -> int:
        return hash((self.op, self.output_index))


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

    def is_removable(self, op: T) -> bool:
        for other_op in self.ops:
            if other_op != op and (
                op in other_op.input_ops or op in other_op.control_dependencies
            ):
                return False
        return True

    def remove_op(self, op: T) -> None:
        assert op in self.ops
        if not self.is_removable(op):
            raise IrremovableOpError
        self.ops.remove(op)
