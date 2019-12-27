from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Set, TypeVar

from mmx.exception import IrremovableOpError

T = TypeVar("T", bound="Op")


class Op(Generic[T]):
    def __init__(self, attrs=None, inputs=None, control_inputs=None):
        self.attrs: Dict[str, Any] = attrs or {}
        self.inputs: List["OutputPort[T]"] = list(inputs or [])
        self.control_inputs: Set[T] = set(control_inputs or set())

    def add_control(self, op: T):
        assert op not in self.control_inputs
        self.control_inputs.add(op)

    def remove_control(self, op: T):
        assert op in self.control_inputs
        self.control_inputs.remove(op)

    @property
    def input_ops(self) -> List[T]:
        return list(map(lambda port: port.op, self.inputs))


@dataclass
class OutputPort(Generic[T]):
    op: T
    output_index: int

    def __hash__(self) -> int:
        return hash((self.op, self.output_index))


class Graph(Generic[T]):
    def __init__(self, ops=None, attrs=None):
        self.ops: Set[T] = set(ops or set())
        self.attrs: Dict[str, Any] = attrs or {}

    def add(self, op: T) -> None:
        assert op not in self.ops
        self.ops.add(op)

    def is_removable(self, op: T) -> bool:
        for other_op in self.ops:
            if other_op != op and (
                op in other_op.input_ops or op in other_op.control_inputs
            ):
                return False
        return True

    def remove(self, op: T) -> None:
        assert op in self.ops
        if not self.is_removable(op):
            raise IrremovableOpError
        self.ops.remove(op)
