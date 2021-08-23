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

    def clone(self):
        return Tensor(
            value=self.value,
            op=self.op,
            outputs=list(self.outputs),
            attrs=dict(self.attrs),
        )


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

    def clone(self):
        return Op(
            id=self.id,
            raw_op=self.raw_op,
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            attrs=dict(self.attrs),
        )


@dataclass
class Graph:
    ops: Dict[Any, Op] = field(default_factory=dict)
    tensors: List[Tensor] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def clone(self):
        op_to_new_ops = {id(op): op.clone() for op in self.ops.values()}
        tensor_to_new_tensors = {id(tensor): tensor.clone() for tensor in self.tensors}
        for new_op in op_to_new_ops.values():
            new_op.inputs = [
                tensor_to_new_tensors[id(input)] for input in new_op.inputs
            ]
            new_op.outputs = [
                tensor_to_new_tensors[id(output)] for output in new_op.outputs
            ]
        for new_tensor in tensor_to_new_tensors.values():
            new_tensor.op = op_to_new_ops[id(new_tensor.op)]
            new_tensor.outputs = [
                op_to_new_ops[id(output)] for output in new_tensor.outputs
            ]
        return Graph(
            ops={key: op_to_new_ops[id(op)] for key, op in self.ops.items()},
            tensors=[tensor_to_new_tensors[id(tensor)] for tensor in self.tensors],
            attrs=dict(self.attrs),
        )
