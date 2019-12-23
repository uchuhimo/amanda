import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, List, Set

from mmx import core
from mmx.core import OutputPort


class Op(core.Op["Op"]):
    def output(self, index=0) -> OutputPort:
        return OutputPort(self, index)

    def input(self, index=0) -> "InputPort":
        if not (0 <= index < len(self.inputs)):
            raise IndexError
        return InputPort(self, index)

    def input_op(self, index: int) -> "Op":
        if not (0 <= index < len(self.input_ops)):
            raise IndexError
        return self.input_ops[index]

    def output_ops(self, graph: "Graph") -> List["Op"]:
        assert self in graph
        return [op for op in graph.ops if self in op.input_ops]

    @property
    def name(self) -> str:
        return self.attrs["name"]

    @name.setter
    def name(self, name: str):
        self.attrs["name"] = name

    @property
    def type(self) -> str:
        return self.attrs["type"]

    @type.setter
    def type(self, type: str):
        self.attrs["type"] = type


class CompositeOp(Op):
    def __init__(self, graph: "Graph", attrs=None):
        attrs = attrs or {}
        attrs["graph"] = graph
        super().__init__(
            inputs=graph.inputs, control_inputs=graph.control_inputs, attrs=attrs,
        )

    @property
    def graph(self) -> "Graph":
        return self.attrs["graph"]


@dataclass
class InputPort:
    op: Op
    input_index: int

    def is_control(self) -> bool:
        return self.input_index == ControlEdge.CONTROL_EDGE_INDEX

    def __hash__(self):
        return hash((self.op, self.input_index))


class Edge(ABC):
    def __init__(self, src_port: OutputPort, dst_port: InputPort):
        self.src: Op = src_port.op
        self.src_output_index = src_port.output_index
        self.src_port: OutputPort = src_port
        self.dst = dst_port.op
        self.dst_input_index = dst_port.input_index
        self.dst_port: InputPort = dst_port

    @abstractmethod
    def is_control_edge(self) -> bool:
        ...


class DataEdge(Edge):
    def is_control_edge(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DataEdge)
            and self.src_port == other.src_port
            and self.dst_port == other.dst_port
        )

    def __hash__(self):
        return hash((self.src_port, self.dst_port))


class ControlEdge(Edge):
    def __init__(self, src: Op, dst: Op):
        super().__init__(
            src_port=OutputPort(src, self.CONTROL_EDGE_INDEX),
            dst_port=InputPort(dst, self.CONTROL_EDGE_INDEX),
        )

    CONTROL_EDGE_INDEX = -1

    def is_control_edge(self) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ControlEdge)
            and self.src == other.src
            and self.dst == other.dst
        )

    def __hash__(self):
        return hash((self.src, self.dst))


class Graph(core.Graph[Op]):
    @property
    def post_order_ops(self) -> Iterator[Op]:
        upstream_ops = set(
            itertools.chain.from_iterable(
                map(lambda op: self.lift_ops(op.input_ops), self.ops,)
            )
        )
        output_ops = list(self.ops - upstream_ops)
        returned_ops: Set[Op] = set()

        def dfs(current_op):
            if current_op in returned_ops:
                return
            else:
                for input_op in self.lift_ops(current_op.input_ops):
                    yield from dfs(input_op)
                yield current_op
                returned_ops.add(current_op)

        for output_op in output_ops:
            yield from dfs(output_op)

    @property
    def edges(self) -> Iterator[Edge]:
        for op in self.post_order_ops:
            for input_index, src_port in enumerate(op.inputs):
                if src_port.op in self.ops:
                    yield DataEdge(src_port=src_port, dst_port=op.input(input_index))
            for src in op.control_inputs:
                if src in self.ops:
                    yield ControlEdge(src=src, dst=op)

    def set_attr(self, attr: str, value: Any) -> None:
        for op in self.ops:
            op.attrs[attr] = value

    def lift_op(self, op: Op) -> Op:
        if op in self.ops:
            return op
        else:
            for op_in_graph in self.ops:
                if isinstance(op_in_graph, CompositeOp) and op in op_in_graph.graph:
                    return op_in_graph
            return op

    def lift_ops(self, ops: List[Op]) -> List[Op]:
        return list(map(lambda op: self.lift_op(op), ops))

    @property
    def inputs(self) -> List[OutputPort[Op]]:
        def inputs_iter():
            for op in self.ops:
                for input in op.inputs:
                    if input.op not in self:
                        yield input

        return list(inputs_iter())

    @property
    def control_inputs(self) -> Set[Op]:
        def control_inputs_iter():
            for op in self.ops:
                for control_input in op.control_inputs:
                    if control_input not in self:
                        yield control_input

        return set(control_inputs_iter())

    def __contains__(self, op: Op) -> bool:
        if op in self.ops:
            return True
        else:
            for op_in_graph in self.ops:
                if isinstance(op_in_graph, CompositeOp) and op in op_in_graph.graph:
                    return True
            return False

    def duplicate(self, split_fn, merge_fn):
        # TODO
        ...
