import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, cast

from mmx import core


class Op(core.Op["Op"]):
    def output(self, index=0) -> "OutputPort":
        return OutputPort(self, index)

    def input(self, index=0) -> "InputPort":
        if not (0 <= index < len(self.inputs)):
            raise IndexError
        return InputPort(self, index)

    def input_op(self, index: int) -> "Op":
        if not (0 <= index < len(self.input_ops)):
            raise IndexError
        return self.input_ops[index]

    def input_index(self, input_op: "Op") -> int:
        for input in self.inputs:
            if input.op == input_op:
                return input.output_index
        raise RuntimeError()

    def output_ports(self, graph: "Graph") -> List["OutputPort"]:
        return graph.output_ports_from_op(self)

    def has_name(self) -> bool:
        return "name" in self.attrs

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

    def __str__(self) -> str:
        return f"Op(name={self.name}, type={self.type})"


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


class OutputPort(core.OutputPort["Op"]):
    def output_edges(self, graph: "Graph") -> List["Edge"]:
        return graph.edges_from_port(self)


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


@dataclass
class OutputEdges:
    output_port: OutputPort
    edges: List[Edge]


class Graph(core.Graph[Op]):
    def __init__(self, ops=None, attrs=None):
        self.name_to_op: Dict[str, Op] = {}
        self.composite_ops: Set[CompositeOp] = []
        self.cached_post_order_ops: List[Op] = None
        self.cached_edges: List[Edge] = None
        self.cached_data_edges: List[Edge] = None
        self.cached_control_edges: List[Edge] = None
        self.cached_op_to_port_to_edges: Dict[Op, Dict[OutputPort, List[Edge]]] = None
        super().__init__(ops, attrs)

    @property
    def names(self) -> List[str]:
        return list(self.name_to_op.keys())

    def clone(self) -> "Graph":
        graph = Graph()
        graph.ops = set(self.ops)
        graph.attrs = dict(self.attrs)
        graph.name_to_op = dict(self.name_to_op)
        graph.composite_ops = set(self.composite_ops)
        if self.cached_post_order_ops is not None:
            graph.cached_post_order_ops = self.cached_post_order_ops
        if self.cached_edges is not None:
            graph.cached_edges = self.cached_edges
        if self.cached_data_edges is not None:
            graph.cached_data_edges = self.cached_data_edges
        if self.cached_control_edges is not None:
            graph.cached_control_edges = self.cached_control_edges
        if self.cached_op_to_port_to_edges is not None:
            graph.cached_op_to_port_to_edges = self.cached_op_to_port_to_edges
        return graph

    def invalidate_cache(self):
        self.cached_post_order_ops = None
        self.cached_edges = None
        self.cached_data_edges = None
        self.cached_control_edges = None
        self.cached_op_to_port_to_edges = None

    def add(self, op: Op) -> None:
        super().add(op)
        if op.has_name():
            assert op.name not in self.name_to_op
            self.name_to_op[op.name] = op
        if isinstance(op, CompositeOp):
            self.composite_ops.add(op)
        self.invalidate_cache()

    def remove(self, op: Op) -> None:
        super().remove(op)
        if op.has_name():
            del self.name_to_op[op.name]
        if isinstance(op, CompositeOp):
            self.composite_ops.remove(op)
        self.invalidate_cache()

    def get_op_by_name(self, name: str) -> Optional[Op]:
        return self.name_to_op.get(name)

    def replace_tensor(self, old_tensor: OutputPort, new_tensor: OutputPort):
        for op in self.ops:
            if old_tensor in op.inputs:
                op.inputs[op.input_index(old_tensor.op)] = new_tensor

    @property
    def post_order_ops(self) -> List[Op]:
        if self.cached_post_order_ops is not None:
            return self.cached_post_order_ops
        upstream_ops = set(
            itertools.chain.from_iterable(
                map(
                    lambda op: self.lift_ops(op.input_ops + list(op.control_inputs)),
                    self.ops,
                )
            )
        )
        output_ops = self.ops - upstream_ops
        dfs_stack = sorted(list(output_ops), key=lambda op: op.name, reverse=True)
        walked_ops: Set[Op] = set()
        returned_ops: Set[Op] = set()
        post_order_ops: List[Op] = []

        while len(dfs_stack) > 0:
            current_op = dfs_stack.pop()
            if current_op in walked_ops:
                walked_ops.remove(current_op)
                returned_ops.add(current_op)
                post_order_ops.append(current_op)
            elif current_op not in returned_ops:
                dfs_stack.append(current_op)
                walked_ops.add(current_op)
                for input_op in self.lift_ops(
                    current_op.input_ops + list(current_op.control_inputs)
                ):
                    dfs_stack.append(input_op)
        self.cached_post_order_ops = post_order_ops
        return post_order_ops

    @property
    def data_edges(self) -> List[Edge]:
        if self.cached_data_edges is not None:
            return self.cached_data_edges
        self.cached_data_edges = [
            DataEdge(
                src_port=cast(OutputPort, src_port),
                dst_port=InputPort(op, input_index),
            )
            for op in self.ops
            for input_index, src_port in enumerate(op.inputs)
            if src_port.op in self.ops
        ]
        return self.cached_data_edges

    @property
    def control_edges(self) -> List[Edge]:
        if self.cached_control_edges is not None:
            return self.cached_control_edges
        self.cached_control_edges = [
            ControlEdge(src=src, dst=op)
            for op in self.ops
            for src in op.control_inputs
            if src in self.ops
        ]
        return self.cached_control_edges

    @property
    def edges(self) -> List[Edge]:
        if self.cached_edges is not None:
            return self.cached_edges
        self.cached_edges = self.data_edges + self.control_edges
        return self.cached_edges

    @property
    def op_to_port_to_edges(self) -> Dict[Op, Dict[OutputPort, List[Edge]]]:
        if self.cached_op_to_port_to_edges is not None:
            return self.cached_op_to_port_to_edges
        op_to_port_to_edges: Dict[Op, Dict[OutputPort, List[Edge]]] = {
            op: {} for op in self.ops
        }
        for edge in self.data_edges:
            port_to_edges = op_to_port_to_edges[edge.src]
            if edge.src_output_index not in port_to_edges:
                port_to_edges[edge.src_port] = []
            port_to_edges[edge.src_port].append(edge)
        self.cached_op_to_port_to_edges = op_to_port_to_edges
        return op_to_port_to_edges

    def output_ports_from_op(self, op: Op) -> List[OutputPort]:
        if op not in self.op_to_port_to_edges:
            return []
        else:
            return list(self.op_to_port_to_edges[op].keys())

    def edges_from_port(self, port: OutputPort) -> List[Edge]:
        if port.op not in self.op_to_port_to_edges:
            return []
        elif port not in self.op_to_port_to_edges[port.op]:
            return []
        else:
            return self.op_to_port_to_edges[port.op][port]

    def set_attr(self, attr: str, value: Any) -> None:
        for op in self.ops:
            op.attrs[attr] = value

    def lift_op(self, op: Op) -> Op:
        if len(self.composite_ops) != 0:
            for composite_op in self.composite_ops:
                if op in composite_op.graph:
                    return composite_op
        return op

    def lift_ops(self, ops: List[Op]) -> List[Op]:
        if len(self.composite_ops) != 0:
            return list(map(lambda op: self.lift_op(op), ops))
        else:
            return ops

    @property
    def inputs(self) -> List[OutputPort]:
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
        elif len(self.composite_ops) != 0:
            for composite_op in self.composite_ops:
                if op in composite_op.graph:
                    return True
        return False

    def duplicate(self, split_fn, merge_fn):
        # TODO
        ...
