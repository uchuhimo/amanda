import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, cast

from amanda import core
from amanda.exception import IrremovableOpError
from amanda.namespace import Namespace, Registry, default_namespace, get_global_registry


class Op(core.Op["Op"]):
    def __init__(
        self,
        attrs=None,
        input_tensors=None,
        control_dependencies=None,
        output_num: int = 1,
    ):
        super().__init__(attrs, input_tensors, control_dependencies, output_num)
        self.output_tensors: List["Tensor"] = [
            Tensor(self, i) for i in range(self.output_num)
        ]

    def output_tensor(self, index=0) -> "Tensor":
        return self.output_tensors[index]

    @property
    def input_ops(self) -> List["Op"]:
        return list(map(lambda port: port.op, self.input_tensors))

    def input_op(self, index: int) -> "Op":
        if not (0 <= index < len(self.input_tensors)):
            raise IndexError
        return self.input_ops[index]

    def input_port(self, index=0) -> "InputPort":
        if not (0 <= index < len(self.input_tensors)):
            raise IndexError
        return InputPort(self, index)

    def input_index(self, input_op: "Op") -> int:
        for index, input in enumerate(self.input_tensors):
            if input.op == input_op:
                return index
        raise IndexError

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

    def __repr__(self) -> str:
        attrs_string = ", ".join(
            [f"{key}={value}" for key, value in self.attrs.items()]
        )
        return f"Op({attrs_string})"


class CompositeOp(Op):
    def __init__(self, graph: "Graph", attrs=None):
        attrs = attrs or {}
        attrs["graph"] = graph
        super().__init__(
            input_tensors=graph.input_tensors,
            control_dependencies=graph.control_dependencies,
            attrs=attrs,
        )

    @property
    def graph(self) -> "Graph":
        return self.attrs["graph"]


class Tensor(core.Tensor["Op"]):
    def __hash__(self) -> int:
        return hash((self.op, self.output_index))


@dataclass
class InputPort:
    op: Op
    input_index: int

    def is_control(self) -> bool:
        return self.input_index == ControlEdge.CONTROL_EDGE_INDEX

    def __hash__(self):
        return hash((self.op, self.input_index))


class Edge(ABC):
    def __init__(self, src_tensor: Tensor, dst_port: InputPort):
        self.src_op: Op = src_tensor.op
        self.src_output_index = src_tensor.output_index
        self.src_tensor: Tensor = src_tensor
        self.dst_op = dst_port.op
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
            and self.src_tensor == other.src_tensor
            and self.dst_port == other.dst_port
        )

    def __hash__(self):
        return hash((self.src_tensor, self.dst_port))


class ControlEdge(Edge):
    def __init__(self, src: Op, dst: Op):
        super().__init__(
            src_tensor=Tensor(src, self.CONTROL_EDGE_INDEX),
            dst_port=InputPort(dst, self.CONTROL_EDGE_INDEX),
        )

    CONTROL_EDGE_INDEX = -1

    def is_control_edge(self) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ControlEdge)
            and self.src_op == other.src_op
            and self.dst_op == other.dst_op
        )

    def __hash__(self):
        return hash((self.src_op, self.dst_op))


@dataclass
class OutputEdges:
    tensor: Tensor
    edges: List[Edge]


class Graph(core.Graph[Op]):
    def __init__(self, ops=None, attrs=None):
        self.name_to_op: Dict[str, Op] = {}
        self.composite_ops: Set[CompositeOp] = []
        self.cached_post_order_ops: List[Op] = None
        self.cached_edges: List[Edge] = None
        self.cached_data_edges: List[Edge] = None
        self.cached_control_edges: List[Edge] = None
        self.cached_tensor_to_edges: Dict[Tensor, List[Edge]] = None
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
        if self.cached_tensor_to_edges is not None:
            graph.cached_tensor_to_edges = self.cached_tensor_to_edges
        return graph

    def invalidate_cache(self):
        self.cached_post_order_ops = None
        self.cached_edges = None
        self.cached_data_edges = None
        self.cached_control_edges = None
        self.cached_tensor_to_edges = None

    def add_op(self, op: Op) -> None:
        super().add_op(op)
        if op.has_name():
            assert op.name not in self.name_to_op
            self.name_to_op[op.name] = op
        if isinstance(op, CompositeOp):
            self.composite_ops.add(op)
        self.invalidate_cache()

    def is_removable(self, op: Op) -> bool:
        for other_op in self.ops:
            if other_op != op and (
                op in other_op.input_ops or op in other_op.control_dependencies
            ):
                return False
        return True

    def remove_op(self, op: Op) -> None:
        if not self.is_removable(op):
            raise IrremovableOpError
        super().remove_op(op)
        if op.has_name():
            del self.name_to_op[op.name]
        if isinstance(op, CompositeOp):
            self.composite_ops.remove(op)
        self.invalidate_cache()

    def get_op_by_name(self, name: str) -> Optional[Op]:
        return self.name_to_op.get(name)

    def replace_tensor(self, old_tensor: Tensor, new_tensor: Tensor):
        for op in self.ops:
            if old_tensor in op.input_tensors:
                op.input_tensors[op.input_index(old_tensor.op)] = new_tensor

    @property
    def post_order_ops(self) -> List[Op]:
        if self.cached_post_order_ops is not None:
            return self.cached_post_order_ops
        upstream_ops = set(
            itertools.chain.from_iterable(
                map(
                    lambda op: self.lift_ops(
                        op.input_ops + list(op.control_dependencies)
                    ),
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
                    current_op.input_ops + list(current_op.control_dependencies)
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
                src_tensor=cast(Tensor, tensor), dst_port=InputPort(op, input_index),
            )
            for op in self.ops
            for input_index, tensor in enumerate(op.input_tensors)
            if tensor.op in self.ops
        ]
        return self.cached_data_edges

    @property
    def control_edges(self) -> List[Edge]:
        if self.cached_control_edges is not None:
            return self.cached_control_edges
        self.cached_control_edges = [
            ControlEdge(src=src, dst=op)
            for op in self.ops
            for src in op.control_dependencies
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
    def tensor_to_edges(self) -> Dict[Tensor, List[Edge]]:
        if self.cached_tensor_to_edges is not None:
            return self.cached_tensor_to_edges
        tensor_to_edges: Dict[Tensor, List[Edge]] = {
            tensor: [] for op in self.ops for tensor in op.output_tensors
        }
        for edge in self.data_edges:
            tensor_to_edges[edge.src_tensor].append(edge)
        self.cached_tensor_to_edges = tensor_to_edges
        return tensor_to_edges

    def edges_from_tensor(self, tensor: Tensor) -> List[Edge]:
        if tensor not in self.tensor_to_edges:
            return []
        else:
            return self.tensor_to_edges[tensor]

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
    def input_tensors(self) -> List[Tensor]:
        def input_tensors_iter():
            for op in self.ops:
                for input in op.input_tensors:
                    if input.op not in self:
                        yield input

        return list(input_tensors_iter())

    @property
    def control_dependencies(self) -> Set[Op]:
        def control_dependencies_iter():
            for op in self.ops:
                for control_dependency in op.control_dependencies:
                    if control_dependency not in self:
                        yield control_dependency

        return set(control_dependencies_iter())

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

    @property
    def namespace(self) -> Namespace:
        return self.attrs["namespace"]

    @namespace.setter
    def namespace(self, namespace: Namespace):
        self.attrs["namespace"] = namespace

    def to_namespace(self, namespace: Namespace, registry: Registry = None) -> "Graph":
        if namespace == self.namespace:
            return self
        else:
            registry = registry or get_global_registry()
            return registry.get_mapper(self.namespace, namespace).map(self, namespace)

    def to_default_namespace(self) -> "Graph":
        return self.to_namespace(default_namespace())
