import itertools
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, cast

import immutables

from amanda.attributes import Attributes
from amanda.exception import IrremovableOpError
from amanda.namespace import (
    Namespace,
    Registry,
    default_namespace,
    get_global_registry,
    internal_namespace,
)


class NamespaceMixin:
    _namespace_key = internal_namespace().qualified("namespace")

    @property
    def attrs(self) -> Attributes:
        ...

    @property
    def namespace(self) -> Namespace:
        return self.attrs.get(self._namespace_key, default_namespace())

    @namespace.setter
    def namespace(self, namespace: Namespace):
        self.attrs[self._namespace_key] = namespace

    def attr_name_in_default_namespace(self, attr_name: str) -> str:
        if self.namespace == default_namespace():
            return attr_name
        else:
            return default_namespace().qualified(attr_name)


class Op(NamespaceMixin):
    def __init__(
        self,
        attrs=None,
        input_tensors=None,
        control_dependencies=None,
        output_num: int = 1,
    ):
        self._attrs: Attributes = Attributes(attrs or {})
        self._input_tensors: List["Tensor"] = list(input_tensors or [])
        self._input_num = len(self._input_tensors)
        self._control_dependencies: List[Op] = list(control_dependencies or [])
        self._output_num = output_num
        self._output_tensors: List["Tensor"] = [
            Tensor(self, i) for i in range(output_num)
        ]

    @property
    def attrs(self) -> Attributes:
        return self._attrs

    @property
    def input_tensors(self) -> List["Tensor"]:
        return self._input_tensors

    @property
    def input_num(self) -> int:
        return self._input_num

    @property
    def control_dependencies(self) -> List["Op"]:
        return self._control_dependencies

    @property
    def output_tensors(self) -> List["Tensor"]:
        return self._output_tensors

    @property
    def output_num(self) -> int:
        return self._output_num

    def output_tensor(self, index=0) -> "Tensor":
        return self.output_tensors[index]

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
        return self.attr_name_in_default_namespace("name") in self.attrs

    @property
    def name(self) -> str:
        return self.attrs[self.attr_name_in_default_namespace("name")]

    @name.setter
    def name(self, name: str):
        self.attrs[self.attr_name_in_default_namespace("name")] = name

    @property
    def type(self) -> str:
        return self.attrs[self.attr_name_in_default_namespace("type")]

    @type.setter
    def type(self, type: str):
        self.attrs[self.attr_name_in_default_namespace("type")] = type

    def __repr__(self) -> str:
        attrs_string = ", ".join(
            [f"{key}={value}" for key, value in self.attrs.items()]
        )
        return f"Op({attrs_string})"

    def dict(self):
        return dict(
            attrs=dict(self.attrs),
            input_tensors=[
                f"{input_tensor.op.name}:{input_tensor.output_index}"
                for input_tensor in self.input_tensors
            ],
            control_dependencies=[
                control_dependency.name
                for control_dependency in self.control_dependencies
            ],
        )

    def json(self):
        return json.dumps(self.dict(), indent=4)


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


@dataclass
class Tensor:
    op: Op
    output_index: int

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


class Graph(NamespaceMixin):
    def __init__(self, ops=None, attrs=None):
        self._ops: immutables.Map = immutables.Map()
        self._attrs: Attributes = Attributes(attrs or {})
        self._name_to_op: immutables.Map = immutables.Map()
        self._composite_ops: immutables.Map = immutables.Map()
        self._cached_post_order_ops: List[Op] = None
        self._cached_edges: List[Edge] = None
        self._cached_data_edges: List[Edge] = None
        self._cached_control_edges: List[Edge] = None
        self._cached_tensor_to_edges: Dict[Tensor, List[Edge]] = None
        if ops is not None:
            for op in ops:
                self.add_op(op)

    @property
    def ops(self) -> Iterable[Op]:
        return self._ops

    @property
    def attrs(self) -> Attributes:
        return self._attrs

    @property
    def names(self) -> Iterable[str]:
        return self._name_to_op

    def update_attr(self, name: str, value: Any):
        self.attrs[name] = value

    def add_op(self, op: Op) -> None:
        assert op not in self._ops
        self._ops = self._ops.set(op, True)
        if op.has_name():
            assert op.name not in self._name_to_op
            self._name_to_op = self._name_to_op.set(op.name, op)
        if isinstance(op, CompositeOp):
            self._composite_ops = self._composite_ops.set(op, True)
        self.invalidate_cache()

    def is_removable(self, op: Op) -> bool:
        for other_op in self._ops:
            if other_op != op and (
                op in other_op.input_ops or op in other_op.control_dependencies
            ):
                return False
        return True

    def remove_op(self, op: Op) -> None:
        if not self.is_removable(op):
            raise IrremovableOpError
        assert op in self._ops
        self._ops = self._ops.delete(op)
        if op.has_name():
            self._name_to_op = self._name_to_op.delete(op.name)
        if isinstance(op, CompositeOp):
            self._composite_ops = self._composite_ops.delete(op)
        self.invalidate_cache()

    def get_op_by_name(self, name: str) -> Optional[Op]:
        return self._name_to_op.get(name)

    def replace_tensor(self, old_tensor: Tensor, new_tensor: Tensor):
        for op in self.ops:
            if old_tensor in op.input_tensors:
                op.input_tensors[op.input_index(old_tensor.op)] = new_tensor

    @property
    def post_order_ops(self) -> List[Op]:
        if self._cached_post_order_ops is not None:
            return self._cached_post_order_ops
        upstream_ops = set(
            itertools.chain.from_iterable(
                map(
                    lambda op: self.lift_ops(op.input_ops + op.control_dependencies),
                    self.ops,
                )
            )
        )
        output_ops = set(self.ops) - upstream_ops
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
                    current_op.input_ops + current_op.control_dependencies
                ):
                    dfs_stack.append(input_op)
        self._cached_post_order_ops = post_order_ops
        return post_order_ops

    @property
    def data_edges(self) -> List[Edge]:
        if self._cached_data_edges is not None:
            return self._cached_data_edges
        self._cached_data_edges = [
            DataEdge(
                src_tensor=cast(Tensor, tensor), dst_port=InputPort(op, input_index),
            )
            for op in self.ops
            for input_index, tensor in enumerate(op.input_tensors)
            if tensor.op in self
        ]
        return self._cached_data_edges

    @property
    def control_edges(self) -> List[Edge]:
        if self._cached_control_edges is not None:
            return self._cached_control_edges
        self._cached_control_edges = [
            ControlEdge(src=src, dst=op)
            for op in self.ops
            for src in op.control_dependencies
            if src in self
        ]
        return self._cached_control_edges

    @property
    def edges(self) -> List[Edge]:
        if self._cached_edges is not None:
            return self._cached_edges
        self._cached_edges = self.data_edges + self.control_edges
        return self._cached_edges

    @property
    def tensor_to_edges(self) -> Dict[Tensor, List[Edge]]:
        if self._cached_tensor_to_edges is not None:
            return self._cached_tensor_to_edges
        tensor_to_edges: Dict[Tensor, List[Edge]] = {
            tensor: [] for op in self.ops for tensor in op.output_tensors
        }
        for edge in self.data_edges:
            tensor_to_edges[edge.src_tensor].append(edge)
        self._cached_tensor_to_edges = tensor_to_edges
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
        if len(self._composite_ops) != 0:
            for composite_op in self._composite_ops:
                if op in composite_op.graph:
                    return composite_op
        return op

    def lift_ops(self, ops: List[Op]) -> List[Op]:
        if len(self._composite_ops) != 0:
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
        if op in self._ops:
            return True
        elif len(self._composite_ops) != 0:
            for composite_op in self._composite_ops:
                if op in composite_op.graph:
                    return True
        return False

    def invalidate_cache(self):
        self._cached_post_order_ops = None
        self._cached_edges = None
        self._cached_data_edges = None
        self._cached_control_edges = None
        self._cached_tensor_to_edges = None

    def clone(self) -> "Graph":
        graph = Graph()
        graph._ops = self._ops
        graph._attrs = self.attrs.copy()
        graph._name_to_op = self._name_to_op
        graph._composite_ops = self._composite_ops
        if self._cached_post_order_ops is not None:
            graph._cached_post_order_ops = self._cached_post_order_ops
        if self._cached_edges is not None:
            graph._cached_edges = self._cached_edges
        if self._cached_data_edges is not None:
            graph._cached_data_edges = self._cached_data_edges
        if self._cached_control_edges is not None:
            graph._cached_control_edges = self._cached_control_edges
        if self._cached_tensor_to_edges is not None:
            graph._cached_tensor_to_edges = self._cached_tensor_to_edges
        return graph

    def __copy__(self):
        return self.clone()

    def copy(self):
        return self.clone()

    def dict(self):
        return dict(ops=[op.dict() for op in self.ops], attrs=dict(self.attrs),)

    def json(self):
        return json.dumps(self.dict(), indent=4)

    def duplicate(self) -> "Graph":
        new_graph = Graph(attrs=self.attrs.copy())
        for op in self.post_order_ops:
            target_op = Op(
                attrs=op.attrs.copy(),
                input_tensors=[
                    new_graph.get_op_by_name(input_tensor.op.name).output_tensor(
                        input_tensor.output_index
                    )
                    for input_tensor in op.input_tensors
                ],
                control_dependencies=[
                    new_graph.get_op_by_name(control_dependency.name)
                    for control_dependency in op.control_dependencies
                ],
                output_num=op.output_num,
            )
            new_graph.add_op(target_op)
        return new_graph

    def to_namespace(self, namespace: Namespace, registry: Registry = None) -> "Graph":
        if namespace == self.namespace:
            return self
        else:
            registry = registry or get_global_registry()
            return registry.get_mapper(self.namespace, namespace).map(self, namespace)

    def to_default_namespace(self) -> "Graph":
        return self.to_namespace(default_namespace())
