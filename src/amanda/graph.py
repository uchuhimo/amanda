import copy
import itertools
import json
import typing
import uuid
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set
from uuid import UUID

import immutables
import numpy as np

from amanda.attributes import Attributes
from amanda.exception import IrremovableOpError
from amanda.namespace import (
    Namespace,
    Registry,
    default_namespace,
    get_global_registry,
    internal_namespace,
)


class OpAttrKey:
    namespace = internal_namespace().qualified("namespace")
    uuid = internal_namespace().qualified("uuid")
    has_name = internal_namespace().qualified("has_name")


class NamespaceMixin:
    @property
    def attrs(self) -> Attributes:
        ...

    @property
    def namespace(self) -> Namespace:
        return self.attrs.get(OpAttrKey.namespace, default_namespace())

    @namespace.setter
    def namespace(self, namespace: Namespace):
        self.attrs[OpAttrKey.namespace] = namespace

    def attr_name_in_default_namespace(self, attr_name: str) -> str:
        if self.namespace == default_namespace():
            return attr_name
        else:
            return default_namespace().qualified(attr_name)


class InputTensors:
    def __init__(self, op: "Op", tensors: List["Tensor"]):
        self.op: "Op" = op
        self.tensors: List["Tensor"] = tensors
        self._input_ports = [InputPort(op, index) for index in range(len(tensors))]
        self._input_edges: List["Edge"] = [
            DataEdge(tensor, self._input_ports[index])
            for index, tensor in enumerate(tensors)
        ]
        for edge in self._input_edges:
            edge.src_tensor._output_edges.add(edge)

    @property
    def input_ports(self) -> List["InputPort"]:
        return self._input_ports

    @property
    def input_edges(self) -> List["Edge"]:
        return self._input_edges

    def __setitem__(self, index: int, tensor: "Tensor") -> None:
        old_tensor: Tensor = self.tensors[index]
        old_tensor._output_edges.remove(self._input_edges[index])
        self.tensors[index] = tensor
        edge = DataEdge(tensor, self._input_ports[index])
        self._input_edges[index] = edge
        tensor._output_edges.add(edge)

    def __len__(self) -> int:
        return len(self.tensors)

    def __iter__(self) -> Iterator["Tensor"]:
        return iter(self.tensors)

    def __getitem__(self, i: int) -> "Tensor":
        return self.tensors[i]

    def __contains__(self, tensor: "Tensor") -> bool:
        return tensor in self.tensors


class Op(NamespaceMixin):
    def __init__(
        self,
        attrs=None,
        input_tensors=None,
        control_dependencies=None,
        output_num: int = 1,
    ):
        self._attrs: Attributes = Attributes(attrs or {})
        self._input_tensors: InputTensors = InputTensors(
            self, list(input_tensors or [])
        )
        self._input_num = len(self._input_tensors)
        self._control_dependencies: List[Op] = []
        self._output_num = output_num
        self._output_tensors: List[Tensor] = [
            Tensor(self, i) for i in range(output_num)
        ]
        self._control_dependents: Set[Op] = typing.cast(Set[Op], weakref.WeakSet())
        self._attrs[OpAttrKey.uuid] = uuid.uuid4()
        control_dependencies = control_dependencies or []
        for control_dependency in control_dependencies:
            self.add_control_dependency(control_dependency)

    @property
    def attrs(self) -> Attributes:
        return self._attrs

    @property
    def input_tensors(self) -> InputTensors:
        return self._input_tensors

    @property
    def input_num(self) -> int:
        return self._input_num

    @property
    def control_dependencies(self) -> List["Op"]:
        return self._control_dependencies

    @property
    def control_dependents(self) -> List["Op"]:
        return list(self._control_dependents)

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
        op._control_dependents.add(self)

    def remove_control_dependency(self, op: "Op"):
        assert op in self.control_dependencies
        self.control_dependencies.remove(op)
        op._control_dependents.remove(self)

    @property
    def input_ops(self) -> List["Op"]:
        return list(map(lambda tensor: tensor.op, self.input_tensors))

    def input_op(self, index: int) -> "Op":
        if not (0 <= index < len(self.input_tensors)):
            raise IndexError
        return self.input_tensors[index].op

    @property
    def input_ports(self) -> List["InputPort"]:
        return self.input_tensors.input_ports

    def input_port(self, index=0) -> "InputPort":
        if not (0 <= index < len(self.input_tensors)):
            raise IndexError
        return self.input_ports[index]

    @property
    def input_edges(self) -> List["Edge"]:
        return self.input_tensors.input_edges

    def input_edge(self, index=0) -> "Edge":
        if not (0 <= index < len(self.input_tensors)):
            raise IndexError
        return self.input_edges[index]

    @property
    def output_edges(self) -> List["Edge"]:
        return [
            edge
            for output_tensor in self.output_tensors
            for edge in output_tensor.output_edges
        ]

    @property
    def out_edges(self) -> List["Edge"]:
        ...

    def add_output_edge(self, output_index: int, dst_op: "Op", input_index: int):
        ...

    def input_index(self, input_op: "Op") -> int:
        for index, input in enumerate(self.input_tensors):
            if input.op == input_op:
                return index
        raise IndexError

    def has_name(self) -> bool:
        if OpAttrKey.has_name in self.attrs:
            return self.attrs[OpAttrKey.has_name]
        else:
            return "name" in self.attrs

    @property
    def name(self) -> str:
        assert self.has_name()
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

    @property
    def uuid(self) -> UUID:
        return self.attrs[OpAttrKey.uuid]

    def copy(self) -> "Op":
        """
        Return a shallow copy of the current op.
        Attribute values are not copied.
        They will be shared between these two ops.
        If you don't want to share attribute values, you can use `Op.deepcopy` instead.
        """
        op = Op(
            attrs=self.attrs.copy(),
            input_tensors=list(self.input_tensors),
            control_dependencies=list(self.control_dependencies),
            output_num=self.output_num,
        )
        for tensor, new_tensor in zip(self.output_tensors, op.output_tensors):
            new_tensor._attrs = tensor.attrs.copy()
        return op

    def __copy__(self):
        return self.copy()

    def deepcopy(self) -> "Op":
        """
        Return a deep copy of the current op.
        """
        op = Op(
            attrs=copy.deepcopy(self.attrs),
            input_tensors=list(self.input_tensors),
            control_dependencies=list(self.control_dependencies),
            output_num=self.output_num,
        )
        for tensor, new_tensor in zip(self.output_tensors, op.output_tensors):
            new_tensor._attrs = copy.deepcopy(tensor.attrs)
        return op

    def __deepcopy__(self, memodict={}):
        return self.deepcopy()

    def __repr__(self) -> str:
        attr_names = list(self.attrs.keys())
        attr_names.remove(OpAttrKey.uuid)
        if OpAttrKey.namespace in attr_names:
            attr_names.remove(OpAttrKey.namespace)
        if OpAttrKey.has_name in attr_names:
            attr_names.remove(OpAttrKey.has_name)
        attr_strings = []
        for name in ["name", "type"]:
            if name in attr_names:
                attr_strings.append(f"{name}={self.attrs[name]}")
                attr_names.remove(name)
        if len(attr_names) != 0:
            attr_names = np.array(attr_names)
            for name in attr_names[np.logical_not(np.char.startswith(attr_names, "/"))]:
                value_string = str(self.attrs[name])
                if "\n" not in value_string:
                    attr_strings.append(f"{name}={value_string}")
            for name in attr_names[np.char.startswith(attr_names, "/")]:
                attr_strings.append(f"{name}={self.attrs[name]}")
        attrs_string = ", ".join(attr_strings)
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


def create_op(
    type=None,
    attrs=None,
    input_tensors=None,
    control_dependencies=None,
    output_num: int = 1,
) -> Op:
    return Op(
        attrs=attrs,
        input_tensors=input_tensors,
        control_dependencies=control_dependencies,
        output_num=output_num,
    )


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
    _attrs: Attributes = field(default_factory=lambda: Attributes())
    _output_edges: Set["Edge"] = field(
        default_factory=lambda: typing.cast(Set[Edge], weakref.WeakSet())
    )

    @property
    def attrs(self) -> Attributes:
        return self._attrs

    @property
    def output_edges(self) -> List["Edge"]:
        return list(self._output_edges)

    @property
    def output_ops(self) -> List[Op]:
        return list(map(lambda edge: edge.dst_op, self.output_edges))

    @property
    def outputs(self) -> List[typing.Tuple[Op, int]]:
        return list(
            map(lambda edge: (edge.dst_op, edge.dst_input_index), self.output_edges)
        )

    def __hash__(self) -> int:
        return hash((self.op, self.output_index))


@dataclass(frozen=True)
class InputPort:
    op: Op
    input_index: int

    def is_control(self) -> bool:
        return self.input_index == ControlEdge.CONTROL_EDGE_INDEX


class Edge(ABC):
    def __init__(self, src_tensor: Tensor, dst_port: InputPort):
        self.src_op: Op = src_tensor.op
        self.src_output_index = src_tensor.output_index
        self.src_tensor: Tensor = src_tensor
        self.dst_op = dst_port.op
        self.dst_input_index = dst_port.input_index
        self.dst_port: InputPort = dst_port
        self._attrs: Attributes = Attributes()

    @abstractmethod
    def is_control_edge(self) -> bool:
        ...

    @property
    def attrs(self) -> Attributes:
        return self._attrs

    def remove(self):
        ...

    def replace_src(self, src: Op):
        ...

    @property
    def tensor_binding(self) -> Tensor:
        ...

    @tensor_binding.setter
    def tensor_binding(self, tensor: Tensor):
        ...

    @property
    def src(self) -> Op:
        ...

    @property
    def dst(self) -> Op:
        ...

    def insert_op(self, op: Op):
        new_edge1 = create_edge(self.src_op, op)
        new_edge1.src_output_index = self.src_output_index
        new_edge1.dst_input_index = 0
        new_edge2 = create_edge(op, self.dst_op)
        new_edge2.src_output_index = 0
        new_edge2.dst_input_index = self.dst_input_index
        self.remove()


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


def create_edge(src: Op, dst: Op) -> Edge:
    ...


def remove_edge(src: Op, dst: Op):
    ...


def connect(src: Op, dst: Op) -> Edge:
    ...


def disconnect(src: Op, dst: Op):
    ...


class Graph(NamespaceMixin):
    def __init__(self, ops=None, attrs=None):
        self._ops: immutables.Map[UUID, Op] = immutables.Map()
        self._attrs: Attributes = Attributes(attrs or {})
        self._name_to_op: immutables.Map[str, Op] = immutables.Map()
        self._composite_ops: immutables.Map[UUID, Op] = immutables.Map()
        self._cached_post_order_ops: List[Op] = None
        if ops is not None:
            self.add_ops(ops)

    @property
    def ops(self) -> List[Op]:
        """
        Return an unsorted list of ops that form this graph.
        """
        return list(self._ops.values())

    @property
    def attrs(self) -> Attributes:
        return self._attrs

    @property
    def names(self) -> List[str]:
        return list(self._name_to_op.keys())

    def update_attr(self, name: str, value: Any):
        self.attrs[name] = value

    def add_op(self, op: Op) -> None:
        assert op.uuid not in self._ops
        self._ops = self._ops.set(op.uuid, op)
        if op.has_name():
            assert op.name not in self._name_to_op
            self._name_to_op = self._name_to_op.set(op.name, op)
        if isinstance(op, CompositeOp):
            self._composite_ops = self._composite_ops.set(op.uuid, op)
        self.invalidate_cache()

    def add_ops(self, ops: Iterable[Op]) -> None:
        with self._ops.mutate() as _ops:
            with self._name_to_op.mutate() as _name_to_op:
                with self._composite_ops.mutate() as _composite_ops:
                    for op in ops:
                        assert op.uuid not in self._ops
                        _ops[op.uuid] = op
                        if op.has_name():
                            assert op.name not in self._name_to_op
                            _name_to_op[op.name] = op
                        if isinstance(op, CompositeOp):
                            _composite_ops[op.uuid] = op
                    self._ops = _ops.finish()
                    self._name_to_op = _name_to_op.finish()
                    self._composite_ops = _composite_ops.finish()
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
        assert op.uuid in self._ops
        self._ops = self._ops.delete(op.uuid)
        if op.has_name():
            self._name_to_op = self._name_to_op.delete(op.name)
        if isinstance(op, CompositeOp):
            self._composite_ops = self._composite_ops.delete(op.uuid)
        self.invalidate_cache()

    def get_op_by_name(self, name: str) -> Optional[Op]:
        return self._name_to_op.get(name)

    def replace_tensor(self, old_tensor: Tensor, new_tensor: Tensor):
        for op in self.ops:
            if old_tensor in op.input_tensors:
                op.input_tensors[op.input_index(old_tensor.op)] = new_tensor

    @property
    def sorted_ops(self) -> List[Op]:
        """
        Return a topologically sorted list of ops that forms this graph.
        """
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
        dfs_stack = list(output_ops)
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
        return [
            edge for op in self.ops for edge in op.input_edges if edge.src_op in self
        ]

    @property
    def control_edges(self) -> List[Edge]:
        return [
            ControlEdge(src=src, dst=op)
            for op in self.ops
            for src in op.control_dependencies
            if src in self
        ]

    @property
    def edges(self) -> List[Edge]:
        return self.data_edges + self.control_edges

    def data_edges_from_tensor(self, tensor: Tensor) -> List[Edge]:
        return [edge for edge in tensor.output_edges if edge.dst_op in self]

    def control_edges_from_op(self, op: Op) -> List[Edge]:
        return [
            ControlEdge(src=op, dst=dst) for dst in op.control_dependents if dst in self
        ]

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
        if op.uuid in self._ops:
            return True
        for composite_op in self._composite_ops:
            if op in composite_op.graph:
                return True
        return False

    def invalidate_cache(self):
        self._cached_post_order_ops = None

    def to_namespace(
        self, namespace: typing.Union[Namespace, str], registry: Registry = None
    ) -> "Graph":
        if isinstance(namespace, str):
            namespace = Namespace(namespace)
        if namespace == self.namespace:
            return self
        else:
            registry = registry or get_global_registry()
            return registry.get_mapper(self.namespace, namespace).map(self, namespace)

    def to_default_namespace(self) -> "Graph":
        return self.to_namespace(default_namespace())

    def copy(self) -> "Graph":
        """
        Return a shallow copy of the current graph.
        Attribute values of graph and ops are not copied.
        They will be shared between these two graphs.
        If you don't want to share attribute values,
        you can use `Graph.deepcopy` instead.
        """
        new_graph = Graph(attrs=self.attrs.copy())
        uuid_to_new_op: Dict[uuid.UUID, Op] = {}
        for op in self.sorted_ops:
            target_op = Op(
                attrs=op.attrs.copy(),
                input_tensors=[
                    uuid_to_new_op[input_tensor.op.uuid].output_tensor(
                        input_tensor.output_index
                    )
                    for input_tensor in op.input_tensors
                ],
                control_dependencies=[
                    uuid_to_new_op[control_dependency.uuid]
                    for control_dependency in op.control_dependencies
                ],
                output_num=op.output_num,
            )
            for tensor, new_tensor in zip(op.output_tensors, target_op.output_tensors):
                new_tensor._attrs = tensor.attrs.copy()
            new_graph.add_op(target_op)
            uuid_to_new_op[op.uuid] = target_op
        return new_graph

    def __copy__(self):
        return self.copy()

    def deepcopy(self) -> "Graph":
        """
        Return a deep copy of the current graph.
        """
        new_graph = Graph(attrs=copy.deepcopy(self.attrs))
        for op in self.sorted_ops:
            target_op = Op(
                attrs=copy.deepcopy(op.attrs),
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
            for tensor, new_tensor in zip(op.output_tensors, target_op.output_tensors):
                new_tensor._attrs = copy.deepcopy(tensor.attrs)
            new_graph.add_op(target_op)
        return new_graph

    def __deepcopy__(self, memodict={}):
        return self.deepcopy()

    def dict(self):
        return dict(ops=[op.dict() for op in self.ops], attrs=dict(self.attrs),)

    def json(self):
        return json.dumps(self.dict(), indent=4)

    def dump_to_str(self) -> str:
        uuid_to_id = dict(zip([op.uuid for op in self.sorted_ops], itertools.count(0)))
        op_strs = []
        for op in self.sorted_ops:
            output_op_ids_ = [
                uuid_to_id[edge.dst_op.uuid]
                for tensor in op.output_tensors
                for edge in self.data_edges_from_tensor(tensor)
            ]
            op_strs.append(f"{uuid_to_id[op.uuid]}: {op} -> {output_op_ids_}")
        return "\n".join(op_strs)

    def print(self) -> None:
        print(self.dump_to_str())
