import builtins
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type, Union, cast

import numpy as np
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.message import Message

from amanda.graph import Edge, Graph, InputPort, IoPort, Op, OutputPort, SubGraph
from amanda.namespace import Namespace, default_namespace
from amanda.type import DataType, unknown_type


@dataclass
class SerdeContext:
    dtype: DataType = None
    root: Graph = None


class Serde(ABC):
    @abstractmethod
    def serialize_type(self, type: Any) -> DataType:
        ...

    @abstractmethod
    def deserialize_type(self, dtype: DataType) -> Any:
        ...

    @abstractmethod
    def serialize(self, value: Any) -> Any:
        ...

    @abstractmethod
    def deserialize(self, value: Any, context: SerdeContext) -> Any:
        ...


class TypeSerde(Serde, ABC):
    def serialize(self, value: Any) -> Any:
        raise NotImplementedError()

    def deserialize(self, value: Any, context: SerdeContext) -> Any:
        raise NotImplementedError()


@dataclass
class SerdeRegistry:
    _type_to_namespace: Dict[Any, Namespace] = field(default_factory=dict)
    _meta_type_to_namespace: Dict[Any, Namespace] = field(default_factory=dict)
    _namespace_to_serde: Dict[Namespace, Serde] = field(default_factory=dict)

    def register_type(self, type: Any, namespace: Namespace) -> None:
        self._type_to_namespace[type] = namespace

    def register_meta_type(self, type: Any, namespace: Namespace) -> None:
        self._meta_type_to_namespace[type] = namespace

    def register_namespace(self, namespace: Namespace, serde: Serde) -> None:
        self._namespace_to_serde[namespace] = serde
        if isinstance(serde, SerdeDispatcher):
            serde.parent = self
            for type in serde._type_to_serde:
                self.register_type(type, namespace)
            for type in serde._meta_type_to_serde:
                self.register_meta_type(type, namespace)

    def dispatch_type(self, type: Any) -> Serde:
        if not isinstance(type, builtins.type):
            if builtins.type(type) in self._meta_type_to_namespace:
                namespace = self._meta_type_to_namespace[builtins.type(type)]
                return self.dispatch_namespace(namespace)
        if type not in self._type_to_namespace:
            raise RuntimeError(
                f"type {type.__module__}.{type.__name__} is unregistered"
            )
        namespace = self._type_to_namespace[type]
        return self.dispatch_namespace(namespace)

    def dispatch_namespace(self, namespace: Namespace) -> Serde:
        if namespace not in self._namespace_to_serde:
            raise RuntimeError(f"namespace {namespace.full_name} is unregistered")
        return self._namespace_to_serde[namespace]

    def serialize_type(self, type: Any) -> DataType:
        return self.dispatch_type(type).serialize_type(type)

    def deserialize_type(self, dtype: DataType) -> Any:
        return self.dispatch_namespace(dtype.namespace).deserialize_type(dtype)

    def serialize(self, value: Any) -> Any:
        return self.dispatch_type(type(value)).serialize(value)

    def deserialize(self, value: Any, context: SerdeContext) -> Any:
        return self.dispatch_namespace(context.dtype.namespace).deserialize(
            value, context
        )


_registry = SerdeRegistry()


def get_serde_registry() -> SerdeRegistry:
    return _registry


def serialize_type(type: Any) -> DataType:
    return get_serde_registry().serialize_type(type)


def deserialize_type(dtype: DataType) -> Any:
    return get_serde_registry().deserialize_type(dtype)


def serialize(value: Any) -> Any:
    return get_serde_registry().serialize(value)


def deserialize(value: Any, context: SerdeContext) -> Any:
    return get_serde_registry().deserialize(value, context)


@dataclass
class SerdeDispatcher(Serde):
    _type_to_serde: Dict[Any, Serde] = field(default_factory=dict)
    _meta_type_to_serde: Dict[Any, Serde] = field(default_factory=dict)
    _dtype_name_to_serde: Dict[str, Serde] = field(default_factory=dict)
    parent: Union[SerdeRegistry, "SerdeDispatcher"] = None
    namespace: Namespace = None

    def search_namespace(self):
        if self.namespace is None:
            for namespace, serde in self.parent._namespace_to_serde.items():
                if serde == self:
                    self.namespace = namespace
                    return

    def register_type(self, type: Any, serde: Serde) -> None:
        self._type_to_serde[type] = serde
        if self.parent is not None:
            if isinstance(self.parent, SerdeRegistry):
                self.search_namespace()
                self.parent.register_type(type, self.namespace)
            else:
                self.parent.register_type(type, serde)
        if isinstance(serde, SerdeDispatcher):
            serde.parent = self

    def register_meta_type(self, type: Any, serde: Serde) -> None:
        self._meta_type_to_serde[type] = serde
        if self.parent is not None:
            if isinstance(self.parent, SerdeRegistry):
                self.search_namespace()
                self.parent.register_meta_type(type, self.namespace)
            else:
                self.parent.register_meta_type(type, serde)
        if isinstance(serde, SerdeDispatcher):
            serde.parent = self

    def register_dtype_name(self, dtype_name: str, serde: Serde) -> None:
        self._dtype_name_to_serde[dtype_name] = serde
        if self.parent is not None and isinstance(self.parent, SerdeDispatcher):
            self.parent.register_dtype_name(dtype_name, serde)
        if isinstance(serde, SerdeDispatcher):
            serde.parent = self

    def dispatch_type(self, type: Any) -> Serde:
        if not isinstance(type, builtins.type):
            if builtins.type(type) in self._meta_type_to_serde:
                return self._meta_type_to_serde[builtins.type(type)]
        if type not in self._type_to_serde:
            raise RuntimeError(
                f"type {type.__module__}.{type.__name__} is unregistered"
            )
        return self._type_to_serde[type]

    def dispatch_dtype(self, dtype: DataType) -> Serde:
        if dtype.name not in self._dtype_name_to_serde:
            raise RuntimeError(
                f"data type {dtype.namespace.full_name}.{dtype.name} is unregistered"
            )
        return self._dtype_name_to_serde[dtype.name]

    def serialize_type(self, type: Any) -> DataType:
        return self.dispatch_type(type).serialize_type(type)

    def deserialize_type(self, dtype: DataType) -> Any:
        return self.dispatch_dtype(dtype).deserialize_type(dtype)

    def serialize(self, value: Any) -> Any:
        return self.dispatch_type(type(value)).serialize(value)

    def deserialize(self, value: Any, context: SerdeContext) -> Any:
        return self.dispatch_dtype(context.dtype).deserialize(value, context)


@dataclass
class ProtoSerde(TypeSerde):
    proto_class: Type[Message]
    name: str = None
    namespace: Namespace = Namespace("proto")

    def __post_init__(self):
        if self.name is None:
            self.name = self.proto_class.DESCRIPTOR.full_name

    def serialize_type(self, type: Any) -> DataType:
        assert type == self.proto_class
        return DataType(
            namespace=self.namespace,
            name=self.name,
        )

    def deserialize_type(self, dtype: DataType) -> Any:
        assert dtype.namespace == self.namespace and dtype.name == self.name
        return self.proto_class


class ProtoToBytesSerde(ProtoSerde):
    def serialize(self, value: Any) -> Any:
        assert type(value) == self.proto_class
        return value.SerializeToString()

    def deserialize(self, value: Any, context: SerdeContext) -> Any:
        assert (
            context.dtype.namespace == self.namespace
            and context.dtype.name == self.name
        )
        proto = self.proto_class()
        proto.ParseFromString(value)
        return proto


class ProtoToDictSerde(ProtoSerde):
    def serialize(self, value: Any) -> Any:
        assert type(value) == self.proto_class
        return MessageToDict(value, preserving_proto_field_name=True)

    def deserialize(self, value: Any, context: SerdeContext) -> Any:
        assert (
            context.dtype.namespace == self.namespace
            and context.dtype.name == self.name
        )
        proto = self.proto_class()
        ParseDict(value, proto)
        return proto


_default_dispatcher = SerdeDispatcher()


def get_default_dispatcher() -> SerdeDispatcher:
    return _default_dispatcher


class UnknownTypeSerde(TypeSerde):
    def serialize_type(self, type: Any) -> DataType:
        return unknown_type

    def deserialize_type(self, dtype: DataType) -> Any:
        return None


_unknown_type_serde = UnknownTypeSerde()

get_default_dispatcher().register_type(None, _unknown_type_serde)
get_default_dispatcher().register_dtype_name(unknown_type.name, _unknown_type_serde)
get_serde_registry().register_namespace(default_namespace(), get_default_dispatcher())

_graph_dispatcher = SerdeDispatcher()


def get_graph_dispatcher() -> SerdeDispatcher:
    return _graph_dispatcher


_graph_namespace = Namespace("graph")


def graph_namespace() -> Namespace:
    return _graph_namespace


@dataclass
class NodeSerde(Serde):
    node_class: Type

    def serialize_type(self, type: Any) -> DataType:
        return DataType(graph_namespace(), self.node_class.__name__)

    def deserialize_type(self, dtype: DataType) -> Any:
        assert self.node_class.__name__ == dtype.name
        return self.node_class

    def serialize(self, value: Any) -> Any:
        return value.path

    def deserialize(self, value: Any, context: SerdeContext) -> Any:
        def from_path(path: List[str], node: Union[Graph, Op]) -> Union[Graph, Op]:
            if len(path) == 0:
                return node
            else:
                return from_path(path[1:], cast(Graph, node).get_op(path[0]))

        assert self.node_class.__name__ == context.dtype.name
        return from_path(value, context.root)


@dataclass
class PortSerde(Serde):
    port_class: Type

    def serialize_type(self, type: Any) -> DataType:
        return DataType(graph_namespace(), self.port_class.__name__)

    def deserialize_type(self, dtype: DataType) -> Any:
        assert self.port_class.__name__ == dtype.name
        return self.port_class

    def serialize(self, value: Any) -> Any:
        op = value.op
        name = value.name
        if value in op.input_ports or value == op.control_input_port:
            return {
                "op": serialize(op),
                "input_port": name,
            }
        else:
            assert value in op.output_ports or value == op.control_output_port
            return {
                "op": serialize(op),
                "output_port": name,
            }

    def deserialize(self, value: Any, context: SerdeContext) -> Any:
        assert self.port_class.__name__ == context.dtype.name
        context.dtype.name = "Op"
        op = deserialize(value["op"], context)
        context.dtype.name = self.port_class.__name__
        if "input_port" in value:
            return op.input_port(value["input_port"])
        else:
            return op.output_port(value["output_port"])


@dataclass
class EdgeSerde(Serde):
    def serialize_type(self, type: Any) -> DataType:
        return DataType(graph_namespace(), "Edge")

    def deserialize_type(self, dtype: DataType) -> Any:
        assert dtype.name == "Edge"
        return Edge

    def serialize(self, value: Any) -> Any:
        return {
            "graph": serialize(value.graph),
            "src": serialize(value.src),
            "dst": serialize(value.dst),
        }

    def deserialize(self, value: Any, context: SerdeContext) -> Any:
        assert context.dtype.name == "Edge"
        context.dtype.name = "Graph"
        graph = deserialize(value["graph"], context)
        context.dtype.name = "OutputPort"
        src = deserialize(value["src"], context)
        context.dtype.name = "InputPort"
        dst = deserialize(value["dst"], context)
        context.dtype.name = "Edge"
        return graph.get_edge(src, dst)


get_graph_dispatcher().register_type(Op, NodeSerde(Op))
get_graph_dispatcher().register_dtype_name("Op", NodeSerde(Op))
get_graph_dispatcher().register_type(Graph, NodeSerde(Graph))
get_graph_dispatcher().register_dtype_name("Graph", NodeSerde(Graph))
get_graph_dispatcher().register_type(SubGraph, NodeSerde(SubGraph))
get_graph_dispatcher().register_dtype_name("SubGraph", NodeSerde(SubGraph))
get_graph_dispatcher().register_type(InputPort, PortSerde(InputPort))
get_graph_dispatcher().register_dtype_name("InputPort", PortSerde(InputPort))
get_graph_dispatcher().register_type(OutputPort, PortSerde(OutputPort))
get_graph_dispatcher().register_dtype_name("OutputPort", PortSerde(OutputPort))
get_graph_dispatcher().register_type(IoPort, PortSerde(IoPort))
get_graph_dispatcher().register_dtype_name("IoPort", PortSerde(IoPort))
get_graph_dispatcher().register_type(Edge, EdgeSerde())
get_graph_dispatcher().register_dtype_name("Edge", EdgeSerde())
get_serde_registry().register_namespace(graph_namespace(), get_graph_dispatcher())


_numpy_dispatcher = SerdeDispatcher()


def get_numpy_dispatcher() -> SerdeDispatcher:
    return _numpy_dispatcher


class NpArraySerde(Serde):
    def serialize_type(self, type: Any) -> DataType:
        return DataType(Namespace("np"), "ndarray")

    def deserialize_type(self, dtype: DataType) -> Any:
        return np.ndarray

    def serialize(self, value: Any) -> Any:
        return {
            "dtype": value.dtype.name,
            "shape": list(value.shape),
            "bytes": value.tobytes(),
        }

    def deserialize(self, value: Any, context: SerdeContext) -> Any:
        return np.frombuffer(value["bytes"], dtype=value["dtype"]).reshape(
            value["shape"]
        )


get_numpy_dispatcher().register_type(np.ndarray, NpArraySerde())
get_numpy_dispatcher().register_dtype_name("ndarray", NpArraySerde())
get_serde_registry().register_namespace(Namespace("np"), get_numpy_dispatcher())
