from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from mmdnn.conversion.common.IR import graph_pb2

from amanda import DataType
from amanda.conversion.utils import to_proto, without_internal_attrs
from amanda.exception import MismatchNamespaceError
from amanda.graph import Graph, create_graph, create_op
from amanda.io import (
    ProtoToBytesSerde,
    ProtoToDictSerde,
    SerdeDispatcher,
    get_serde_registry,
)
from amanda.namespace import Namespace, default_namespace

_namespace = default_namespace() / Namespace("mmdnn")
_internal_namespace = _namespace / Namespace("internal")
_type_namespace = Namespace("mmdnn")


def mmdnn_namespace() -> Namespace:
    return _namespace


def mmdnn_internal_namespace() -> Namespace:
    return _internal_namespace


def mmdnn_type_namespace() -> Namespace:
    return _type_namespace


def mmdnn_dtype(name: str):
    return DataType(mmdnn_type_namespace(), name)


dtype_enum_to_name = {
    graph_pb2.DT_UNDEFINED: "undefined",
    graph_pb2.DT_INT8: "int8",
    graph_pb2.DT_INT16: "int16",
    graph_pb2.DT_INT32: "int32",
    graph_pb2.DT_INT64: "int64",
    graph_pb2.DT_UINT8: "uint8",
    graph_pb2.DT_UINT16: "uint16",
    graph_pb2.DT_UINT32: "uint32",
    graph_pb2.DT_UINT64: "uint64",
    graph_pb2.DT_FLOAT16: "float16",
    graph_pb2.DT_FLOAT32: "float32",
    graph_pb2.DT_FLOAT64: "float64",
    graph_pb2.DT_COMPLEX64: "complex64",
    graph_pb2.DT_COMPLEX128: "complex128",
    graph_pb2.DT_BOOL: "bool",
    graph_pb2.DT_STRING: "string",
}

dtype_name_to_enum = {name: enum for enum, name in dtype_enum_to_name.items()}


def serialize_type(type: int) -> DataType:
    return mmdnn_dtype(dtype_enum_to_name[type])


def deserialize_type(dtype: DataType) -> int:
    return dtype_name_to_enum[dtype.name]


@dataclass
class MmdnnSerdeDispatcher(SerdeDispatcher):
    def __post_init__(self):
        for proto_type in [
            graph_pb2.TensorShape,
        ]:
            serde = ProtoToDictSerde(
                proto_type,
                name=proto_type.__name__,
                namespace=mmdnn_type_namespace(),
            )
            self.register_type(proto_type, serde)
            self.register_dtype_name(proto_type.__name__, serde)
        for proto_type in [
            graph_pb2.LiteralTensor,
        ]:
            serde = ProtoToBytesSerde(
                proto_type,
                name=proto_type.__name__,
                namespace=mmdnn_type_namespace(),
            )
            self.register_type(proto_type, serde)
            self.register_dtype_name(proto_type.__name__, serde)


get_serde_registry().register_namespace(mmdnn_type_namespace(), MmdnnSerdeDispatcher())


@dataclass(frozen=True)
class MmdnnTensor:
    op: str
    output_index: int


def import_from_graph_def(
    graph_def: Union[graph_pb2.GraphDef, str, bytes, Path]
) -> Graph:
    graph_def = to_proto(graph_def, graph_pb2.GraphDef)
    graph = create_graph(namespace=mmdnn_namespace())
    name_to_node = {node.name: node for node in graph_def.node}

    def add_op(node):
        if graph.get_op(node.name) is not None:
            return
        input_tensors: List[MmdnnTensor] = []
        input: str
        for input in node.input:
            names = input.split(":")
            assert len(names) == 1 or len(names) == 2
            if len(names) == 1:
                input_tensors.append(MmdnnTensor(names[0], 0))
            else:
                input_tensors.append(MmdnnTensor(names[0], int(names[1])))
        for input_tensor in input_tensors:
            add_op(name_to_node[input_tensor.op])
        op = create_op(
            name=node.name,
            type=node.op,
            inputs=len(input_tensors),
            outputs=1,
        )
        #  add attrs into op
        for key in node.attr:
            ir_attr_value = node.attr[key]
            field_set = ir_attr_value.WhichOneof("value")
            if field_set == "type":
                op.attrs[key] = serialize_type(ir_attr_value.type)
            elif field_set == "shape":
                op.attrs[key] = ir_attr_value.shape
            elif field_set == "tensor":
                op.attrs[key] = ir_attr_value.tensor
            elif field_set == "s":
                op.attrs[key] = ir_attr_value.s
            elif field_set == "b":
                op.attrs[key] = ir_attr_value.b
            elif field_set == "i":
                op.attrs[key] = ir_attr_value.i
            elif field_set == "f":
                op.attrs[key] = ir_attr_value.f
            elif field_set == "list":
                ir_list = ir_attr_value.list
                attr_value_list = []
                for value in ir_list.s:
                    attr_value_list.append(value)
                for value in ir_list.b:
                    attr_value_list.append(value)
                for value in ir_list.i:
                    attr_value_list.append(value)
                for value in ir_list.f:
                    attr_value_list.append(value)
                for value in ir_list.type:
                    attr_value_list.append(serialize_type(value))
                for value in ir_list.shape:
                    attr_value_list.append(value)
                for value in ir_list.tensor:
                    attr_value_list.append(value)
                op.attrs[key] = attr_value_list
            else:
                raise ValueError("unknown field met")
        graph.add_op(op)
        for index, input_tensor in enumerate(input_tensors):
            graph.create_edge(
                graph.get_op(input_tensor.op).output_port(input_tensor.output_index),
                op.input_port(index),
            )

    for ir_node in graph_def.node:
        add_op(ir_node)

    return graph


def export_to_graph_def(graph: Graph) -> graph_pb2.GraphDef:
    if not graph.namespace.belong_to(mmdnn_namespace()):
        raise MismatchNamespaceError(expect=mmdnn_namespace(), actual=graph.namespace)
    graph_def = graph_pb2.GraphDef()
    for op in graph.ops:
        node = graph_def.node.add()
        node.op = op.type
        node.name = op.name
        for port in op.input_ports:
            src_port = port.in_edges[0].src
            if src_port.name == "0":
                port_string = src_port.op.name
            else:
                port_string = f"{src_port.op.name}:{src_port.name}"
            node.input.append(port_string)
        for key in without_internal_attrs(op.attrs):
            ir_value = node.attr[key]
            value = op.attrs[key]
            if type(value) == bytes:
                ir_value.s = value
            elif type(value) == bool:
                ir_value.b = value
            # bool is a subclass of int, so we should check bool before checking int
            elif type(value) == int:
                ir_value.i = value
            elif type(value) == float:
                ir_value.f = value
            elif type(value) == DataType:
                ir_value.type = deserialize_type(value)
            elif type(value) == graph_pb2.TensorShape:
                ir_value.shape.CopyFrom(value)
            elif type(value) == graph_pb2.LiteralTensor:
                ir_value.tensor.CopyFrom(value)
            elif type(value) == list:
                ir_value.list.SetInParent()
                if len(value) > 0:
                    elem = value[0]
                    if type(elem) == bytes:
                        ir_value.list.s.extend(value)
                    # bool is a subclass of int,
                    # so we should check bool before checking int
                    elif type(elem) == bool:
                        ir_value.list.b.extend(value)
                    elif type(elem) == int:
                        ir_value.list.i.extend(value)
                    elif type(elem) == float:
                        ir_value.list.f.extend(value)
                    elif type(elem) == DataType:
                        for elem_ in value:
                            ir_value.list.type.append(deserialize_type(elem_))
                    elif type(elem) == graph_pb2.TensorShape:
                        for elem_ in value:
                            ir_value.list.shape.append(elem_)
                    elif type(elem) == graph_pb2.LiteralTensor:
                        for elem_ in value:
                            ir_value.list.tensor.append(elem_)

    return graph_def


import_types = {
    "mmdnn": import_from_graph_def,
}

export_types = {
    "mmdnn": export_to_graph_def,
}
