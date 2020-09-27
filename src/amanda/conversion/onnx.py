from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Set, Union, cast

import onnx
from onnx import defs

from amanda.attributes import Attributes
from amanda.conversion.utils import to_proto, without_internal_attrs
from amanda.exception import MismatchNamespaceError
from amanda.graph import (
    Graph,
    InputPort,
    OutputPort,
    SubGraph,
    create_op,
    create_subgraph,
)
from amanda.io.serde import (
    ProtoToBytesSerde,
    ProtoToDictSerde,
    SerdeContext,
    SerdeDispatcher,
    TypeSerde,
    deserialize_type,
    get_serde_registry,
    serialize_type,
)
from amanda.namespace import Namespace, default_namespace
from amanda.type import DataType

_namespace = default_namespace() / Namespace("onnx")
_internal_namespace = _namespace / Namespace("internal")
_type_namespace = Namespace("onnx")


def onnx_namespace() -> Namespace:
    return _namespace


def onnx_internal_namespace() -> Namespace:
    return _internal_namespace


def onnx_type_namespace() -> Namespace:
    return _type_namespace


INT_TO_STR_DTYPE = {
    1: "float",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "double",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
}

STR_TO_INT_DTYPE = {value: key for key, value in INT_TO_STR_DTYPE.items()}


def to_dtype_str(dtype: int) -> str:
    return INT_TO_STR_DTYPE[dtype]


def to_dtype_enum(dtype: str) -> int:
    return STR_TO_INT_DTYPE[dtype]


def onnx_dtype(name: str, attrs: Dict[str, Any] = None) -> DataType:
    return DataType(
        namespace=onnx_type_namespace(), name=name, attrs=Attributes(attrs or {})
    )


class OnnxTypeSerde(TypeSerde):
    def __init__(self):
        self.dtype = onnx_dtype("Type")
        self.serde = ProtoToDictSerde(
            onnx.TypeProto, self.dtype.name, onnx_type_namespace()
        )

    def serialize_type(self, type: Any) -> DataType:
        field_name = type.WhichOneof("value")
        field_to_name = {
            "tensor_type": "Tensor",
            "sequence_type": "Sequence",
            "map_type": "Map",
        }
        name = field_to_name[field_name]
        return onnx_dtype(name, attrs=self.serde.serialize(type)[field_name])

    def deserialize_type(self, dtype: DataType) -> Any:
        name_to_field = {
            "Tensor": "tensor_type",
            "Sequence": "sequence_type",
            "Map": "map_type",
        }
        field_name = name_to_field[dtype.name]
        return self.serde.deserialize(
            {field_name: dtype.attrs}, SerdeContext(dtype=self.dtype)
        )


@dataclass
class OnnxSerdeDispatcher(SerdeDispatcher):
    def __post_init__(self):
        type_serde = OnnxTypeSerde()
        self.register_meta_type(onnx.TypeProto, type_serde)
        for name in [
            "Tensor",
            "Sequence",
            "Map",
        ]:
            self.register_dtype_name(name, type_serde)
        for proto_type in [
            onnx.OperatorSetIdProto,
        ]:
            serde = ProtoToDictSerde(
                proto_type,
                name=proto_type.__name__,
                namespace=onnx_type_namespace(),
            )
            self.register_type(proto_type, serde)
            self.register_dtype_name(proto_type.__name__, serde)
        for proto_type in [
            onnx.TensorProto,
        ]:
            serde = ProtoToBytesSerde(
                proto_type,
                name=proto_type.__name__,
                namespace=onnx_type_namespace(),
            )
            self.register_type(proto_type, serde)
            self.register_dtype_name(proto_type.__name__, serde)


_serde_dispatcher = OnnxSerdeDispatcher()

get_serde_registry().register_namespace(onnx_type_namespace(), _serde_dispatcher)


def to_proto_dict(dtype: DataType) -> Dict[str, Any]:
    name_to_field = {
        "Tensor": "tensor_type",
        "Sequence": "sequence_type",
        "Map": "map_type",
    }
    return {name_to_field[dtype.name]: dtype.attrs}


def from_type_str(type_str: Union[Set[str], str]) -> DataType:
    if isinstance(type_str, set):
        types = [from_type_str(type) for type in type_str]
        if len(types) == 1:
            return types[0]
        else:
            return onnx_dtype("Union", {"types": types})
    if type_str.startswith("tensor"):
        return onnx_dtype(
            "Tensor",
            attrs={"elem_type": to_dtype_enum(type_str[len("tensor") + 1 : -1])},
        )
    if type_str.startswith("seq"):
        return onnx_dtype(
            "Sequence",
            attrs={
                "elem_type": to_proto_dict(from_type_str(type_str[len("seq") + 1 : -1]))
            },
        )
    if type_str.startswith("map"):
        key_type, value_type = type_str[len("map") + 1 : -1].split(", ")
        return onnx_dtype(
            "Map",
            attrs={
                "key_type": to_dtype_enum(key_type),
                "value_type": to_proto_dict(from_type_str(value_type)),
            },
        )
    return onnx_dtype("Tensor", attrs={"elem_type": to_dtype_enum(type_str)})


def import_from_model_def(
    model_def: Union[onnx.ModelProto, str, bytes, Path]
) -> SubGraph:
    model_def = to_proto(model_def, onnx.ModelProto)
    graph_def = model_def.graph
    graph = import_from_graph_def(graph_def)
    for key in [
        "ir_version",
        "producer_name",
        "producer_version",
        "domain",
        "model_version",
        "doc_string",
    ]:
        if model_def.HasField(key):
            graph.attrs[key] = getattr(model_def, key)
    for key in ["opset_import", "metadata_props"]:
        graph.attrs[key] = list(getattr(model_def, key))
    return graph


def import_from_graph_def(
    graph_def: Union[onnx.GraphProto, str, bytes, Path]
) -> SubGraph:
    graph_def = to_proto(graph_def, onnx.GraphProto)
    initializers = {initializer.name for initializer in graph_def.initializer}
    sparse_initializers = {
        initializer.values.name for initializer in graph_def.sparse_initializer
    }
    input_defs: Dict[str, onnx.ValueInfoProto] = {}
    graph = create_subgraph(
        name=getattr(graph_def, "name") if graph_def.HasField("name") else None,
        namespace=onnx_namespace(),
        inputs=OrderedDict(
            (input_def.name, serialize_type(input_def.type))
            for input_def in graph_def.input
            if (input_def.name not in initializers)
            or (input_def.name not in sparse_initializers)
        ),
        outputs=OrderedDict(
            (output_def.name, serialize_type(output_def.type))
            for output_def in graph_def.output
        ),
    )
    name_to_output_port: Dict[str, OutputPort] = {}

    def add_initializer(graph: Graph, initializer: onnx.TensorProto):
        input_def = input_defs[initializer.name]
        op = create_op(
            name=initializer.name,
            type="Initializer",
            inputs=[],
            outputs=OrderedDict(value=serialize_type(input_def.type)),
        )
        op.attrs["value"] = initializer
        if input_def.HasField("doc_string"):
            op.attrs["doc_string"] = input_def.doc_string
        graph.add_op(op)
        name_to_output_port[op.name] = op.output_port(0)

    def add_sparse_initializer(graph: Graph, initializer: onnx.SparseTensorProto):
        input_def = input_defs[initializer.values.name]
        op = create_op(
            name=initializer.values.name,
            type="SparseInitializer",
            inputs=[],
            outputs=OrderedDict(value=serialize_type(input_def.type)),
        )
        op.attrs["value"] = initializer
        if input_def.HasField("doc_string"):
            op.attrs["doc_string"] = input_def.doc_string
        graph.add_op(op)
        name_to_output_port[op.name] = op.output_port(0)

    def add_op(node: onnx.NodeProto):
        if graph.get_op(node.name) is not None:
            raise RuntimeError(f"{node.name} have been added")
        attrs = {}
        for attr in node.attribute:
            attr_name = attr.name
            attrs[attr_name] = from_attr_proto(attr)
            if attr.HasField("doc_string"):
                attrs[f"{attr_name}/doc_string"] = attr.doc_string
        domain = node.domain if node.HasField("domain") else ""
        schema = defs.get_schema(node.op_type, domain)
        op = create_op(
            name=getattr(node, "name") if node.HasField("name") else None,
            type=node.op_type if domain == "" else f"{domain}/{node.op_type}",
            namespace=onnx_namespace(),
            attrs=attrs,
            inputs=OrderedDict(
                (input.name, from_type_str(input.types)) for input in schema.inputs
            ),
            outputs=OrderedDict(
                (output.name, from_type_str(output.types)) for output in schema.outputs
            ),
        )
        if node.HasField("doc_string"):
            op.attrs["doc_string"] = node.doc_string
        graph.add_op(op)
        for dst_port, input_name in zip(op.input_ports, node.input):
            graph.create_edge(name_to_output_port[input_name], dst_port)
        for port, name in zip(op.output_ports, node.output):
            name_to_output_port[name] = port
            if name in graph.name_to_output_port:
                graph.create_edge(port, cast(InputPort, graph.output_port(name)))
        if op.output_num > 1 or (
            op.output_num == 1 and list(node.output)[0] == op.name
        ):
            for port, name in zip(op.output_ports, node.output):
                op.attrs[f"output_port/{port.name}/value_name"] = name

    if graph_def.HasField("doc_string"):
        graph.attrs["doc_string"] = graph_def.doc_string
    for key in [
        "value_info",
        "quantization_annotation",
    ]:
        graph.attrs[key] = list(getattr(graph_def, key))
    for input_def in graph_def.input:
        if (input_def.name not in initializers) and (
            input_def.name not in sparse_initializers
        ):
            if input_def.HasField("doc_string"):
                graph.attrs[
                    f"input_port/{input_def.name}/doc_string"
                ] = input_def.doc_string
            name_to_output_port[input_def.name] = cast(
                OutputPort, graph.input_port(input_def.name)
            )
        else:
            input_defs[input_def.name] = input_def
    for output_def in graph_def.output:
        if output_def.HasField("doc_string"):
            graph.attrs[
                f"output_port/{output_def.name}/doc_string"
            ] = output_def.doc_string
    for initializer in graph_def.initializer:
        add_initializer(graph, initializer)
    for initializer in graph_def.sparse_initializer:
        add_sparse_initializer(graph, initializer)
    for node in graph_def.node:
        add_op(node)
    return graph


def export_to_model_def(
    graph: SubGraph, file: Union[str, Path] = None
) -> onnx.ModelProto:
    graph_def = export_to_graph_def(graph)
    model_def = onnx.ModelProto()
    model_def.graph.CopyFrom(graph_def)
    for key in [
        "ir_version",
        "producer_name",
        "producer_version",
        "domain",
        "model_version",
        "doc_string",
    ]:
        if key in graph.attrs:
            setattr(model_def, key, graph.attrs[key])
    for key in ["opset_import", "metadata_props"]:
        if len(graph.attrs[key]) != 0:
            getattr(model_def, key).extend(graph.attrs[key])
    if file is not None:
        onnx.save(model_def, str(file))
    return model_def


def export_to_graph_def(
    graph: SubGraph, file: Union[str, Path] = None
) -> onnx.GraphProto:
    if not graph.namespace.belong_to(onnx_namespace()):
        raise MismatchNamespaceError(expect=onnx_namespace(), actual=graph.namespace)
    graph_def = onnx.GraphProto()
    if graph.name is not None:
        graph_def.name = graph.name

    def add_input(graph_def, name, type):
        input_def = graph_def.input.add()
        input_def.name = name
        input_def.type.CopyFrom(deserialize_type(type))
        doc_key = f"input_port/{name}/doc_string"
        if doc_key in graph.attrs:
            input_def.doc_string = graph.attrs[doc_key]

    for op in graph.sorted_ops:
        if op.type == "Initializer":
            graph_def.initializer.append(op.attrs["value"])
            add_input(graph_def, op.name, op.output_port(0).type)
        elif op.type == "SparseInitializer":
            graph_def.sparse_initializer.append(op.attrs["value"])
            add_input(graph_def, op.name, op.output_port(0).type)
        else:
            attrs = without_internal_attrs(op.attrs, ["doc_string"])
            node = graph_def.node.add()
            if "/" in op.type:
                domain, op_type = op.type.split("/")
            else:
                op_type = op.type
                domain = ""
            node.name = op.name
            node.op_type = op_type
            if domain != "":
                node.domain = domain
            if "doc_string" in op.attrs:
                node.doc_string = op.attrs["doc_string"]
            for input_port in op.input_ports:
                in_edges = input_port.in_edges
                if len(in_edges) == 0:
                    break
                src_port = in_edges[0].src
                src_op = src_port.op
                if src_op == graph:
                    node.input.append(src_port.name)
                else:
                    value_name_key = f"output_port/{src_port.name}/value_name"
                    if value_name_key in src_op.attrs:
                        node.input.append(src_op.attrs[value_name_key])
                    else:
                        node.input.append(src_op.name)
            for output_port in op.output_ports:
                if len(output_port.out_edges) == 0:
                    break
                value_name_key = f"output_port/{output_port.name}/value_name"
                if value_name_key in op.attrs:
                    node.output.append(op.attrs[value_name_key])
                else:
                    node.output.append(op.name)
            for name, value in attrs.items():
                if not name.endswith("/doc_string"):
                    attr_value = node.attribute.add()
                    to_attr_proto(value, attr_value)
                    attr_value.name = name
                    if f"{name}/doc_string" in op.attrs:
                        attr_value.doc_string = op.attrs[f"{name}/doc_string"]
    if "doc_string" in graph.attrs:
        graph_def.doc_string = graph.attrs["doc_string"]
    for key in [
        "value_info",
        "quantization_annotation",
    ]:
        if len(graph.attrs[key]) != 0:
            getattr(graph_def, key).extend(graph.attrs[key])
    for port in graph.input_ports:
        add_input(graph_def, port.name, port.type)
    for port in graph.output_ports:
        output_def = graph_def.output.add()
        output_def.name = port.name
        output_def.type.CopyFrom(deserialize_type(port.type))
        doc_key = f"output_port/{port.name}/doc_string"
        if doc_key in graph.attrs:
            output_def.doc_string = graph.attrs[doc_key]
    if file is not None:
        file = Path(file)
        file.write_bytes(graph_def.SerializeToString())
    return graph_def


def from_attr_proto(attr_value: onnx.AttributeProto) -> Any:
    attr_type = attr_value.type
    if attr_type == onnx.AttributeProto.AttributeType.STRING:
        return attr_value.s
    elif attr_type == onnx.AttributeProto.AttributeType.INT:
        return attr_value.i
    elif attr_type == onnx.AttributeProto.AttributeType.FLOAT:
        return attr_value.f
    elif attr_type == onnx.AttributeProto.AttributeType.TENSOR:
        return attr_value.t
    elif attr_type == onnx.AttributeProto.AttributeType.GRAPH:
        return import_from_graph_def(attr_value.g)
    elif attr_type == onnx.AttributeProto.AttributeType.SPARSE_TENSOR:
        return attr_value.sparse_tensor
    elif attr_type == onnx.AttributeProto.AttributeType.STRINGS:
        return [value for value in attr_value.strings]
    elif attr_type == onnx.AttributeProto.AttributeType.INTS:
        return [value for value in attr_value.ints]
    elif attr_type == onnx.AttributeProto.AttributeType.FLOATS:
        return [value for value in attr_value.floats]
    elif attr_type == onnx.AttributeProto.AttributeType.TENSORS:
        return [value for value in attr_value.tensors]
    elif attr_type == onnx.AttributeProto.AttributeType.GRAPHS:
        return [import_from_graph_def(graph_def) for graph_def in attr_value.graphs]
    elif attr_type == onnx.AttributeProto.AttributeType.SPARSE_TENSORS:
        return [value for value in attr_value.sparse_tensors]
    else:
        raise RuntimeError(f"cannot import from AttributeProto {attr_value}")


def to_attr_proto(value: Any, attr_value: onnx.AttributeProto) -> None:
    if isinstance(value, str):
        attr_value.s = bytes(value, "UTF-8")
        attr_value.type = onnx.AttributeProto.AttributeType.STRING
    elif isinstance(value, int):
        attr_value.i = value
        attr_value.type = onnx.AttributeProto.AttributeType.INT
    elif isinstance(value, float):
        attr_value.f = value
        attr_value.type = onnx.AttributeProto.AttributeType.FLOAT
    elif isinstance(value, onnx.TensorProto):
        attr_value.t.CopyFrom(value)
        attr_value.type = onnx.AttributeProto.AttributeType.TENSOR
    elif isinstance(value, onnx.GraphProto):
        attr_value.g.CopyFrom(export_to_graph_def(value))
        attr_value.type = onnx.AttributeProto.AttributeType.GRAPH
    elif isinstance(value, onnx.SparseTensorProto):
        attr_value.sparse_tensor.CopyFrom(value)
        attr_value.type = onnx.AttributeProto.AttributeType.SPARSE_TENSOR
    elif isinstance(value, list) and isinstance(value[0], str):
        attr_value.strings.extend(value)
        attr_value.type = onnx.AttributeProto.AttributeType.STRINGS
    elif isinstance(value, list) and isinstance(value[0], int):
        attr_value.ints.extend(value)
        attr_value.type = onnx.AttributeProto.AttributeType.INTS
    elif isinstance(value, list) and isinstance(value[0], float):
        attr_value.floats.extend(value)
        attr_value.type = onnx.AttributeProto.AttributeType.FLOATS
    elif isinstance(value, list) and isinstance(value[0], onnx.TensorProto):
        attr_value.tensors.extend(value)
        attr_value.type = onnx.AttributeProto.AttributeType.TENSORS
    elif isinstance(value, list) and isinstance(value[0], onnx.GraphProto):
        attr_value.graphs.extend([export_to_graph_def(graph) for graph in value])
        attr_value.type = onnx.AttributeProto.AttributeType.GRAPHS
    elif isinstance(value, list) and isinstance(value[0], onnx.SparseTensorProto):
        attr_value.sparse_tensors.extend(value)
        attr_value.type = onnx.AttributeProto.AttributeType.SPARSE_TENSORS
    else:
        raise RuntimeError(f"cannot export to AttributeProto from {value}")


import_types = {
    "onnx_model": import_from_model_def,
    "onnx_graph": import_from_graph_def,
}

export_types = {
    "onnx_model": export_to_model_def,
    "onnx_graph": export_to_graph_def,
}
