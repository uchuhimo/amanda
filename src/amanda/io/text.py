from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

from amanda.graph import Graph
from amanda.io.graph_pb2 import EdgeDef, NodeDef, PortDef
from amanda.io.proto import from_proto, to_proto
from amanda.io.serde import graph_namespace
from amanda.io.value_pb2 import (
    DataTypeValue,
    ListValue,
    NullValue,
    RefValue,
    SerializedValue,
    Struct,
    Value,
)
from amanda.type import unknown_type


def to_text(graph: Graph) -> Dict[str, Any]:
    return from_graph(graph)


def from_text(text: Dict[str, Any]) -> Graph:
    return to_graph(text)


@dataclass
class Context:
    path: List[str] = field(default_factory=list)


def from_dict(proto: Struct, context: Context) -> Dict[str, Any]:
    return {key: from_value(proto.fields[key], context) for key in proto.fields}


def from_list(proto: ListValue, context: Context) -> List[Any]:
    return [from_value(value, context) for value in proto.values]


@dataclass
class DTypeWrapper:
    dtype: Union[str, Dict[str, Any]]


@dataclass
class RefWrapper:
    ref: str
    tag: str


def from_serialized_value(
    proto: SerializedValue, context: Context
) -> Union[RefWrapper, str, Dict[str, Any]]:
    if proto.type.namespace == graph_namespace().full_name:
        if proto.type.name == "Node":
            return RefWrapper(from_serialized_op(proto.value, context), tag="node")
        elif proto.type.name == "Port":
            ref, tag = from_serialized_port(proto.value, context)
            return RefWrapper(ref, tag)
        else:
            assert proto.type.name == "Edge"
            return RefWrapper(from_serialized_edge(proto.value, context), tag="edge")
    value = from_value(proto.value, context)
    if isinstance(value, dict):
        value["@type"] = from_dtype(proto.type, context)
        return value
    else:
        return {
            "@type": from_dtype(proto.type, context),
            "@value": value,
        }


def from_dtype(proto: DataTypeValue, context: Context) -> Union[str, Dict[str, Any]]:
    full_name = f"{proto.namespace}.{proto.name}"
    if len(proto.attrs.fields) == 0:
        return full_name
    else:
        return {full_name: from_dict(proto.attrs, context)}


def from_ref(proto: RefValue) -> str:
    return f"{proto.schema}:{proto.value}"


def from_value(proto: Value, context: Context) -> Any:
    kind = proto.WhichOneof("kind")
    if kind in [
        "bool_value",
        "int_value",
        "double_value",
        "string_value",
        "bytes_value",
    ]:
        return getattr(proto, kind)
    elif kind == "null_value":
        return None
    elif kind == "struct_value":
        return from_dict(proto.struct_value, context)
    elif kind == "list_value":
        return from_list(proto.list_value, context)
    elif kind == "serialized_value":
        return from_serialized_value(proto.serialized_value, context)
    elif kind == "type_value":
        return DTypeWrapper(from_dtype(proto.type_value, context))
    else:
        assert kind == "ref_value"
        return RefWrapper(from_ref(proto.ref_value), tag="ref")


def from_port(proto: PortDef, context: Context) -> Union[str, Dict[str, Any]]:
    if (
        proto.type.namespace == unknown_type.namespace.full_name
        and proto.type.name == unknown_type.name
    ):
        return proto.name
    else:
        return {proto.name: from_dtype(proto.type, context)}


def path_is_in(path1: List[str], path2: List[str]) -> bool:
    if len(path1) >= len(path2):
        for name1, name2 in zip(path1, path2):
            if name1 != name2:
                return False
        return True
    else:
        return False


def from_serialized_op(proto: Value, context: Context) -> str:
    path = from_list(proto.list_value, context)
    if path_is_in(path, context.path):
        if len(path) == len(context.path):
            return "."
        else:
            return ".".join(path[len(context.path) :])
    else:
        if len(path) == 0:
            return ":."
        else:
            return ":" + ".".join(path)


def from_serialized_port(proto: Value, context: Context) -> Tuple[str, str]:
    fields = proto.struct_value.fields
    if "input_port" in fields:
        port_field = "input_port"
        tag = "in"
    else:
        port_field = "output_port"
        tag = "out"
    op_text = from_serialized_op(fields["op"], context)
    return f"{op_text}:{fields[port_field].string_value}", tag


def from_serialized_edge(proto: Value, context: Context) -> str:
    fields = proto.struct_value.fields
    op_text = from_serialized_op(fields["graph"], context)
    src_text = from_serialized_port(fields["src"], context)[0]
    dst_text = from_serialized_port(fields["dst"], context)[0]
    if op_text != "":
        return f"{op_text}[{src_text} -> {dst_text}]"
    else:
        return f"{src_text} -> {dst_text}"


def from_edge(proto: EdgeDef, context: Context) -> Union[str, Dict[str, Any]]:
    edge_text = (
        f"{from_serialized_port(proto.src.value, context)[0]} -> "
        f"{from_serialized_port(proto.dst.value, context)[0]}"
    )
    if len(proto.attrs.fields) == 0:
        return edge_text
    else:
        return {edge_text: from_dict(proto.attrs, context)}


def from_node(proto: NodeDef, context: Context = None) -> Dict[str, Any]:
    if context is None:
        context = Context()
    else:
        context.path.append(proto.name)
    text: Dict[str, Any] = {}
    if proto.type != "":
        text["type"] = proto.type
    if proto.namespace != "":
        text["namespace"] = proto.namespace
    if len(proto.input_ports) != 0:
        text["input_ports"] = [from_port(port, context) for port in proto.input_ports]
    if len(proto.output_ports) != 0:
        text["output_ports"] = [from_port(port, context) for port in proto.output_ports]
    if len(proto.attrs.fields) != 0:
        text["attrs"] = from_dict(proto.attrs, context)
    if len(proto.ops) != 0:
        text["ops"] = {op.name: from_node(op, context) for op in proto.ops}
    if len(proto.edges) != 0:
        text["edges"] = [from_edge(edge, context) for edge in proto.edges]
    if len(context.path) != 0:
        context.path.pop()
    return text


def from_graph(graph: Graph) -> Dict[str, Any]:
    proto = to_proto(graph)
    return from_proto_to_text(proto)


def from_proto_to_text(proto: NodeDef) -> Dict[str, Any]:
    return {proto.name: from_node(proto)}


def to_graph(text: Dict[str, Any]) -> Graph:
    return from_proto(from_text_to_proto(text))


def to_list(text: List[Any], context: Context, proto: ListValue = None) -> ListValue:
    proto = proto or ListValue()
    for value in text:
        value_def = proto.values.add()
        to_value(value, context, value_def)
    return proto


def to_dict(text: Dict[str, Any], context: Context, proto: Struct = None) -> Struct:
    proto = proto or Struct()
    for key, value in text.items():
        to_value(value, context, proto.fields[key])
    return proto


def to_serialized_value(
    text: Dict[str, Any], context: Context, proto: SerializedValue = None
) -> SerializedValue:
    proto = proto or SerializedValue()
    to_dtype(text["@type"], context, proto.type)
    if "@value" in text:
        to_value(text["@value"], context, proto.value)
    else:
        del text["@type"]
        to_value(text, context, proto.value)
    return proto


def to_ref(text: str, proto: RefValue = None) -> RefValue:
    proto = proto or RefValue()
    schema, _, value = text.partition(":")
    proto.schema = schema
    proto.value = value
    return proto


def to_src_and_dst(text: str, src_proto: Value, dst_proto: Value, context: Context):
    src_and_dst = text.split("->")
    assert len(src_and_dst) == 2
    src, dst = src_and_dst
    src = src.strip()
    dst = dst.strip()
    op_text, _, port_text = src.rpartition(":")
    to_serialized_port(op_text, "out", port_text, context, src_proto)
    op_text, _, port_text = dst.rpartition(":")
    to_serialized_port(op_text, "in", port_text, context, dst_proto)
    src_fields = src_proto.struct_value.fields
    dst_fields = dst_proto.struct_value.fields
    src_path = [value.string_value for value in src_fields["op"].list_value.values]
    dst_path = [value.string_value for value in dst_fields["op"].list_value.values]
    if path_is_in(src_path, dst_path) and len(src_path) != len(dst_path):
        dst_fields["output_port"].CopyFrom(dst_fields["input_port"])
        del dst_fields["input_port"]
    if path_is_in(dst_path, src_path) and len(src_path) != len(dst_path):
        src_fields["input_port"].CopyFrom(src_fields["output_port"])
        del src_fields["output_port"]


def to_graph_ref(
    text: str, tag: str, context: Context, proto: SerializedValue
) -> SerializedValue:
    proto = proto or SerializedValue()
    proto.type.namespace = graph_namespace().full_name
    fields = proto.value.struct_value.fields
    if tag == "edge":
        proto.type.name = "Edge"
        if text.endswith("]"):
            graph_text, _, edge_text = text[:-1].partition("[")
        else:
            graph_text = ""
            edge_text = text
        to_serialized_op(graph_text, context, fields["graph"])
        to_src_and_dst(edge_text, fields["src"], fields["dst"], context)
    elif tag == "node":
        proto.type.name = "Node"
        to_serialized_op(text, context, proto.value)
    else:
        assert tag in ["in", "out"]
        proto.type.name = "Port"
        op_text, _, port_text = text.rpartition(":")
        to_serialized_port(op_text, tag, port_text, context, proto.value)
    return proto


def to_value(text: Any, context: Context, proto: Value = None) -> Value:
    proto = proto or Value()
    if text is None:
        proto.null_value = NullValue.NULL_VALUE
    elif isinstance(text, bool):
        proto.bool_value = text
    # bool is a subclass of int, so we should check bool before checking int
    elif isinstance(text, int):
        proto.int_value = text
    elif isinstance(text, float):
        proto.double_value = text
    elif isinstance(text, str):
        proto.string_value = text
    elif isinstance(text, bytes):
        proto.bytes_value = text
    elif isinstance(text, DTypeWrapper):
        proto.type_value.SetInParent()
        to_dtype(text.dtype, context, proto.type_value)
    elif isinstance(text, RefWrapper):
        tag = text.tag[1:]
        if tag == "ref":
            to_ref(text.ref, proto.ref_value)
        else:
            to_graph_ref(text.ref, tag, context, proto.serialized_value)
    elif isinstance(text, list):
        proto.list_value.SetInParent()
        to_list(text, context, proto.list_value)
    else:
        assert isinstance(text, dict)
        if "@type" in text:
            proto.serialized_value.SetInParent()
            to_serialized_value(text, context, proto.serialized_value)
        else:
            proto.struct_value.SetInParent()
            to_dict(text, context, proto.struct_value)
    assert proto.WhichOneof("kind") is not None
    return proto


def to_dtype(
    text: Union[str, Dict[str, Any]], context: Context, proto: DataTypeValue = None
) -> DataTypeValue:
    proto = proto or DataTypeValue()
    if isinstance(text, str):
        full_name = text
    else:
        full_name = list(text.keys())[0]
        to_dict(list(text.values())[0], context, proto.attrs)
    names = full_name.split(".")
    assert len(names) == 2
    namespace, name = names
    proto.namespace = namespace
    proto.name = name
    return proto


def to_serialized_op(text: str, context: Context, proto: Value = None) -> Value:
    proto = proto or Value()
    proto.list_value.SetInParent()
    if text == ".":
        to_list(context.path, context, proto.list_value)
    elif text == ":.":
        to_list([], context, proto.list_value)
    elif text.startswith(":"):
        to_list(text[1:].split("."), context, proto.list_value)
    else:
        to_list(context.path + text.split("."), context, proto.list_value)
    return proto


def to_serialized_port(
    op_text: str, tag: str, port_text: str, context: Context, proto: Value = None
) -> Value:
    proto = proto or Value()
    fields = proto.struct_value.fields
    if tag == "in":
        field_name = "input_port"
    else:
        assert tag == "out"
        field_name = "output_port"
    to_serialized_op(op_text, context, fields["op"])
    fields[field_name].string_value = port_text
    return proto


def to_edge(text, context: Context, proto: EdgeDef = None) -> EdgeDef:
    proto = proto or EdgeDef()
    if isinstance(text, str):
        edge_text = text
    else:
        edge_text = list(text.keys())[0]
        to_dict(list(text.values())[0], context, proto.attrs)
    proto.src.type.namespace = graph_namespace().full_name
    proto.src.type.name = "Port"
    proto.dst.type.namespace = graph_namespace().full_name
    proto.dst.type.name = "Port"
    to_src_and_dst(edge_text, proto.src.value, proto.dst.value, context)
    return proto


def to_port(
    text: Union[str, Dict[str, Any]], context: Context, proto: PortDef = None
) -> PortDef:
    proto = proto or PortDef()
    if isinstance(text, str):
        proto.name = text
        proto.type.namespace = unknown_type.namespace.full_name
        proto.type.name = unknown_type.name
    else:
        proto.name = list(text.keys())[0]
        to_dtype(list(text.values())[0], context, proto.type)
    return proto


def to_op(
    name: str, text: Dict[str, Any], context: Context = None, proto: NodeDef = None
) -> NodeDef:
    proto = proto or NodeDef()
    if context is None:
        context = Context()
    else:
        context.path.append(name)
    proto.name = name
    if "type" in text:
        proto.type = text["type"]
    if "namespace" in text:
        proto.namespace = text["namespace"]
    if "input_ports" in text:
        for port in text["input_ports"]:
            to_port(port, context, proto.input_ports.add())
    if "output_ports" in text:
        for port in text["output_ports"]:
            to_port(port, context, proto.output_ports.add())
    if "attrs" in text:
        to_dict(text["attrs"], context, proto.attrs)
    if "ops" in text:
        proto.node_kind = NodeDef.SUBGRAPH
        for name, op in text["ops"].items():
            to_op(name, op, context, proto.ops.add())
    else:
        proto.node_kind = NodeDef.OP
    if "edges" in text:
        for edge in text["edges"]:
            to_edge(edge, context, proto.edges.add())
    if len(context.path) != 0:
        context.path.pop()
    return proto


def from_text_to_proto(text: Dict[str, Any]) -> NodeDef:
    name = list(text.keys())[0]
    proto = to_op(name, text[name])
    if "input_ports" not in text[name] and "output_ports" not in text[name]:
        proto.node_kind = NodeDef.GRAPH
    return proto
