from collections import OrderedDict
from typing import Any, Dict, List, Union, cast

from amanda.attributes import Attributes
from amanda.graph import (
    Graph,
    InputPort,
    Op,
    OutputPort,
    SubGraph,
    create_graph,
    create_op,
    create_subgraph,
)
from amanda.io.graph_pb2 import NodeDef
from amanda.io.serde import SerdeContext, deserialize, serialize, serialize_type
from amanda.io.value_pb2 import (
    DataTypeValue,
    ListValue,
    NullValue,
    SerializedValue,
    Struct,
    Value,
)
from amanda.namespace import Namespace
from amanda.type import DataType


def to_proto(graph: Graph, proto: NodeDef = None) -> NodeDef:
    return from_graph(graph, proto)


def from_proto(proto: NodeDef) -> Graph:
    return to_graph(proto)


def from_list(values: List[Any], proto: ListValue = None) -> ListValue:
    proto = proto or ListValue()
    for value in values:
        value_def = proto.values.add()
        from_value(value, value_def)
    return proto


def from_dict(struct: Dict[str, Any], proto: Struct = None) -> Struct:
    proto = proto or Struct()
    for key, value in struct.items():
        from_value(value, proto.fields[key])
    return proto


def from_raw_value(value: Any, proto: SerializedValue = None) -> SerializedValue:
    proto = proto or SerializedValue()
    from_dtype(serialize_type(type(value)), proto.type)
    from_value(serialize(value), proto.value)
    return proto


def from_value(value: Any, proto: Value = None) -> Value:
    proto = proto or Value()
    if value is None:
        proto.null_value = NullValue.NULL_VALUE
    elif isinstance(value, bool):
        proto.bool_value = value
    # bool is a subclass of int, so we should check bool before checking int
    elif isinstance(value, int):
        proto.int_value = value
    elif isinstance(value, float):
        proto.double_value = value
    elif isinstance(value, str):
        proto.string_value = value
    elif isinstance(value, bytes):
        proto.bytes_value = value
    elif isinstance(value, DataType):
        proto.type_value.SetInParent()
        from_dtype(value, proto.type_value)
    elif isinstance(value, list):
        proto.list_value.SetInParent()
        from_list(value, proto.list_value)
    elif isinstance(value, dict):
        proto.struct_value.SetInParent()
        from_dict(value, proto.struct_value)
    else:
        proto.serialized_value.SetInParent()
        from_raw_value(value, proto.serialized_value)
    assert proto.WhichOneof("kind") is not None
    return proto


def from_attrs(attrs: Attributes, proto: Struct = None) -> Struct:
    return from_dict(attrs, proto)


def from_dtype(dtype: DataType, proto: DataTypeValue = None) -> DataTypeValue:
    proto = proto or DataTypeValue()
    proto.namespace = dtype.namespace.full_name
    proto.name = dtype.name
    from_attrs(dtype.attrs, proto.attrs)
    return proto


def from_op(op: Op, proto: NodeDef = None) -> NodeDef:
    proto = proto or NodeDef()
    if op.name is not None:
        proto.name = op.name
    if op.type is not None:
        proto.type = op.type
    if op.namespace is not None:
        proto.namespace = op.namespace.full_name
    for input_port in op.input_ports:
        port_def = proto.input_ports.add()
        port_def.name = input_port.name
        from_dtype(input_port.type, port_def.type)
    for output_port in op.output_ports:
        port_def = proto.output_ports.add()
        port_def.name = output_port.name
        from_dtype(output_port.type, port_def.type)
    from_attrs(op.attrs, proto.attrs)
    return proto


def from_graph(graph: Graph, proto: NodeDef = None) -> NodeDef:
    proto = proto or NodeDef()
    if isinstance(graph, SubGraph):
        proto.node_kind = NodeDef.SUBGRAPH
        from_op(graph, proto)
    else:
        proto.node_kind = NodeDef.GRAPH
        proto.namespace = graph.namespace.full_name
        from_attrs(graph.attrs, proto.attrs)
    for op in graph.sorted_ops:
        op_def = proto.ops.add()
        if isinstance(op, SubGraph):
            from_graph(op, op_def)
        else:
            op_def.node_kind = NodeDef.OP
            from_op(op, op_def)
    for edge in graph.edges:
        edge_def = proto.edges.add()
        from_raw_value(edge.src, edge_def.src)
        from_raw_value(edge.dst, edge_def.dst)
        from_attrs(edge.attrs, edge_def.attrs)
    return proto


def to_list(proto: ListValue, context: SerdeContext) -> List[Any]:
    return [to_value(value, context) for value in proto.values]


def to_dict(proto: Struct, context: SerdeContext) -> Dict[str, Any]:
    return {name: to_value(proto.fields[name], context) for name in proto.fields}


def to_raw_value(proto: SerializedValue, context: SerdeContext) -> Any:
    dtype = to_dtype(proto.type, context)
    value = to_value(proto.value, context)
    context.dtype = dtype
    raw_value = deserialize(value, context)
    context.dtype = None
    return raw_value


def to_value(proto: Value, context: SerdeContext) -> Any:
    kind = proto.WhichOneof("kind")
    assert kind is not None
    if kind == "null_value":
        return None
    elif kind == "bool_value":
        return proto.bool_value
    elif kind == "int_value":
        return proto.int_value
    elif kind == "double_value":
        return proto.double_value
    elif kind == "string_value":
        return proto.string_value
    elif kind == "bytes_value":
        return proto.bytes_value
    elif kind == "type_value":
        return to_dtype(proto.type_value, context)
    elif kind == "struct_value":
        return to_dict(proto.struct_value, context)
    elif kind == "list_value":
        return to_list(proto.list_value, context)
    else:
        assert kind == "serialized_value"
        return to_raw_value(proto.serialized_value, context)


def to_attrs(proto: Struct, context: SerdeContext) -> Attributes:
    return Attributes(to_dict(proto, context))


def to_dtype(proto: DataTypeValue, context: SerdeContext) -> DataType:
    return DataType(
        namespace=Namespace(proto.namespace),
        name=proto.name,
        attrs=to_attrs(proto.attrs, context),
    )


def to_node(proto: NodeDef, context: SerdeContext) -> Union[Graph, Op]:
    namespace = Namespace(proto.namespace) if len(proto.namespace) != 0 else None
    type = proto.type if len(proto.type) != 0 else None
    name = proto.name if len(proto.name) != 0 else None
    inputs = OrderedDict(
        [
            (port_def.name, to_dtype(port_def.type, context))
            for port_def in proto.input_ports
        ]
    )
    outputs = OrderedDict(
        [
            (port_def.name, to_dtype(port_def.type, context))
            for port_def in proto.output_ports
        ]
    )
    node: Union[Op, Graph]
    if proto.node_kind == NodeDef.GRAPH:
        node = create_graph(name=name, namespace=namespace)
    elif proto.node_kind == NodeDef.SUBGRAPH:
        node = create_subgraph(
            type=type,
            name=name,
            namespace=namespace,
            inputs=inputs,
            outputs=outputs,
        )
    else:
        assert proto.node_kind == NodeDef.OP
        node = create_op(
            type=type,
            name=name,
            namespace=namespace,
            inputs=inputs,
            outputs=outputs,
        )
    if isinstance(node, Graph):
        graph = cast(Graph, node)
        for op_def in proto.ops:
            op = to_node(op_def, context)
            graph.add_op(cast(Op, op))
    return node


def assign_edges(node: Union[Graph, Op], proto: NodeDef, context: SerdeContext):
    for op_def in proto.ops:
        op = cast(Graph, node).get_op(op_def.name)
        assign_edges(op, op_def, context)
    for edge_def in proto.edges:
        src = to_raw_value(edge_def.src, context)
        assert isinstance(src, OutputPort)
        dst = to_raw_value(edge_def.dst, context)
        assert isinstance(dst, InputPort)
        cast(Graph, node).create_edge(src, dst)


def assign_attrs(node: Union[Graph, Op], proto: NodeDef, context: SerdeContext):
    node.attrs = to_attrs(proto.attrs, context)
    for op_def in proto.ops:
        if len(op_def.attrs.fields) != 0:
            op = cast(Graph, node).get_op(op_def.name)
            assign_attrs(op, op_def, context)
    for edge_def in proto.edges:
        if len(edge_def.attrs.fields) != 0:
            src = to_raw_value(edge_def.src, context)
            dst = to_raw_value(edge_def.dst, context)
            edge = cast(Graph, node).get_edge(src, dst)
            edge.attrs = to_attrs(edge_def.attrs, context)


def to_graph(proto: NodeDef) -> Graph:
    context = SerdeContext()
    graph = cast(Graph, to_node(proto, context))
    context.root = graph
    assign_edges(graph, proto, context)
    assign_attrs(graph, proto, context)
    return graph
