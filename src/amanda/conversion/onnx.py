from pathlib import Path
from typing import Any, Dict, List, Set, Union

import onnx

from amanda.conversion.utils import to_proto
from amanda.graph import Graph, Op, Tensor
from amanda.namespace import (
    Namespace,
    default_namespace,
    get_global_registry,
    is_qualified,
)
from amanda.rule import OpMapping, Rule, RuleMapper

_namespace = default_namespace() / Namespace("onnx")
_internal_namespace = _namespace / Namespace("internal")


def onnx_namespace() -> Namespace:
    return _namespace


def onnx_internal_namespace() -> Namespace:
    return _internal_namespace


class ToDefaultRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        for key in ["name", "type"]:
            if onnx_namespace().qualified(key) in op.attrs:
                op.attrs[key] = op.attrs[onnx_namespace().qualified(key)]
        return OpMapping(source_ops=[op], target_ops=[op])


class ToONNXRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        for key in ["name", "type"]:
            if default_namespace().qualified(key) in op.attrs:
                op.attrs[key] = op.attrs[default_namespace().qualified(key)]
        return OpMapping(source_ops=[op], target_ops=[op])


_onnx_to_default_mapper = RuleMapper(rules=[ToDefaultRule()])
_default_to_onnx_mapper = RuleMapper(rules=[ToONNXRule()])


def onnx_to_default_mapper() -> RuleMapper:
    return _onnx_to_default_mapper


def default_to_onnx_mapper() -> RuleMapper:
    return _default_to_onnx_mapper


get_global_registry().add_mapper(
    onnx_namespace(), default_namespace(), onnx_to_default_mapper()
)
get_global_registry().add_mapper(
    default_namespace(), onnx_namespace(), default_to_onnx_mapper()
)


def import_from_model_def(model_def: Union[onnx.ModelProto, str, bytes, Path]) -> Graph:
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
        model_key = f"model:{key}"
        if model_def.HasField(key):
            graph.attrs[model_key] = getattr(model_def, key)
    for key in ["opset_import", "metadata_props"]:
        model_key = f"model:{key}"
        graph.attrs[model_key] = list(getattr(model_def, key))
    return graph


def import_from_graph_def(graph_def: Union[onnx.GraphProto, str, bytes, Path]) -> Graph:
    graph_def = to_proto(graph_def, onnx.GraphProto)
    graph = Graph()
    name_to_node = {node.name: node for node in graph_def.node}
    name_to_tensor: Dict[str, Tensor] = {}
    graph_inputs: Set[str] = set()

    def add_op(node: onnx.NodeProto):
        if graph.get_op_by_name(node.name) is not None:
            return
        for input_tensor in node.input:
            if input_tensor in graph_inputs:
                continue
            add_op(name_to_node[name_to_tensor[input_tensor].op.name])

        attrs = {}
        for attr in node.attribute:
            attr_name = attr.name
            attrs[attr_name] = from_attr_proto(attr)
            attrs[onnx_internal_namespace().qualified(f"{attr_name}/type")] = attr.type
            if attr.HasField("doc_string"):
                attrs[
                    onnx_internal_namespace().qualified(f"{attr_name}/doc_string")
                ] = attr.doc_string

        op = Op(
            attrs=attrs,
            input_tensors=[
                graph.get_op_by_name(input_tensor).output_tensor(0)
                for input_tensor in node.input
            ],
            control_dependencies=[],
            output_num=len(node.output),
        )
        for index, output_tensor_name in enumerate(node.output):
            output_tensor = op.output_tensor(index)
            output_tensor.attrs["name"] = output_tensor_name
            name_to_tensor[output_tensor_name] = output_tensor
        op.namespace = onnx_namespace()
        op.type = node.op_type
        for key in ["name", "domain", "doc_string"]:
            if node.HasField(key):
                op.attrs[key] = getattr(node, key)
        graph.add_op(op)

    graph.namespace = onnx_namespace()
    for key in ["name", "doc_string"]:
        if graph_def.HasField(key):
            graph.attrs[key] = getattr(graph_def, key)
    for key in [
        "initializer",
        "sparse_initializer",
        "input",
        "output",
        "value_info",
        "quantization_annotation",
    ]:
        graph.attrs[key] = list(getattr(graph_def, key))
        if key == "input":
            for input_def in graph_def.input:
                add_value_info_as_input(graph, input_def)
                graph_inputs.add(input_def.name)
    for node in graph_def.node:
        add_op(node)
    return graph


def add_value_info_as_input(graph: Graph, value_info: onnx.ValueInfoProto):
    op = Op()
    op.namespace = onnx_namespace()
    op.name = value_info.name
    op.type = onnx_namespace().qualified("input")
    op.attrs[onnx_internal_namespace().qualified("value_info")] = value_info
    graph.add_op(op)


def export_to_model_def(graph: Graph, file: Union[str, Path] = None) -> onnx.ModelProto:
    if graph.namespace != onnx_namespace():
        graph = graph.to_default_namespace().to_namespace(onnx_namespace())
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
        model_key = f"model:{key}"
        if model_key in graph.attrs:
            setattr(model_def, key, graph.attrs[model_key])
    for key in ["opset_import", "metadata_props"]:
        model_key = f"model:{key}"
        getattr(model_def, key).extend(graph.attrs[model_key])
    if file is not None:
        onnx.save(model_def, str(file))
    return model_def


def export_to_graph_def(graph: Graph, file: Union[str, Path] = None) -> onnx.GraphProto:
    if graph.namespace != onnx_namespace():
        graph = graph.to_default_namespace().to_namespace(onnx_namespace())
    graph_def = onnx.GraphProto()
    graph_inputs = {}
    for op in graph.sorted_ops:
        if op.type == onnx_namespace().qualified("input"):
            graph_inputs[op.name] = op.attrs[
                onnx_internal_namespace().qualified("value_info")
            ]
            continue
        attrs = without_internal_attrs(op.attrs)
        node = graph_def.node.add()
        node.op_type = op.type
        for key in ["name", "domain", "doc_string"]:
            if key in op.attrs:
                setattr(node, key, op.attrs[key])
        node.output.extend(
            [output_tensor.attrs["name"] for output_tensor in op.output_tensors]
        )
        node.input.extend(
            [
                input_tensor.op.name
                if input_tensor.op.name in graph_inputs
                else input_tensor.attrs["name"]
                for input_tensor in op.input_tensors
            ]
        )
        for name, value in attrs.items():
            attr_value = node.attribute.add()
            attr_value.type = op.attrs[
                onnx_internal_namespace().qualified(f"{name}/type")
            ]
            to_attr_proto(value, attr_value)
            attr_value.name = name
            doc_string_key = onnx_internal_namespace().qualified(f"{name}/doc_string")
            if doc_string_key in op.attrs:
                attr_value.doc_string = op.attrs[doc_string_key]
    for key in ["name", "doc_string"]:
        if key in graph.attrs:
            setattr(graph_def, key, graph.attrs[key])
    for key in [
        "initializer",
        "sparse_initializer",
        "input",
        "output",
        "value_info",
        "quantization_annotation",
    ]:
        if key == "input":
            input_defs: List[onnx.ValueInfoProto] = graph.attrs[key]
            name_to_input_def = {input_def.name: input_def for input_def in input_defs}
            for input_name in set(
                list(graph_inputs.keys()) + list(name_to_input_def.keys())
            ):
                if input_name in name_to_input_def and input_name not in graph_inputs:
                    input_defs.remove(name_to_input_def[input_name])
                elif input_name not in name_to_input_def and input_name in graph_inputs:
                    input_defs.append(graph_inputs[input_name])
        getattr(graph_def, key).extend(graph.attrs[key])
    if file is not None:
        file = Path(file)
        file.write_bytes(graph_def.SerializeToString())
    return graph_def


def without_internal_attrs(attrs):
    return {
        name: value
        for name, value in attrs.items()
        if not (is_qualified(name) or name in ["name", "type", "domain", "doc_string"])
    }


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
    attr_type = attr_value.type
    if attr_type == onnx.AttributeProto.AttributeType.STRING:
        attr_value.s = value
    elif attr_type == onnx.AttributeProto.AttributeType.INT:
        attr_value.i = value
    elif attr_type == onnx.AttributeProto.AttributeType.FLOAT:
        attr_value.f = value
    elif attr_type == onnx.AttributeProto.AttributeType.TENSOR:
        attr_value.t.CopyFrom(value)
    elif attr_type == onnx.AttributeProto.AttributeType.GRAPH:
        attr_value.g.CopyFrom(export_to_graph_def(value))
    elif attr_type == onnx.AttributeProto.AttributeType.SPARSE_TENSOR:
        attr_value.sparse_tensor.CopyFrom(value)
    elif attr_type == onnx.AttributeProto.AttributeType.STRINGS:
        attr_value.strings.extend(value)
    elif attr_type == onnx.AttributeProto.AttributeType.INTS:
        attr_value.ints.extend(value)
    elif attr_type == onnx.AttributeProto.AttributeType.FLOATS:
        attr_value.floats.extend(value)
    elif attr_type == onnx.AttributeProto.AttributeType.TENSORS:
        attr_value.tensors.extend(value)
    elif attr_type == onnx.AttributeProto.AttributeType.GRAPHS:
        attr_value.graphs.extend([export_to_graph_def(graph) for graph in value])
    elif attr_type == onnx.AttributeProto.AttributeType.SPARSE_TENSORS:
        attr_value.sparse_tensors.extend(value)
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
