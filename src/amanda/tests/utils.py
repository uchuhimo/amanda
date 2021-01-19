from typing import Any, Dict

import jsondiff
import onnx
import tensorflow as tf
from google.protobuf import json_format

from amanda.conversion.tensorflow import export_to_graph_def, import_from_graph_def
from amanda.conversion.utils import recursionlimit, repeated_fields_to_dict
from amanda.namespace import Namespace, default_namespace

_test_namespace = default_namespace() / Namespace("test")


def test_namespace() -> Namespace:
    return _test_namespace


def get_diff_after_conversion(graph_def: tf.GraphDef) -> Dict[str, Any]:
    graph = import_from_graph_def(graph_def)
    new_graph_def = export_to_graph_def(graph)
    return diff_graph_def(graph_def, new_graph_def)


def diff_graph_def(graph_def1, graph_def2) -> Dict[str, Any]:
    with recursionlimit(10000):
        return jsondiff.diff(
            graph_def_to_dict(graph_def1),
            graph_def_to_dict(graph_def2),
            syntax="explicit",
        )


def diff_proto(proto1, proto2) -> Dict[str, Any]:
    return jsondiff.diff(
        json_format.MessageToDict(proto1, preserving_proto_field_name=True),
        json_format.MessageToDict(proto2, preserving_proto_field_name=True),
        syntax="explicit",
    )


def node_def_to_dict(node_def) -> Dict[str, Any]:
    if isinstance(node_def, onnx.NodeProto):
        attrs = repeated_fields_to_dict(node_def.attribute)
        node_def.ClearField("attribute")
        node_dict = json_format.MessageToDict(
            node_def, preserving_proto_field_name=True
        )
        node_dict["attribute"] = attrs
        return node_dict
    else:
        return json_format.MessageToDict(node_def, preserving_proto_field_name=True)


def graph_def_to_dict(graph_def) -> Dict[str, Any]:
    nodes = {node.name: node_def_to_dict(node) for node in graph_def.node}
    graph_def.ClearField("node")
    graph_dict = json_format.MessageToDict(graph_def, preserving_proto_field_name=True)
    graph_dict["node"] = nodes
    return graph_dict
