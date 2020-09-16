import sys
import typing
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Union

import jsondiff
import onnx
from google.protobuf import json_format
from google.protobuf.message import Message

T = TypeVar("T", bound=Message)


def to_proto(proto: Union[T, str, bytes, Path], message_type: Type[T]) -> T:
    if not isinstance(proto, message_type):
        if isinstance(proto, bytes):
            proto_bytes = proto
        else:
            if isinstance(proto, str):
                proto = Path(proto)
            proto_bytes = typing.cast(Path, proto).read_bytes()
        proto = message_type()
        proto.ParseFromString(proto_bytes)
    return proto


def repeated_fields_to_dict(repeated_fields) -> Dict[str, Any]:
    return {
        proto_def.name: json_format.MessageToDict(
            proto_def, preserving_proto_field_name=True
        )
        for proto_def in repeated_fields
    }


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


@contextmanager
def recursionlimit(limit: int):
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(limit)
        yield None
    finally:
        sys.setrecursionlimit(old_limit)


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
