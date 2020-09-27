from typing import Any, Dict

from google.protobuf import json_format

from amanda.graph import Graph
from amanda.io.graph_pb2 import NodeDef
from amanda.io.proto import from_proto, to_proto


def to_text(graph: Graph) -> Dict[str, Any]:
    return from_graph(graph)


def from_text(text: Dict[str, Any]) -> Graph:
    return to_graph(text)


def from_graph(graph: Graph) -> Dict[str, Any]:
    proto = to_proto(graph)
    return json_format.MessageToDict(proto, preserving_proto_field_name=True)


def to_graph(text: Dict[str, Any]) -> Graph:
    proto = NodeDef()
    json_format.ParseDict(text, proto)
    return from_proto(proto)
