import os
from pathlib import Path
from typing import Union

import yaml

from amanda.graph import Graph
from amanda.io.graph_pb2 import NodeDef
from amanda.io.proto import from_proto, to_proto
from amanda.io.text import from_text, to_text


def save_to_proto(graph: Graph, path: Union[str, Path]) -> None:
    path = ensure_dir(str(path) + ".amanda.pb")
    with open(path, "wb") as file:
        file.write(to_proto(graph).SerializeToString())


def load_from_proto(path: Union[str, Path]) -> Graph:
    path = str(path) + ".amanda.pb"
    proto = NodeDef()
    with open(path, "rb") as file:
        proto.ParseFromString(file.read())
    return from_proto(proto)


def save_to_yaml(graph: Graph, path: Union[str, Path]) -> None:
    path = ensure_dir(str(path) + ".amanda.yaml")
    with open(path, "w") as file:
        yaml.dump(to_text(graph), file)


def load_from_yaml(path: Union[str, Path]) -> Graph:
    path = str(path) + ".amanda.yaml"
    with open(path, "r") as file:
        text = yaml.load(file)
    return from_text(text)


def ensure_dir(path: str) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except FileExistsError:
            pass
    return path


def root_dir() -> Path:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return Path(current_dir).parents[2]
