# type: ignore
from typing import Any, Dict, List


class Op:
    in_edges: List["Edge"]
    out_edges: List["Edge"]
    attrs: Dict[str, Any]

    # builtin attributes
    name: str  # is attrs["name"]
    type: str  # is attrs["type"]


class Edge:
    src: Op
    dst: Op
    attrs: Dict[str, Any]

    # builtin attributes
    tensor: Any  # is attrs["tensor"]

    def replace_src(self, src: Op):
        self.src.out_edges.remove(self)
        src.out_edges.append(self)
        self.src = src


class Graph:
    ops: List[Op]
    attrs: Dict[str, Any]

    # builtin attributes
    namespace: str  # is attrs["namespace"]

    def add_op(self, op: Op) -> None:
        self.ops.append(op)

    def remove_op(self, op: Op) -> None:
        self.ops.remove(op)

    def to_namespace(self, namespace: str) -> "Graph":
        ...


def create_op(type: str) -> Op:
    return Op(attrs={"type": type})


def connect(src: Op, dst: Op) -> Edge:
    edge = Edge(src=src, dst=dst)
    src.out_edges.append(edge)
    dst.in_edges.append(edge)
