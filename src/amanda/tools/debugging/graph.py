# type: ignore
from typing import Any, Dict, List


class Tensor:
    ...


class Op:
    in_edges: List["Edge"]
    out_edges: List["Edge"]
    attrs: Dict[str, Any]

    # builtin attributes
    name: str  # is attrs["name"]
    type: str  # is attrs["type"]
    output_tensor: List[Tensor]  # is attrs["output_tensor"]


class Edge:
    src: Op
    dst: Op
    attrs: Dict[str, Any]

    # builtin attributes
    tensor: Tensor  # is attrs["tensor"]

    def replace_src(self, src: Op):
        self.src.out_edges.remove(self)
        src.out_edges.append(self)
        self.src = src

    def insert_op(self, op: Op):
        new_edge1 = connect(src=self.src, dst=op)
        new_edge1.tensor = self.tensor
        new_edge2 = connect(src=op, dst=self.dst)
        new_edge2.tensor = op.output_tensor[0]
        disconnect(src=self.src, dst=self.dst)


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


def get_edge(src: Op, dst: Op) -> Edge:
    for edge in src.out_edges:
        if edge.dst == dst:
            return edge


def connect(src: Op, dst: Op) -> Edge:
    edge = Edge(src=src, dst=dst)
    src.out_edges.append(edge)
    dst.in_edges.append(edge)


def disconnect(src: Op, dst: Op):
    edge = get_edge(src=src, dst=dst)
    src.out_edges.remove(edge)
    dst.in_edges.remove(edge)
