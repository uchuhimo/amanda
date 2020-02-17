# type: ignore
from typing import Any, Callable, Dict, List, Tuple, Union


class Tensor:
    attrs: Dict[str, Any]


class InputPort:
    op: "Op"
    index: int

    # cached fields
    in_edges: List["Edge"]


class OutputPort:
    op: "Op"
    index: int

    # cached fields
    out_edges: List["Edge"]
    tensor: Tensor


class Op:
    attrs: Dict[str, Any]
    input_ports: List[InputPort]
    output_ports: List[OutputPort]

    # builtin attributes
    name: str  # is attrs["name"]
    type: str  # is attrs["type"]
    output_tensors: List[Tensor]  # is attrs["output_tensors"]

    # cached fields
    input_ops: List["Op"]
    output_ops: List["Op"]
    input_tensors: List[Tensor]


class Edge:
    src: OutputPort
    dst: InputPort
    attrs: Dict[str, Any]

    def insert_op(self, op: Op):
        ...


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


# global store
edges: Dict[Tuple[OutputPort, InputPort], Edge] = {}


def create_op(type: str) -> Op:
    return Op(attrs={"type": type})


def get_edge(src: OutputPort, dst: InputPort) -> Edge:
    return edges[(src, dst)]


def create_edge(src: OutputPort, dst: InputPort) -> Edge:
    edge = Edge(src=src, dst=dst)
    edges[(src, dst)] = edge
    return edge


def remove_edge(edge: Edge):
    del edges[(edge.src, edge.dst)]


ColumnType = Union[str, List, Callable, None]


class Rule:
    rule_id: int
    src_op: ColumnType
    src_attr_name: ColumnType
    src_attr_value: ColumnType
    dst_op: ColumnType
    dst_attr_name: ColumnType
    dst_attr_value: ColumnType
    tag: str = None


def create_rule(
    src_op: ColumnType,
    src_attr_name: ColumnType,
    src_attr_value: ColumnType,
    dst_op: ColumnType,
    dst_attr_name: ColumnType,
    dst_attr_value: ColumnType,
    tag: str = None,
) -> Rule:
    ...


class MappingTable:
    def insert_rule(self, rule: Rule, index: int = 0):
        ...

    def remove_rule(self, rule: Rule):
        ...

    def get_rule(self, rule_id: int) -> Rule:
        ...

    def get_rules(self) -> List[Rule]:
        ...


def get_mapping_table(src: str, dst: str) -> MappingTable:
    ...
