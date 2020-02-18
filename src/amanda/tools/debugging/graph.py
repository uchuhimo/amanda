# type: ignore
from typing import Any, Callable, Dict, List, Tuple, Union


class Tensor:
    attrs: Dict[str, Any]


class InputPort:
    op: "Op"
    index: int

    # additional APIs for convenience
    @property
    def in_edges(self) -> List["Edge"]:
        return [edge for edge in edges.values() if edge.dst == self]


class OutputPort:
    op: "Op"
    index: int
    tensor: Tensor

    # additional APIs for convenience
    @property
    def out_edges(self) -> List["Edge"]:
        return [edge for edge in edges.values() if edge.src == self]


class Op:
    attrs: Dict[str, Any]
    input_ports: List[InputPort]
    output_ports: List[OutputPort]
    control_input_port: InputPort
    control_output_port: OutputPort

    # builtin attributes
    name: str  # is attrs["name"]
    type: str  # is attrs["type"]

    # additional APIs for convenience
    @property
    def input_ops(self) -> List["Op"]:
        return [
            edge.src.op
            for input_port in self.input_ports
            for edge in input_port.in_edges
        ]

    @property
    def output_ops(self) -> List["Op"]:
        return [
            edge.dst.op
            for output_port in self.output_ports
            for edge in output_port.out_edges
        ]

    @property
    def input_tensors(self) -> List[Tensor]:
        return [
            edge.src.tensor
            for input_port in self.input_ports
            for edge in input_port.in_edges
        ]

    @property
    def output_tensors(self) -> List[Tensor]:
        return [output_port.tensor for output_port in self.output_ports]

    def insert_op_before(self, op: "Op"):
        assert len(self.input_ports) == len(op.input_ports) == len(op.output_ports)
        for input_port in self.input_ports:
            index = input_port.index
            for edge in input_port.in_edges:
                create_edge(src=edge.src, dst=op.input_ports[index])
                remove_edge(edge)
            create_edge(src=op.output_ports[index], dst=input_port)

    def insert_op_after(self, op: "Op"):
        assert len(self.output_ports) == len(op.input_ports) == len(op.output_ports)
        for output_port in self.output_ports:
            index = output_port.index
            for edge in output_port.out_edges:
                create_edge(src=op.output_ports[index], dst=edge.dst)
                remove_edge(edge)
            create_edge(src=output_port, dst=op.input_ports[index])


class Edge:
    src: OutputPort
    dst: InputPort
    attrs: Dict[str, Any]

    # additional APIs for convenience
    def insert_op(self, op: Op):
        assert len(op.input_ports) == 1 and len(op.output_ports) == 1
        create_edge(src=self.src, dst=op.input_ports[0])
        create_edge(src=op.output_ports[0], dst=self.dst)
        remove_edge(self)


class Graph:
    ops: List[Op]
    attrs: Dict[str, Any]

    # builtin attributes
    namespace: str  # is attrs["namespace"]

    def add_op(self, op: Op) -> None:
        self.ops.append(op)

    def remove_op(self, op: Op) -> None:
        self.ops.remove(op)

    def to_namespace(self, namespace: str, tags: List[str] = None) -> "Graph":
        ...


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


MatcherType = Union[
    str, List, Callable[[...], bool], None,
]
MapperType = Union[
    str, List, Callable[[...], Any], None,
]


class Rule:
    rule: Dict[str, Any]


def create_rule(rule: Dict[str, Any]) -> Rule:
    return Rule(rule=rule)


class MappingTable:
    def insert_rule(self, rule: Rule, index: int = 0):
        ...

    def remove_rule(self, index: int):
        ...

    def get_rule(self, index: int) -> Rule:
        ...

    def get_rules(self) -> List[Rule]:
        ...

    def save(self, file):
        ...

    def load(self, file):
        ...


mapping_tables: Dict[Tuple[str, str], MappingTable] = {}


def get_mapping_table(src: str, dst: str) -> MappingTable:
    return mapping_tables[(src, dst)]


def load_mapping_tables(file):
    ...


def save_mapping_tables(file):
    ...
