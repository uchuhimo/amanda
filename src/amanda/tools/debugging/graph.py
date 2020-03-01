# type: ignore
from typing import Any, Callable, Dict, List, Set, Tuple, Union


class Tensor:
    attrs: Dict[str, Any]


class InputPort:
    op: "Op"
    index: int
    attrs: Dict[str, Any]

    # builtin attributes
    name: str  # is attrs["name"]

    # additional APIs for convenience
    @property
    def in_edges(self) -> List["Edge"]:
        return [edge for edge in self.op.graph.edges.values() if edge.dst == self]


class OutputPort:
    op: "Op"
    index: int
    attrs: Dict[str, Any]

    # builtin attributes
    name: str  # is attrs["name"]

    # additional APIs for convenience
    @property
    def out_edges(self) -> List["Edge"]:
        return [edge for edge in self.op.graph.edges.values() if edge.src == self]


class Op:
    attrs: Dict[str, Any]
    input_ports: List[InputPort]
    output_ports: List[OutputPort]
    control_input_port: InputPort
    control_output_port: OutputPort

    # cached fields
    graph: "Graph"

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

    def insert_op_before(self, op: "Op"):
        assert len(self.input_ports) == len(op.input_ports) == len(op.output_ports)
        for input_port in self.input_ports:
            index = input_port.index
            for edge in input_port.in_edges:
                self.graph.create_edge(src=edge.src, dst=op.input_ports[index])
                self.graph.remove_edge(edge)
            self.graph.create_edge(src=op.output_ports[index], dst=input_port)

    def insert_op_after(self, op: "Op"):
        assert len(self.output_ports) == len(op.input_ports) == len(op.output_ports)
        for output_port in self.output_ports:
            index = output_port.index
            for edge in output_port.out_edges:
                self.graph.create_edge(src=op.output_ports[index], dst=edge.dst)
                self.graph.remove_edge(edge)
            self.graph.create_edge(src=output_port, dst=op.input_ports[index])


class Edge:
    src: OutputPort
    dst: InputPort
    attrs: Dict[str, Any]

    # cached fields
    graph: "Graph"

    # additional APIs for convenience
    def insert_op(self, op: Op):
        assert len(op.input_ports) == 1 and len(op.output_ports) == 1
        self.graph.create_edge(src=self.src, dst=op.input_ports[0])
        self.graph.create_edge(src=op.output_ports[0], dst=self.dst)
        self.graph.remove_edge(self)


class Graph:
    ops: List[Op]
    edges: Dict[Tuple[OutputPort, InputPort], Edge]
    attrs: Dict[str, Any]

    # builtin attributes
    namespace: str  # is attrs["namespace"]

    def add_op(self, op: Op) -> None:
        self.ops.append(op)

    def remove_op(self, op: Op) -> None:
        self.ops.remove(op)

    def get_edge(self, src: OutputPort, dst: InputPort) -> Edge:
        return self.edges[(src, dst)]

    def create_edge(self, src: OutputPort, dst: InputPort) -> Edge:
        edge = Edge(src=src, dst=dst)
        self.edges[(src, dst)] = edge
        return edge

    def remove_edge(self, edge: Edge):
        del self.edges[(edge.src, edge.dst)]

    def to_namespace(self, namespace: str, tags: Set[str] = None) -> "Graph":
        table = get_mapping_table(self.namespace, namespace)
        graph = self
        tags = tags or set()
        for rule in table.rules:
            if len(tags) == 0 or len(tags.intersection(rule.tags)) != 0:
                graph = rule.apply(graph)
        return graph


def create_op(type: str) -> Op:
    return Op(attrs={"type": type})


MatcherType = Union[
    str, List, Callable[[...], bool], None,
]
MapperType = Union[
    str, List, Callable[[...], Any], None,
]


class Rule:
    rule: Dict[str, Any]
    tags: List[str]

    def apply(self, graph: Graph) -> Graph:
        ...


def create_rule(rule: Dict[str, Any]) -> Rule:
    return Rule(rule=rule)


class MappingTable:
    def insert_rule(self, rule: Rule, index: int = 0):
        ...

    def remove_rule(self, index: int):
        ...

    def get_rule(self, index: int) -> Rule:
        ...

    @property
    def rules(self) -> List[Rule]:
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
