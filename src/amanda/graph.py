import copy
import itertools
import json
import typing
import uuid
from collections import OrderedDict
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, List, Set, Union

from amanda.attributes import Attributes
from amanda.exception import IrremovableOpError
from amanda.namespace import Namespace, Registry, default_namespace, get_global_registry
from amanda.type import DataType, unknown_type


def parse_ports(
    ports: typing.Union["OrderedDict[str, DataType]", List[str], int]
) -> "OrderedDict[str, DataType]":
    if isinstance(ports, int):
        ports = [str(index) for index in range(ports)]
    if isinstance(ports, list):
        ports = OrderedDict((name, unknown_type) for name in ports)
    return ports


@dataclass
class Op:
    name: str = None
    type: str = None
    namespace: Namespace = None
    graph: "Graph" = None
    attrs: Attributes = field(default_factory=Attributes)
    name_to_input_port: "OrderedDict[str, InputPort]" = field(init=False)
    name_to_output_port: "OrderedDict[str, OutputPort]" = field(init=False)
    control_input_port: "InputPort" = field(init=False)
    control_output_port: "OutputPort" = field(init=False)

    inputs: InitVar[typing.Union["OrderedDict[str, DataType]", List[str], int]] = None
    outputs: InitVar[typing.Union["OrderedDict[str, DataType]", List[str], int]] = None

    CONTROL_PORT_NAME: typing.ClassVar[str] = "^control"
    CONTROL_PORT_TYPE: typing.ClassVar[DataType] = DataType(
        namespace=default_namespace(), name="ControlPort"
    )

    def __post_init__(self, inputs, outputs):
        self.control_input_port = create_control_input_port(self)
        self.control_output_port = create_control_output_port(self)
        self.name_to_input_port = OrderedDict(
            (name, InputPort(self, name, type))
            for name, type in parse_ports(inputs if inputs is not None else 1).items()
        )
        self.name_to_output_port = OrderedDict(
            (name, OutputPort(self, name, type))
            for name, type in parse_ports(outputs if outputs is not None else 1).items()
        )

    @property
    def input_ports(self) -> List["InputPort"]:
        return list(self.name_to_input_port.values())

    @property
    def output_ports(self) -> List["OutputPort"]:
        return list(self.name_to_output_port.values())

    @property
    def input_port_names(self) -> List[str]:
        return list(self.name_to_input_port)

    @property
    def output_port_names(self) -> List[str]:
        return list(self.name_to_output_port)

    @property
    def input_num(self) -> int:
        return len(self.name_to_input_port)

    @property
    def output_num(self) -> int:
        return len(self.name_to_output_port)

    def input_port(self, index: Union[int, str]) -> "InputPort":
        if isinstance(index, int):
            if not (0 <= index < self.input_num):
                raise IndexError
            return self.input_ports[index]
        else:
            if index == Op.CONTROL_PORT_NAME:
                return self.control_input_port
            else:
                return self.name_to_input_port[index]

    def output_port(self, index: Union[int, str]) -> "OutputPort":
        if isinstance(index, int):
            if not (0 <= index < self.output_num):
                raise IndexError
            return self.output_ports[index]
        else:
            if index == Op.CONTROL_PORT_NAME:
                return self.control_output_port
            else:
                return self.name_to_output_port[index]

    @property
    def control_dependencies(self) -> List["Op"]:
        return self.control_input_port.in_ops

    @property
    def in_ops(self) -> List["Op"]:
        return [edge.src.op for edge in self.in_edges]

    @property
    def out_ops(self) -> List["Op"]:
        return [edge.dst.op for edge in self.out_edges]

    @property
    def in_edges(self) -> List["Edge"]:
        return [edge for port in self.input_ports for edge in port.in_edges]

    @property
    def out_edges(self) -> List["Edge"]:
        return [edge for port in self.output_ports for edge in port.out_edges]

    def copy_op_with(self, copy_func) -> "Op":
        op = Op(
            name=self.name,
            type=self.type,
            namespace=self.namespace,
            attrs=copy_func(self.attrs),
            inputs=OrderedDict([(port.name, port.type) for port in self.input_ports]),
            outputs=OrderedDict([(port.name, port.type) for port in self.output_ports]),
        )
        return op

    def copy(self) -> "Op":
        """
        Return a shallow copy of the current op.
        Attribute values are not copied.
        They will be shared between these two ops.
        If you don't want to share attribute values, you can use `Op.deepcopy` instead.
        """
        return self.copy_op_with(lambda attrs: attrs.copy())

    def __copy__(self):
        return self.copy()

    def deepcopy(self) -> "Op":
        """
        Return a deep copy of the current op.
        """
        return self.copy_op_with(copy.deepcopy)

    def __deepcopy__(self, memodict={}):
        return self.deepcopy()

    @property
    def path(self) -> List[str]:
        return self.graph.path + [self.name]

    def __repr__(self) -> str:
        items = [
            f"name={self.name}",
            f"type={self.type}",
        ] + self.attrs.to_item_strings()
        attrs_string = ", ".join(items)
        return f"Op({attrs_string})"

    def __hash__(self) -> int:
        return id(self)

    def dict(self, ignore_attrs: bool = False):
        result: Dict[str, Any] = dict(
            name=self.name,
            type=self.type,
            input_ports=[
                port.name if port.type == unknown_type else f"{port.name}: {port.type}"
                for port in self.input_ports
            ],
            output_ports=[
                port.name if port.type == unknown_type else f"{port.name}: {port.type}"
                for port in self.output_ports
            ],
        )
        if not ignore_attrs:
            result["attrs"] = dict(self.attrs)
        if self.namespace is not None:
            result["namespace"] = self.namespace.full_name
        return result

    def json(self):
        return json.dumps(self.dict(), indent=4)


def create_op(
    type: str,
    name: str = None,
    namespace: Namespace = None,
    attrs: Dict[str, Any] = None,
    inputs: typing.Union["OrderedDict[str, DataType]", List[str], int] = None,
    outputs: typing.Union["OrderedDict[str, DataType]", List[str], int] = None,
) -> Op:
    name = name or f"{type}_{uuid.uuid4()}"
    attrs = attrs or {}
    op = Op(
        type=type,
        name=name,
        namespace=namespace,
        attrs=Attributes(attrs),
        inputs=inputs,
        outputs=outputs,
    )
    return op


@dataclass(frozen=True)
class OutputPort:
    op: Op
    name: str
    type: DataType

    def is_control(self) -> bool:
        return self.name == Op.CONTROL_PORT_NAME

    @property
    def out_edges(self) -> List["Edge"]:
        graph: Graph
        if isinstance(self.op, SubGraph) and (
            self in self.op.input_ports or self == self.op.control_output_port
        ):
            graph = self.op
        else:
            graph = self.op.graph
        if graph is None:
            return []
        else:
            return [edge for edge in graph.edges if edge.src == self]

    @property
    def out_ops(self) -> List[Op]:
        return list(map(lambda edge: edge.dst.op, self.out_edges))

    def __hash__(self):
        return hash((self.op, self.name))


def create_control_output_port(op: Op) -> OutputPort:
    return OutputPort(op=op, name=Op.CONTROL_PORT_NAME, type=Op.CONTROL_PORT_TYPE)


@dataclass(frozen=True)
class InputPort:
    op: Op
    name: str
    type: DataType

    def is_control(self) -> bool:
        return self.name == Op.CONTROL_PORT_NAME

    @property
    def in_edges(self) -> List["Edge"]:
        graph: Graph
        if isinstance(self.op, SubGraph) and (
            self in self.op.output_ports or self == self.op.control_input_port
        ):
            graph = self.op
        else:
            graph = self.op.graph
        if graph is None:
            return []
        else:
            return [edge for edge in graph.edges if edge.dst == self]

    @property
    def in_ops(self) -> List[Op]:
        return list(map(lambda edge: edge.src.op, self.in_edges))

    def __hash__(self):
        return hash((self.op, self.name))


@dataclass(frozen=True)
class IoPort(InputPort, OutputPort):
    def __hash__(self):
        return hash((self.op, self.name))


def create_control_input_port(op: Op) -> InputPort:
    return InputPort(op=op, name=Op.CONTROL_PORT_NAME, type=Op.CONTROL_PORT_TYPE)


@dataclass
class Edge:
    src: OutputPort
    dst: InputPort
    graph: "Graph" = None
    attrs: Attributes = field(default_factory=Attributes)

    def is_control_edge(self) -> bool:
        return self.src.is_control() and self.dst.is_control()

    def insert_op(self, op: Op):
        assert op.input_num == 1 and op.output_num == 1
        self.graph.create_edge(src=self.src, dst=op.input_port(0))
        self.graph.create_edge(src=op.output_port(0), dst=self.dst)
        self.graph.remove_edge(self)

    def __hash__(self) -> int:
        return hash((self.src, self.dst))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Edge) and self.src == other.src and self.dst == other.dst
        )

    def __repr__(self) -> str:
        return (
            f"Edge({self.src.op.name}.{self.src.name} ->"
            f"{self.dst.op.name}.{self.dst.name})"
        )


def create_edge(
    src: OutputPort,
    dst: InputPort,
    graph: "Graph" = None,
    attrs: Dict[str, Any] = None,
) -> Edge:
    attrs = attrs or {}
    return Edge(src=src, dst=dst, attrs=Attributes(attrs), graph=graph)


def create_control_edge(
    src: Op,
    dst: Op,
    graph: "Graph" = None,
    attrs: Dict[str, Any] = None,
) -> Edge:
    return create_edge(
        src=src.control_output_port,
        dst=dst.control_input_port,
        attrs=attrs,
        graph=graph,
    )


@dataclass
class Graph:
    name: str = "graph"
    name_to_op: Dict[str, Op] = field(default_factory=dict)
    index_to_edge: Dict[typing.Tuple[OutputPort, InputPort], Edge] = field(
        default_factory=dict
    )
    attrs: Attributes = field(default_factory=Attributes)
    namespace: Namespace = None
    name_to_subgraph: Dict[str, "SubGraph"] = None
    _cached_post_order_ops: List[Op] = None

    def __post_init__(self):
        if self.name_to_subgraph is None:
            self.name_to_subgraph = {
                op.name: op for op in self.ops if isinstance(op, SubGraph)
            }

    @property
    def ops(self) -> List[Op]:
        return list(self.name_to_op.values())

    @property
    def edges(self) -> List[Edge]:
        return list(self.index_to_edge.values())

    @property
    def subgraphs(self) -> List["SubGraph"]:
        return list(self.name_to_subgraph.values())

    def add_op(self, op: Op) -> None:
        if op.name in self.name_to_op:
            raise KeyError(f"op {op.name} has been in the graph")
        self.name_to_op[op.name] = op
        if isinstance(op, SubGraph):
            self.name_to_subgraph[op.name] = op
        op.graph = self
        self.invalidate_cache()

    def is_removable(self, op: Op) -> bool:
        for other_op in self.ops:
            if other_op != op and (
                op in other_op.in_ops or op in other_op.control_dependencies
            ):
                return False
        return True

    def remove_op(self, op: Op) -> None:
        if not self.is_removable(op):
            raise IrremovableOpError
        assert op.name in self.name_to_op
        del self.name_to_op[op.name]
        if isinstance(op, SubGraph):
            del self.name_to_subgraph[op.name]
        op.graph = None
        self.invalidate_cache()

    def get_op(self, name: str) -> typing.Optional[Op]:
        return self.name_to_op[name] if name in self.name_to_op else None

    def get_edge(self, src: OutputPort, dst: InputPort) -> Edge:
        return self.index_to_edge[(src, dst)]

    def get_control_edge(self, src: Op, dst: Op) -> Edge:
        return self.index_to_edge[(src.control_output_port, dst.control_input_port)]

    def create_edge(self, src: OutputPort, dst: InputPort) -> Edge:
        edge = create_edge(src=src, dst=dst)
        self.add_edge(edge)
        return edge

    def create_control_edge(self, src: Op, dst: Op) -> Edge:
        return self.create_edge(
            src=src.control_output_port,
            dst=dst.control_input_port,
        )

    def add_edge(self, edge: Edge):
        if not isinstance(self, SubGraph) and edge.src.op.name not in self.name_to_op:
            raise KeyError(
                f"{edge.src.op.name} is not in graph when adding edge {edge}"
            )
        if not isinstance(self, SubGraph) and edge.dst.op.name not in self.name_to_op:
            raise KeyError(
                f"{edge.dst.op.name} is not in graph when adding edge {edge}"
            )
        self.index_to_edge[(edge.src, edge.dst)] = edge
        edge.graph = self
        self.invalidate_cache()

    def remove_edge(self, edge: Edge):
        del self.index_to_edge[(edge.src, edge.dst)]
        edge.graph = None
        self.invalidate_cache()

    @property
    def sorted_ops(self) -> List[Op]:
        """
        Return a topologically sorted list of ops that forms this graph.
        """
        if self._cached_post_order_ops is not None:
            return self._cached_post_order_ops
        upstream_ops = set(
            itertools.chain.from_iterable(
                map(
                    lambda op: self.lift_ops(op.in_ops + op.control_dependencies),
                    self.ops,
                )
            )
        )
        output_ops = set(self.ops) - upstream_ops
        dfs_stack = list(output_ops)
        walked_ops: Set[Op] = set()
        returned_ops: Set[Op] = set()
        post_order_ops: List[Op] = []

        while len(dfs_stack) > 0:
            current_op = dfs_stack.pop()
            if current_op in walked_ops:
                walked_ops.remove(current_op)
                returned_ops.add(current_op)
                post_order_ops.append(current_op)
            elif current_op not in returned_ops:
                dfs_stack.append(current_op)
                walked_ops.add(current_op)
                for input_op in self.lift_ops(
                    current_op.in_ops + current_op.control_dependencies
                ):
                    if input_op.name in self.name_to_op:
                        dfs_stack.append(input_op)
        self._cached_post_order_ops = post_order_ops
        return post_order_ops

    @property
    def data_edges(self) -> List[Edge]:
        return [edge for edge in self.edges if not edge.is_control_edge()]

    @property
    def control_edges(self) -> List[Edge]:
        return [edge for edge in self.edges if edge.is_control_edge()]

    def lift_op(self, op: Op) -> Op:
        if len(self.name_to_subgraph) != 0:
            for subgraph in self.subgraphs:
                if op in subgraph:
                    return subgraph
        return op

    def lift_ops(self, ops: List[Op]) -> List[Op]:
        if len(self.name_to_subgraph) != 0:
            return list(map(lambda op: self.lift_op(op), ops))
        else:
            return ops

    def __contains__(self, op: Op) -> bool:
        if op.name in self.name_to_op:
            return True
        for subgraph in self.subgraphs:
            if op in subgraph:
                return True
        return False

    def invalidate_cache(self):
        self._cached_post_order_ops = None

    def to_namespace(
        self, namespace: typing.Union[Namespace, str], registry: Registry = None
    ) -> "Graph":
        if isinstance(namespace, str):
            namespace = Namespace(namespace)
        if namespace == self.namespace:
            return self
        else:
            registry = registry or get_global_registry()
            return registry.get_mapper(self.namespace, namespace).map(self, namespace)

    def to_default_namespace(self) -> "Graph":
        return self.to_namespace(default_namespace())

    def copy_graph_with(self, op_copy_func, attrs_copy_func) -> "Graph":
        old_op_to_new_op = {op: op_copy_func(op) for op in self.ops}
        new_graph = Graph(
            name_to_op={op.name: old_op_to_new_op[op] for op in self.ops},
            attrs=attrs_copy_func(self.attrs),
            namespace=self.namespace,
            name_to_subgraph={
                subgraph.name: old_op_to_new_op[subgraph] for subgraph in self.subgraphs
            },
            _cached_post_order_ops=[
                old_op_to_new_op[op] for op in self._cached_post_order_ops
            ]
            if self._cached_post_order_ops is not None
            else None,
        )
        for new_op in old_op_to_new_op.values():
            new_op.graph = new_graph
        for (src, dst), edge in self.index_to_edge.items():
            if edge.is_control_edge():
                new_edge = create_control_edge(
                    src=old_op_to_new_op[src.op],
                    dst=old_op_to_new_op[dst.op],
                    graph=new_graph,
                    attrs=attrs_copy_func(edge.attrs),
                )
            else:
                new_edge = create_edge(
                    src=old_op_to_new_op[src.op].output_port(src.name),
                    dst=old_op_to_new_op[dst.op].input_port(dst.name),
                    graph=new_graph,
                    attrs=attrs_copy_func(edge.attrs),
                )
            new_graph.index_to_edge[(new_edge.src, new_edge.dst)] = new_edge
        return new_graph

    def copy(self) -> "Graph":
        """
        Return a shallow copy of the current graph.
        Attribute values of graph, ops and edges are not copied.
        They will be shared between these two graphs.
        If you don't want to share attribute values,
        you can use `Graph.deepcopy` instead.
        """
        return self.copy_graph_with(
            op_copy_func=lambda op: op.copy(),
            attrs_copy_func=lambda attrs: attrs.copy(),
        )

    def __copy__(self):
        return self.copy()

    def deepcopy(self) -> "Graph":
        """
        Return a deep copy of the current graph.
        """
        return self.copy_graph_with(
            op_copy_func=lambda op: op.deepcopy(),
            attrs_copy_func=copy.deepcopy,
        )

    def __deepcopy__(self, memodict={}):
        return self.deepcopy()

    @property
    def path(self) -> List[str]:
        return []

    def __hash__(self) -> int:
        return id(self)

    def dict(self, ignore_attrs: bool = False):
        result = dict(
            name=self.name,
            ops={op.name: op.dict(ignore_attrs) for op in self.ops},
            edges={
                f"{edge.src.op.name}.{edge.src.name} -> "
                f"{edge.dst.op.name}.{edge.dst.name}"
                for edge in self.edges
            },
        )
        if not ignore_attrs:
            result["attrs"] = dict(self.attrs)
        if self.namespace is not None:
            result["namespace"] = self.namespace.full_name
        return result

    def json(self):
        return json.dumps(self.dict(), indent=4)

    def dump_to_str(self) -> str:
        strs = ["Graph("]
        fields = [f"name={self.name}"]
        if self.namespace is not None:
            fields.append(f"namespace={self.namespace}")
        fields = fields + self.attrs.to_item_strings()
        fields.append("edges=[")
        edges = []
        for op in self.sorted_ops:
            out_edges = [
                f"{edge.src.name}->{edge.dst.op.name}.{edge.dst.name}"
                for edge in op.out_edges
            ]
            out_edges_str = ", ".join(out_edges)
            edges.append(f"{repr(op)} -> [{out_edges_str}]")
        fields = fields + ["    " + edge for edge in edges] + ["]"]
        strs = strs + ["    " + field for field in fields] + [")\n"]
        return "\n".join(strs)

    def print(self) -> None:
        print(self.dump_to_str())


def create_graph(
    name: str = "graph",
    namespace: Namespace = None,
    attrs: Dict[str, Any] = None,
    ops: List[Op] = None,
    edges: List[Edge] = None,
) -> Graph:
    attrs = attrs or {}
    graph = Graph(name=name, namespace=namespace, attrs=Attributes(attrs))
    for op in ops or []:
        graph.add_op(op)
    for edge in edges or []:
        graph.add_edge(edge)
    return graph


@dataclass
class SubGraph(Op, Graph):
    name_to_input_port: "OrderedDict[str, IoPort]" = field(init=False)  # type: ignore
    name_to_output_port: "OrderedDict[str, IoPort]" = field(init=False)  # type: ignore
    control_input_port: IoPort = field(init=False)
    control_output_port: IoPort = field(init=False)

    def __post_init__(self, inputs, outputs):
        self.control_input_port = IoPort(
            op=self, name=Op.CONTROL_PORT_NAME, type=Op.CONTROL_PORT_TYPE
        )
        self.control_output_port = IoPort(
            op=self, name=Op.CONTROL_PORT_NAME, type=Op.CONTROL_PORT_TYPE
        )
        self.name_to_input_port = OrderedDict(
            (name, IoPort(self, name, type))
            for name, type in parse_ports(inputs if inputs is not None else 0).items()
        )
        self.name_to_output_port = OrderedDict(
            (name, IoPort(self, name, type))
            for name, type in parse_ports(outputs if outputs is not None else 0).items()
        )
        if self.name_to_subgraph is None:
            self.name_to_subgraph = {
                op.name: op for op in self.ops if isinstance(op, SubGraph)
            }

    @property
    def input_ports(self) -> List["IoPort"]:  # type: ignore
        return list(self.name_to_input_port.values())

    @property
    def output_ports(self) -> List["IoPort"]:  # type: ignore
        return list(self.name_to_output_port.values())

    def copy_graph_with(self, op_copy_func, attrs_copy_func) -> "SubGraph":
        old_op_to_new_op = {op: op_copy_func(op) for op in self.ops}
        new_graph = SubGraph(
            name=self.name,
            type=self.type,
            name_to_op={op.name: old_op_to_new_op[op] for op in self.ops},
            attrs=attrs_copy_func(self.attrs),
            namespace=self.namespace,
            inputs=OrderedDict([(port.name, port.type) for port in self.input_ports]),
            outputs=OrderedDict([(port.name, port.type) for port in self.output_ports]),
            name_to_subgraph={
                subgraph.name: old_op_to_new_op[subgraph] for subgraph in self.subgraphs
            },
            _cached_post_order_ops=[
                old_op_to_new_op[op] for op in self._cached_post_order_ops
            ]
            if self._cached_post_order_ops is not None
            else None,
        )
        for new_op in old_op_to_new_op.values():
            new_op.graph = new_graph
        old_op_to_new_op[self] = new_graph
        for (src, dst), edge in self.index_to_edge.items():
            if edge.is_control_edge():
                new_edge = create_edge(
                    src=old_op_to_new_op[src.op].control_output_port
                    if src.op != self
                    else new_graph.control_input_port,
                    dst=old_op_to_new_op[dst.op].control_input_port
                    if dst.op != self
                    else new_graph.control_output_port,
                    graph=new_graph,
                    attrs=attrs_copy_func(edge.attrs),
                )
            else:
                new_edge = create_edge(
                    src=old_op_to_new_op[src.op].output_port(src.name)
                    if src.op != self
                    else new_graph.input_port(src.name),
                    dst=old_op_to_new_op[dst.op].input_port(dst.name)
                    if dst.op != self
                    else new_graph.output_port(dst.name),
                    graph=new_graph,
                    attrs=attrs_copy_func(edge.attrs),
                )
            new_graph.index_to_edge[(new_edge.src, new_edge.dst)] = new_edge
        return new_graph

    def copy(self) -> "SubGraph":
        """
        Return a shallow copy of the current subgraph.
        Attribute values of subgraph, ops and edges are not copied.
        They will be shared between these two subgraphs.
        If you don't want to share attribute values,
        you can use `SubGraph.deepcopy` instead.
        """
        return self.copy_graph_with(
            op_copy_func=lambda op: op.copy(),
            attrs_copy_func=lambda attrs: attrs.copy(),
        )

    def deepcopy(self) -> "SubGraph":
        """
        Return a deep copy of the current subgraph.
        """
        return self.copy_graph_with(
            op_copy_func=lambda op: op.deepcopy(),
            attrs_copy_func=copy.deepcopy,
        )

    @property
    def path(self) -> List[str]:
        if self.graph is None:
            return []
        else:
            return self.graph.path + [self.name]

    def __repr__(self) -> str:
        items = [
            f"name={self.name}",
            f"type={self.type}",
        ] + self.attrs.to_item_strings()
        attrs_string = ", ".join(items)
        return f"SubGraph({attrs_string})"

    def __hash__(self) -> int:
        return id(self)

    def dict(self, ignore_attrs: bool = False):
        result = dict(
            name=self.name,
            type=self.type,
            input_ports=[
                port.name if port.type == unknown_type else f"{port.name}: {port.type}"
                for port in self.input_ports
            ],
            output_ports=[
                port.name if port.type == unknown_type else f"{port.name}: {port.type}"
                for port in self.output_ports
            ],
            ops={op.name: op.dict(ignore_attrs) for op in self.ops},
            edges={
                f"{edge.src.op.name}.{edge.src.name} -> "
                f"{edge.dst.op.name}.{edge.dst.name}"
                for edge in self.edges
            },
        )
        if not ignore_attrs:
            result["attrs"] = dict(self.attrs)
        if self.namespace is not None:
            result["namespace"] = self.namespace.full_name
        return result


def create_subgraph(
    type: str = None,
    name: str = None,
    namespace: Namespace = None,
    attrs: Dict[str, Any] = None,
    inputs: typing.Union["OrderedDict[str, DataType]", List[str], int] = None,
    outputs: typing.Union["OrderedDict[str, DataType]", List[str], int] = None,
    ops: List[Op] = None,
    edges: List[Edge] = None,
) -> SubGraph:
    if type is None:
        name = name or str(uuid.uuid4())
    else:
        name = name or f"{type}_{uuid.uuid4()}"
    attrs = attrs or {}
    op = SubGraph(
        type=type,
        name=name,
        namespace=namespace,
        attrs=Attributes(attrs),
        inputs=inputs,
        outputs=outputs,
    )
    for child_op in ops or []:
        op.add_op(child_op)
    for edge in edges or []:
        op.add_edge(edge)
    return op
