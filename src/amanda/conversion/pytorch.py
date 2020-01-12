from typing import Dict

from torch._C import Graph as TorchGraph
from torch._C import Node as TorchNode

from amanda import Graph, Op
from amanda.namespace import Namespace, default_namespace, get_global_registry
from amanda.rule import OpMapping, Rule, RuleMapper

_namespace = default_namespace() / Namespace("pytorch")
_internal_namespace = _namespace / Namespace("internal")


def pytorch_namespace() -> Namespace:
    return _namespace


def pytorch_internal_namespace() -> Namespace:
    return _internal_namespace


class ToDefaultRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        for key in ["name", "type"]:
            if pytorch_namespace().qualified(key) in op.attrs:
                op.attrs[key] = op.attrs[pytorch_namespace().qualified(key)]
        return OpMapping(source_ops=[op], target_ops=[op])


class ToPyTorchRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        for key in ["name", "type"]:
            if default_namespace().qualified(key) in op.attrs:
                op.attrs[key] = op.attrs[default_namespace().qualified(key)]
        return OpMapping(source_ops=[op], target_ops=[op])


_pytorch_to_default_mapper = RuleMapper(rules=[ToDefaultRule()])
_default_to_pytorch_mapper = RuleMapper(rules=[ToPyTorchRule()])


def pytorch_to_default_mapper() -> RuleMapper:
    return _pytorch_to_default_mapper


def default_to_pytorch_mapper() -> RuleMapper:
    return _default_to_pytorch_mapper


get_global_registry().add_mapper(
    pytorch_namespace(), default_namespace(), pytorch_to_default_mapper()
)
get_global_registry().add_mapper(
    default_namespace(), pytorch_namespace(), default_to_pytorch_mapper()
)


def import_from_graph(torch_graph: TorchGraph) -> Graph:
    graph = Graph()
    node_to_op: Dict[TorchNode, Op] = {}
    node: TorchNode

    def add_op(node: TorchNode):
        if node in node_to_op:
            return
        for input in node.inputs():
            add_op(input.node())
        for block in node.blocks():
            for inner_node in block.nodes():
                add_op(inner_node)
        op = Op(
            attrs={},
            input_tensors=[
                node_to_op[input.node()].output_tensor(input.offset())
                for input in node.inputs()
            ],
            control_dependencies=[],
            output_num=node.outputsSize(),
        )
        op.type = node.kind()
        graph.add_op(op)
        node_to_op[node] = op

    for node in torch_graph.nodes():
        add_op(node)

    return graph