from typing import Any, Dict

from torch._C import Graph as TorchGraph
from torch._C import Node as TorchNode

from amanda import Graph, Op
from amanda.graph import OpAttrKey
from amanda.namespace import (
    Namespace,
    default_namespace,
    get_global_registry,
    is_qualified,
)
from amanda.rule import OpMapping, Rule, RuleMapper

_namespace = default_namespace() / Namespace("pytorch")
_internal_namespace = _namespace / Namespace("internal")


def pytorch_namespace() -> Namespace:
    return _namespace


def pytorch_internal_namespace() -> Namespace:
    return _internal_namespace


class ToDefaultRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        for key in ["type"]:
            if pytorch_namespace().qualified(key) in op.attrs:
                op.attrs[key] = op.attrs[pytorch_namespace().qualified(key)]
        return OpMapping(source_ops=[op], target_ops=[op])


class ToPyTorchRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        for key in ["type"]:
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
            attrs={
                attr_name: from_ir_attr(node, attr_name)
                for attr_name in node.attributeNames()
            },
            input_tensors=[
                node_to_op[input.node()].output_tensor(input.offset())
                for input in node.inputs()
            ],
            control_dependencies=[],
            output_num=node.outputsSize(),
        )
        op.type = node.kind()
        op.attrs["scope"] = node.scopeName()
        op.attrs["sourceRange"] = node.sourceRange()
        op.attrs[OpAttrKey.has_name] = False
        graph.add_op(op)
        node_to_op[node] = op

    for node in torch_graph.nodes():
        add_op(node)

    graph.attrs["inputs"] = [
        node_to_op[input_value.node()].output_tensor(input_value.offset())
        for input_value in torch_graph.inputs()
    ]
    graph.attrs["outputs"] = [
        node_to_op[output_value.node()].output_tensor(output_value.offset())
        for output_value in torch_graph.outputs()
    ]
    return graph


def export_to_graph(graph: Graph) -> TorchGraph:
    if graph.namespace != pytorch_namespace():
        graph = graph.to_default_namespace().to_namespace(pytorch_namespace())
    torch_graph = TorchGraph()
    op_to_node: Dict[Op, TorchNode] = {}
    for op in graph.sorted_ops:
        node = torch_graph.create(
            op.type,
            [
                op_to_node[input_tensor.op].outputsAt(input_tensor.output_index)
                for input_tensor in op.input_tensors
            ],
            op.output_num,
        )
        attrs = without_internal_attrs(op.attrs)
        for attr_name, attr_value in attrs.items():
            set_ir_attr(node, attr_name, attr_value)
        op_to_node[op] = node
        torch_graph.appendNode(node)
    # for input_tensor in graph.attrs["inputs"]:
    #     torch_graph.addInput(op_to_node[input_tensor.op].outputsAt(input_tensor.output_index))
    # torch_graph.lint()
    return torch_graph


def without_internal_attrs(attrs):
    return {
        name: value
        for name, value in attrs.items()
        if not (is_qualified(name) or name in ["type", "scope", "sourceRange"])
    }


def from_ir_attr(node: TorchNode, attr_name: str) -> Any:
    attr_kind = node.kindOf(attr_name)
    if attr_kind == "f":
        return node.f(attr_name)
    elif attr_kind == "fs":
        return node.fs(attr_name)
    elif attr_kind == "i":
        return node.i(attr_name)
    elif attr_kind == "is":
        return getattr(node, "is")(attr_name)
    elif attr_kind == "s":
        return node.s(attr_name)
    elif attr_kind == "ss":
        return node.ss(attr_name)
    elif attr_kind == "t":
        return node.t(attr_name)
    elif attr_kind == "ts":
        return node.ts(attr_name)
    elif attr_kind == "g":
        return node.g(attr_name)
    elif attr_kind == "gs":
        return node.gs(attr_name)
    else:
        raise RuntimeError(f"cannot import from attr {attr_name} in node {node}")


def set_ir_attr(node: TorchNode, attr_name: str, attr_value: Any) -> None:
    attr_kind = node.kindOf(attr_name)
    if attr_kind == "f":
        node.f_(attr_name, attr_value)
    elif attr_kind == "fs":
        node.fs_(attr_name, attr_value)
    elif attr_kind == "i":
        node.i_(attr_name, attr_value)
    elif attr_kind == "is":
        node.is_(attr_name, attr_value)
    elif attr_kind == "s":
        node.s_(attr_name, attr_value)
    elif attr_kind == "ss":
        node.ss_(attr_name, attr_value)
    elif attr_kind == "t":
        node.t_(attr_name, attr_value)
    elif attr_kind == "ts":
        node.ts_(attr_name, attr_value)
    elif attr_kind == "g":
        node.g_(attr_name, attr_value)
    elif attr_kind == "gs":
        node.gs_(attr_name, attr_value)
    else:
        raise RuntimeError(
            f"cannot export {attr_value} to attr {attr_name} in node {node}"
        )
