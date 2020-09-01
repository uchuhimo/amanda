import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Set, Union

import torch
import torch._C
import torch.jit
from torch._C import Graph as TorchGraph
from torch._C import Node as TorchNode
from torch._C import Value as TorchValue

from amanda.graph import Graph, Op, OpAttrKey
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


def import_from_func(func: torch.jit.ScriptFunction) -> Graph:
    return import_from_graph(func.graph)


def import_from_module(module: Union[torch.nn.Module, str, Path]) -> Graph:
    if isinstance(module, (str, Path)):
        module = torch.jit.load(module)
    if not isinstance(module, torch.jit.ScriptModule):
        module = torch.jit.script(module)
    torch_graph = module.graph
    torch_graph, params = torch._C._jit_pass_lower_graph(torch_graph, module._c)
    graph = import_from_graph(torch_graph)
    graph.attrs["training"] = module.training
    graph.attrs["params"] = params
    return graph


@dataclass(frozen=True)
class TorchTensor:
    op_name: str
    output_index: int


def get_name(op: Op) -> str:
    return op.attrs.get(default_namespace().qualified("name"), default="")


def import_from_graph(torch_graph: TorchGraph) -> Graph:
    graph = Graph()
    node_to_op: Dict[TorchNode, Op] = {}
    node: TorchNode
    op_type_counter: Counter = Counter()

    torch._C._jit_pass_inline(torch_graph)

    def add_op(node: TorchNode):
        if node in node_to_op:
            return
        for input in node.inputs():
            add_op(input.node())
        for block in node.blocks():
            for inner_node in block.nodes():
                add_op(inner_node)
        attr_kinds: Dict[str, str] = {}
        op = Op(
            attrs={
                attr_name: from_ir_attr(node, attr_name, attr_kinds)
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
        op.attrs[pytorch_internal_namespace().qualified("attr_kinds")] = attr_kinds
        count = op_type_counter[op.type]
        op_type_counter[op.type] += 1
        op.attrs[default_namespace().qualified("name")] = f"{op.type}_{count}"
        for output_tensor, output_value in zip(op.output_tensors, node.outputs()):
            output_tensor.attrs["type"] = output_value.type()
        graph.add_op(op)
        node_to_op[node] = op

    for input_value in torch_graph.inputs():
        add_op(input_value.node())
    for node in torch_graph.nodes():
        add_op(node)

    graph.attrs["inputs"] = [
        TorchTensor(
            op_name=get_name(node_to_op[input_value.node()]),
            output_index=input_value.offset(),
        )
        for input_value in torch_graph.inputs()
    ]
    graph.attrs["outputs"] = [
        TorchTensor(
            op_name=get_name(node_to_op[output_value.node()]),
            output_index=output_value.offset(),
        )
        for output_value in torch_graph.outputs()
    ]
    graph.namespace = pytorch_namespace()
    return graph


def export_to_graph(graph: Graph) -> TorchGraph:
    if graph.namespace != pytorch_namespace():
        graph = graph.to_default_namespace().to_namespace(pytorch_namespace())
    torch_graph = TorchGraph()
    op_to_node: Dict[str, TorchNode] = {}
    input_ops: Set[str] = set()
    input_tensors: Dict[TorchTensor, TorchValue] = {}
    for input_tensor in graph.attrs["inputs"]:
        input_ops.add(input_tensor.op_name)
        input_tensors[input_tensor] = torch_graph.addInput()
    for op in graph.sorted_ops:
        if get_name(op) in input_ops:
            continue
        op_input_tensors = [
            TorchTensor(get_name(input_tensor.op), input_tensor.output_index)
            for input_tensor in op.input_tensors
        ]
        node: TorchNode = torch_graph.create(
            op.type,
            [
                input_tensors[input_tensor]
                if input_tensor in graph.attrs["inputs"]
                else op_to_node[input_tensor.op_name].outputsAt(
                    input_tensor.output_index
                )
                for input_tensor in op_input_tensors
            ],
            op.output_num,
        )
        attrs = without_internal_attrs(op.attrs)
        for attr_name, attr_value in attrs.items():
            set_ir_attr(node, attr_name, attr_value, op)
        for output_tensor, output_value in zip(op.output_tensors, node.outputs()):
            output_value.setType(output_tensor.attrs["type"])
        op_to_node[get_name(op)] = node
        torch_graph.appendNode(node)
    for output_torch_tensor in graph.attrs["outputs"]:
        if output_torch_tensor in graph.attrs["inputs"]:
            torch_graph.registerOutput(input_tensors[output_torch_tensor])
        else:
            torch_graph.registerOutput(
                op_to_node[output_torch_tensor.op_name].outputsAt(
                    output_torch_tensor.output_index
                )
            )
    return torch_graph


def export_to_module(graph: Graph, file: Union[str, Path] = None) -> torch.nn.Module:
    torch_graph = export_to_graph(graph)
    forward_func = torch._C._create_function_from_graph("forward", torch_graph)
    module = torch.nn.Module()
    module.forward = types.MethodType(
        lambda self, *args: forward_func(*args, *graph.attrs["params"]), module
    )
    if "training" in graph.attrs:
        module.train(graph.attrs["training"])
    if file is not None:
        torch.jit.save(module, file)
    return module


def without_internal_attrs(attrs):
    return {
        name: value
        for name, value in attrs.items()
        if not (is_qualified(name) or name in ["type", "scope", "sourceRange"])
    }


def from_ir_attr(node: TorchNode, attr_name: str, attr_kinds: Dict[str, str]) -> Any:
    attr_kind = node.kindOf(attr_name)
    attr_kinds[attr_name] = attr_kind
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


def infer_attr_kind(attr_value: Any) -> str:
    if isinstance(attr_value, float):
        return "f"
    elif isinstance(attr_value, int):
        return "i"
    elif isinstance(attr_value, str):
        return "s"
    elif isinstance(attr_value, torch.Tensor):
        return "t"
    elif isinstance(attr_value, TorchGraph):
        return "g"
    elif isinstance(attr_value, list):
        if len(attr_value) == 0:
            raise RuntimeError(f"cannot infer attr kind for empty list")
        return infer_attr_kind(attr_value[0])
    else:
        raise RuntimeError(f"cannot infer attr kind for {attr_value}")


def set_ir_attr(node: TorchNode, attr_name: str, attr_value: Any, op: Op) -> None:
    if pytorch_internal_namespace().qualified("attr_kinds") in op.attrs:
        attr_kind = op.attrs[pytorch_internal_namespace().qualified("attr_kinds")][
            attr_name
        ]
    else:
        attr_kind = infer_attr_kind(attr_value)
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


import_types = {
    "torchscript": import_from_module,
}

export_types = {
    "torchscript": export_to_module,
}
