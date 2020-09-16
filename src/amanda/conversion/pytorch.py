import types
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, cast

import torch
import torch._C
import torch.jit
from torch._C import Graph as TorchGraph
from torch._C import Node as TorchNode
from torch._C import Type
from torch._C import Value as TorchValue
from torch.nn import Parameter

from amanda.exception import MismatchNamespaceError
from amanda.graph import (
    Graph,
    InputPort,
    Op,
    OutputPort,
    SubGraph,
    create_op,
    create_subgraph,
)
from amanda.namespace import Namespace, default_namespace, is_qualified
from amanda.type import DataType

_namespace = default_namespace() / Namespace("pytorch")
_internal_namespace = _namespace / Namespace("internal")
_type_namespace = Namespace("torch")


def pytorch_namespace() -> Namespace:
    return _namespace


def pytorch_internal_namespace() -> Namespace:
    return _internal_namespace


def pytorch_type_namespace() -> Namespace:
    return _type_namespace


class TorchType(DataType):
    def __init__(self, torch_type: Type):
        super().__init__(
            namespace=pytorch_type_namespace(), name=torch_type.str(), raw=torch_type
        )

    def __eq__(self, other):
        return isinstance(other, TorchType) and self.name == other.name


def import_from_func(func: torch.jit.ScriptFunction) -> Graph:
    return import_from_graph(func.graph)


def import_from_module(module: Union[torch.nn.Module, str, Path]) -> Graph:
    if isinstance(module, (str, Path)):
        module = torch.jit.load(module)
    if not isinstance(module, torch.jit.ScriptModule):
        module = torch.jit.script(module)
    torch_graph = module.graph
    torch_graph, params = torch._C._jit_pass_lower_graph(torch_graph, module._c)
    graph = import_from_graph(torch_graph, params)
    graph.attrs["training"] = module.training
    return graph


def import_from_graph(torch_graph: TorchGraph, params: List[Parameter] = None) -> Graph:
    torch._C._jit_pass_inline(torch_graph)
    torch._C._jit_pass_inline_fork_wait(torch_graph)
    params = params or []
    graph = create_subgraph(
        namespace=pytorch_namespace(),
        inputs=OrderedDict(
            [
                (str(index), TorchType(input.type()))
                for index, input in enumerate(
                    list(torch_graph.inputs())[: -len(params)]
                )
            ]
        ),
        outputs=OrderedDict(
            [
                (str(index), TorchType(output.type()))
                for index, output in enumerate(torch_graph.outputs())
            ]
        ),
    )
    node_to_op: Dict[TorchNode, Op] = {}
    output_to_port: Dict[Tuple[TorchNode, int], OutputPort] = {}

    def add_param(input_value: TorchValue, param: Parameter):
        op = create_op(
            type=param_node.kind(),
            attrs={
                attr_name: from_ir_attr(node, attr_name)
                for attr_name in param_node.attributeNames()
            },
            inputs=[],
            outputs=OrderedDict([("0", TorchType(input_value.type()))]),
        )
        op.attrs["value"] = param
        graph.add_op(op)
        output_to_port[(param_node, input_value.offset())] = op.output_port(0)

    def add_op(node: TorchNode):
        if (node in node_to_op) or (node == param_node):
            return
        for input in node.inputs():
            add_op(input.node())
        for block in node.blocks():
            for inner_node in block.nodes():
                add_op(inner_node)
        op = create_op(
            type=node.kind(),
            attrs={
                attr_name: from_ir_attr(node, attr_name)
                for attr_name in node.attributeNames()
            },
            inputs=OrderedDict(
                [
                    (str(index), TorchType(input.type()))
                    for index, input in enumerate(node.inputs())
                ]
            ),
            outputs=OrderedDict(
                [
                    (str(index), TorchType(output.type()))
                    for index, output in enumerate(node.outputs())
                ]
            ),
        )
        op.attrs["scope"] = node.scopeName()
        op.attrs["sourceRange"] = node.sourceRange()
        graph.add_op(op)
        node_to_op[node] = op
        for index, input in enumerate(node.inputs()):
            graph.create_edge(
                output_to_port[(input.node(), input.offset())], op.input_port(index)
            )
        for index, port in enumerate(op.output_ports):
            output_to_port[(node, index)] = port

    for index, input in enumerate(list(torch_graph.inputs())[: -len(params)]):
        output_to_port[(input.node(), input.offset())] = cast(
            OutputPort, graph.input_port(index)
        )
    param_node = list(torch_graph.inputs())[0].node()
    for input_value, param in zip(list(param_node.outputs())[-len(params) :], params):
        add_param(input_value, param)
    for node in torch_graph.nodes():
        add_op(node)
    for index, output in enumerate(torch_graph.outputs()):
        graph.create_edge(
            output_to_port[(output.node(), output.offset())],
            cast(InputPort, graph.output_port(index)),
        )
    return graph


def export_to_graph(graph: SubGraph) -> TorchGraph:
    if not graph.namespace.belong_to(pytorch_namespace()):
        raise MismatchNamespaceError(expect=pytorch_namespace(), actual=graph.namespace)
    torch_graph = TorchGraph()
    port_to_output: Dict[OutputPort, TorchValue] = {}
    for port in graph.input_ports:
        port_to_output[port] = torch_graph.addInput()
    for op in graph.sorted_ops:
        if op.type == "prim::Param":
            port_to_output[op.output_port(0)] = torch_graph.addInput()
        else:
            node: TorchNode = torch_graph.create(
                op.type,
                [port_to_output[port.in_edges[0].src] for port in op.input_ports],
                op.output_num,
            )
            attrs = without_internal_attrs(op.attrs)
            for attr_name, attr_value in attrs.items():
                set_ir_attr(node, attr_name, attr_value, op)
            for output_port, output_value in zip(op.output_ports, node.outputs()):
                output_value.setType(output_port.type.raw)
                port_to_output[output_port] = output_value
            torch_graph.appendNode(node)
    for port in graph.output_ports:
        torch_graph.registerOutput(port_to_output[port.in_edges[0].src])
    return torch_graph


def export_to_module(graph: SubGraph, file: Union[str, Path] = None) -> torch.nn.Module:
    torch_graph = export_to_graph(graph)
    params = [op.attrs["value"] for op in graph.sorted_ops if op.type == "prim::Param"]
    forward_func = torch._C._create_function_from_graph("forward", torch_graph)
    module = torch.nn.Module()
    module.forward = types.MethodType(
        lambda self, *args: forward_func(*args, *params), module
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
        if not (is_qualified(name) or name in ["scope", "sourceRange"])
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
            raise RuntimeError("cannot infer attr kind for empty list")
        return infer_attr_kind(attr_value[0]) + "s"
    else:
        raise RuntimeError(f"cannot infer attr kind for {attr_value}")


def set_ir_attr(node: TorchNode, attr_name: str, attr_value: Any, op: Op) -> None:
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
