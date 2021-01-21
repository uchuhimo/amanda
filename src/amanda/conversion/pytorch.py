import gc
import types
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, cast

import torch
import torch._C
import torch.jit
from torch._C import (
    AnyType,
    BoolType,
    ClassType,
    DeviceObjType,
    DictType,
    FloatType,
    FutureType,
)
from torch._C import Graph as TorchGraph
from torch._C import InterfaceType, IntType, ListType
from torch._C import Node as TorchNode
from torch._C import (
    NoneType,
    NumberType,
    OptionalType,
    PyObjectType,
    RRefType,
    StringType,
    TensorType,
    TupleType,
    Type,
)
from torch._C import Value as TorchValue
from torch.nn import Parameter

from amanda import Adapter, EventContext, get_adapter_registry
from amanda.conversion.utils import without_internal_attrs
from amanda.event import (
    after_subgraph_executed,
    before_subgraph_executed,
    on_graph_loaded,
)
from amanda.exception import MismatchNamespaceError
from amanda.graph import InputPort, Op, OutputPort, SubGraph, create_op, create_subgraph
from amanda.io.serde import (
    Serde,
    SerdeContext,
    SerdeDispatcher,
    TypeSerde,
    deserialize,
    deserialize_type,
    get_serde_registry,
    serialize,
    serialize_type,
)
from amanda.lang import replace_all_refs
from amanda.namespace import Namespace, default_namespace
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


def torch_dtype(name: str) -> DataType:
    return DataType(pytorch_type_namespace(), name)


_name_to_torch_type = {
    "AnyType": AnyType.get(),
    "NumberType": NumberType.get(),
    "IntType": IntType.get(),
    "FloatType": FloatType.get(),
    "TensorType": TensorType.get(),
    "BoolType": BoolType.get(),
    "StringType": StringType.get(),
    "DeviceObjType": DeviceObjType.get(),
    "PyObjectType": PyObjectType.get(),
    "NoneType": NoneType.get(),
}

_name_to_dtype = {name: torch_dtype(name) for name in _name_to_torch_type}


class PyTorchPrimitiveSerde(TypeSerde):
    def serialize_type(self, type: Any) -> DataType:
        return _name_to_dtype[type.kind()]

    def deserialize_type(self, dtype: DataType) -> Any:
        return _name_to_torch_type[dtype.name]


_pytorch_primitive_serde = PyTorchPrimitiveSerde()


class PyTorchTupleSerde(TypeSerde):
    def serialize_type(self, type: Any) -> DataType:
        dtype = torch_dtype("TupleType")
        dtype.attrs["elements"] = [
            _serde_dispatcher.serialize_type(element) for element in type.elements()
        ]
        return dtype

    def deserialize_type(self, dtype: DataType) -> Any:
        return TupleType(
            [
                _serde_dispatcher.deserialize_type(element)
                for element in dtype.attrs["elements"]
            ]
        )


class PyTorchDictSerde(TypeSerde):
    def serialize_type(self, type: Any) -> DataType:
        dtype = torch_dtype("DictType")
        dtype.attrs["key"] = _serde_dispatcher.serialize_type(type.getKeyType())
        dtype.attrs["value"] = _serde_dispatcher.serialize_type(type.getValueType())
        return dtype

    def deserialize_type(self, dtype: DataType) -> Any:
        return DictType(
            _serde_dispatcher.deserialize_type(dtype.attrs["key"]),
            _serde_dispatcher.deserialize_type(dtype.attrs["value"]),
        )


@dataclass
class PyTorchContainerSerde(TypeSerde):
    torch_type: Type

    def serialize_type(self, type: Any) -> DataType:
        dtype = torch_dtype(self.torch_type.__name__)
        dtype.attrs["element"] = _serde_dispatcher.serialize_type(type.getElementType())
        return dtype

    def deserialize_type(self, dtype: DataType) -> Any:
        return self.torch_type(
            _serde_dispatcher.deserialize_type(dtype.attrs["element"])
        )


class PyTorchClassSerde(TypeSerde):
    def serialize_type(self, type: Any) -> DataType:
        dtype = torch_dtype("ClassType")
        dtype.attrs["name"] = type.name()
        return dtype

    def deserialize_type(self, dtype: DataType) -> Any:
        return ClassType(dtype.attrs["name"])


class PyTorchInterfaceSerde(TypeSerde):
    def serialize_type(self, type: Any) -> DataType:
        dtype = torch_dtype("InterfaceType")
        dtype.attrs["name"] = type.name()
        return dtype

    def deserialize_type(self, dtype: DataType) -> Any:
        return InterfaceType(dtype.attrs["name"])


class PyTorchTensorSerde(Serde):
    def serialize_type(self, type: Any) -> DataType:
        return torch_dtype("Tensor")

    def deserialize_type(self, dtype: DataType) -> Any:
        return torch.Tensor

    def serialize(self, value: Any) -> Any:
        return serialize(value.numpy())

    def deserialize(self, value: Any, context: SerdeContext) -> Any:
        assert context.dtype == torch_dtype("Tensor")
        context.dtype = DataType(Namespace("np"), "ndarray")
        tensor = torch.tensor(deserialize(value, context))
        context.dtype = torch_dtype("Tensor")
        return tensor


class PyTorchParameterSerde(Serde):
    def serialize_type(self, type: Any) -> DataType:
        return torch_dtype("Parameter")

    def deserialize_type(self, dtype: DataType) -> Any:
        return Parameter

    def serialize(self, value: Any) -> Any:
        return {
            "data": serialize(value.data),
            "requires_grad": value.requires_grad,
        }

    def deserialize(self, value: Any, context: SerdeContext) -> Any:
        assert context.dtype == torch_dtype("Parameter")
        context.dtype = torch_dtype("Tensor")
        parameter = Parameter(
            deserialize(value["data"], context),
            requires_grad=value["requires_grad"],
        )
        context.dtype = torch_dtype("Parameter")
        return parameter


@dataclass
class PyTorchSerdeDispatcher(SerdeDispatcher):
    def __post_init__(self):
        for torch_type in _name_to_torch_type.values():
            self.register_meta_type(type(torch_type), _pytorch_primitive_serde)
        for dtype in _name_to_dtype.values():
            self.register_dtype_name(dtype.name, _pytorch_primitive_serde)
        for torch_type in [
            ListType,
            OptionalType,
            RRefType,
            FutureType,
        ]:
            serde = PyTorchContainerSerde(torch_type)
            self.register_meta_type(torch_type, serde)
            self.register_dtype_name(torch_type.__name__, serde)
        self.register_meta_type(TupleType, PyTorchTupleSerde())
        self.register_dtype_name("TupleType", PyTorchTupleSerde())
        self.register_meta_type(DictType, PyTorchDictSerde())
        self.register_dtype_name("DictType", PyTorchDictSerde())
        self.register_meta_type(ClassType, PyTorchClassSerde())
        self.register_dtype_name("ClassType", PyTorchClassSerde())
        self.register_meta_type(InterfaceType, PyTorchInterfaceSerde())
        self.register_dtype_name("InterfaceType", PyTorchInterfaceSerde())
        self.register_type(torch.Tensor, PyTorchTensorSerde())
        self.register_dtype_name("Tensor", PyTorchTensorSerde())
        self.register_type(Parameter, PyTorchParameterSerde())
        self.register_dtype_name("Parameter", PyTorchParameterSerde())


_serde_dispatcher = PyTorchSerdeDispatcher()

get_serde_registry().register_namespace(pytorch_type_namespace(), _serde_dispatcher)


def import_from_func(func: torch.jit.ScriptFunction) -> SubGraph:
    return import_from_graph(func.graph)


def import_from_module(module: Union[torch.nn.Module, str, Path]) -> SubGraph:
    if isinstance(module, (str, Path)):
        module = torch.jit.load(module)
    if not isinstance(module, torch.jit.ScriptModule):
        module = torch.jit.script(module)
    torch_graph = module.graph
    torch_graph, params = torch._C._jit_pass_lower_graph(torch_graph, module._c)
    graph = import_from_graph(torch_graph, params)
    graph.attrs["training"] = module.training
    return graph


def import_from_graph(
    torch_graph: TorchGraph, params: List[Parameter] = None
) -> SubGraph:
    torch._C._jit_pass_inline(torch_graph)
    torch._C._jit_pass_inline_fork_wait(torch_graph)
    params = params or []
    graph = create_subgraph(
        namespace=pytorch_namespace(),
        inputs=OrderedDict(
            [
                (str(index), serialize_type(input.type()))
                for index, input in enumerate(
                    list(torch_graph.inputs())[: -len(params)]
                )
            ]
        ),
        outputs=OrderedDict(
            [
                (str(index), serialize_type(output.type()))
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
            outputs=OrderedDict([("0", serialize_type(input_value.type()))]),
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
                    (str(index), serialize_type(input.type()))
                    for index, input in enumerate(node.inputs())
                ]
            ),
            outputs=OrderedDict(
                [
                    (str(index), serialize_type(output.type()))
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
            attrs = without_internal_attrs(op.attrs, ["scope", "sourceRange"])
            for attr_name, attr_value in attrs.items():
                set_ir_attr(node, attr_name, attr_value, op)
            for output_port, output_value in zip(op.output_ports, node.outputs()):
                output_value.setType(deserialize_type(output_port.type))
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
    # elif attr_kind == "ty":
    #     return node.ty(attr_name)
    # elif attr_kind == "tys":
    #     return node.tys(attr_name)
    # elif attr_kind == "ival":
    #     return node.ival(attr_name)
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
    # elif attr_kind == "ty":
    #     node.ty_(attr_name, attr_value)
    # elif attr_kind == "tys":
    #     node.tys_(attr_name, attr_value)
    # elif attr_kind == "ival":
    #     node.ival_(attr_name, attr_value)
    else:
        raise RuntimeError(
            f"cannot export {attr_value} to attr {attr_name} in node {node}"
        )


def module_to_op(module: torch.nn.Module, name: str) -> Op:
    return create_op(
        name=name,
        type=type(module).__module__ + "." + type(module).__name__,
        namespace=pytorch_namespace(),
        attrs={
            "raw": module,
            **dict(module.named_children()),
            **dict(module.named_parameters()),
            **dict(module.named_buffers()),
        },
    )


class ModuleAdapter(Adapter):
    def __init__(self):
        super(ModuleAdapter, self).__init__(namespace="pytorch")

    def apply(self, target: torch.nn.Module, context: EventContext) -> None:
        def add_hook(module: torch.nn.Module, name: str):
            def forward_pre_hook(module, input):
                context.trigger(
                    before_subgraph_executed, op=module_to_op(module, name), input=input
                )
                return context["input"]

            def forward_hook(module, input, output):
                context.trigger(
                    after_subgraph_executed,
                    op=module_to_op(module, name),
                    input=input,
                    output=output,
                )
                return context["output"]

            module.register_forward_pre_hook(forward_pre_hook)
            module.register_forward_hook(forward_hook)

        def apply_hook(fn, module: torch.nn.Module, name: str):
            for child_name, module in module.named_children():
                apply_hook(fn, module, f"{name}.{child_name}")
            fn(module, name)

        apply_hook(add_hook, target, "model")


class ScriptModuleAdapter(Adapter):
    def __init__(self):
        super(ScriptModuleAdapter, self).__init__(namespace="pytorch")

    def apply(self, target: torch.jit.ScriptModule, context: EventContext) -> None:
        module = target
        graph = import_from_module(module)
        context.trigger(on_graph_loaded, graph=graph)
        new_graph = context["graph"]
        new_module = export_to_module(new_graph)
        gc.collect()
        replace_all_refs(module, new_module)


get_adapter_registry().register_adapter(torch.jit.ScriptModule, ScriptModuleAdapter())
get_adapter_registry().register_adapter(torch.nn.Module, ModuleAdapter())

import_types = {
    "torchscript": import_from_module,
}

export_types = {
    "torchscript": export_to_module,
}
