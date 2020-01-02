import copy
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import google.protobuf.text_format
import jsondiff
import tensorflow as tf
from google.protobuf import json_format
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework.op_def_pb2 import OpDef
from tensorflow.core.protobuf.saver_pb2 import SaverDef
from tensorflow.python import RefVariable, tensor_shape, types_pb2
from tensorflow.python.framework.op_def_library import (
    _IsListValue,
    _MakeBool,
    _MakeFloat,
    _MakeInt,
    _MakeShape,
    _MakeStr,
    _MakeTensor,
    _MakeType,
)
from tensorflow.python.util import compat

from amanda.graph import Graph, Op, Tensor
from amanda.namespace import Namespace, default_namespace, get_global_registry
from amanda.rule import NoopRule, RuleMapper

_namespace = Namespace(name="tensorflow")


def tf_namespace() -> Namespace:
    return _namespace


_tf_to_default_mapper = RuleMapper(rules=[NoopRule()])


def tf_to_default_mapper() -> RuleMapper:
    return _tf_to_default_mapper


_default_to_tf_mapper = RuleMapper(rules=[NoopRule()])


def default_to_tf_mapper() -> RuleMapper:
    return _default_to_tf_mapper


get_global_registry().add_mapper(
    tf_namespace(), default_namespace(), tf_to_default_mapper()
)
get_global_registry().add_mapper(
    default_namespace(), tf_namespace(), default_to_tf_mapper()
)


class GraphKey:
    meta_graph = "meta_graph"
    initialized_variables = "initialized_variables"
    lower_name_func = "lower_name_func"


def import_from_tf_graph(
    tf_graph: tf.Graph, saver_def: SaverDef = None, session: tf.Session = None,
) -> Graph:
    graph = Graph()
    meta_graph: tf.MetaGraphDef = tf.train.export_meta_graph(
        graph=tf_graph, saver_def=saver_def
    )
    graph_def = meta_graph.graph_def
    node_defs = {op_def.name: op_def for op_def in graph_def.node}

    def add_op(tf_op: tf.Operation):
        if graph.get_op_by_name(tf_op.name) is not None:
            return
        input_tensor: tf.Tensor
        for input_tensor in tf_op.inputs:
            add_op(input_tensor.op)
        control_input_op: tf.Operation
        for control_input_op in tf_op.control_inputs:
            add_op(control_input_op)
        attrs = {
            attr_name: tf_op.get_attr(attr_name) for attr_name in tf_op.node_def.attr
        }
        dtypes = [output_tensor.dtype for output_tensor in tf_op.outputs]
        node = node_defs[tf_op.name]
        op = Op(
            attrs=dict(
                name=tf_op.name,
                type=tf_op.type,
                device=tf_op.device,
                __dtypes=dtypes,
                __contains_index_in_input_name=[
                    len(input_name.split(":")) != 1
                    for input_name in node.input
                    if not input_name.startswith("^")
                ],
                **attrs,
            ),
            input_tensors=[
                from_tf_tensor(input_tensor, graph) for input_tensor in tf_op.inputs
            ],
            control_dependencies=[
                graph.get_op_by_name(control_input_op.name)
                for control_input_op in tf_op.control_inputs
            ],
            output_num=len(dtypes),
        )
        if node.HasField("experimental_debug_info"):
            op.attrs["experimental_debug_info"] = node.experimental_debug_info
        graph.add_op(op)

    for tf_op in tf_graph.get_operations():
        add_op(tf_op)
    init_graph_attrs(graph)
    meta_graph.graph_def.Clear()
    graph.attrs[GraphKey.meta_graph] = meta_graph
    if session is not None:
        update_initialized_variables(graph, tf_graph, session)
    return graph


def init_graph_attrs(graph: Graph):
    graph.namespace = tf_namespace()
    graph.attrs[GraphKey.meta_graph] = tf.MetaGraphDef()
    graph.attrs[GraphKey.initialized_variables] = {}
    graph.attrs[GraphKey.lower_name_func] = lru_cache(maxsize=None)(
        lambda name: name.lower()
    )


def update_initialized_variables(graph: Graph, tf_graph: tf.Graph, session: tf.Session):
    with tf_graph.as_default():
        all_variables = tf.global_variables()
        uninitialized_variables = session.run(
            tf.report_uninitialized_variables(tf.global_variables())
        )
        if len(all_variables) == len(uninitialized_variables):
            return
        initialized_variables = set(
            map(lambda variable: variable.op.name, all_variables)
        ) - set(map(lambda x: x.decode(), uninitialized_variables))
        initialized_variables = session.run(
            {
                variable.name: variable.read_value()
                for variable in all_variables
                if variable.op.name in initialized_variables
            }
        )
        graph.attrs[GraphKey.initialized_variables].update(initialized_variables)


def from_tf_tensor(tensor: Union[tf.Tensor, RefVariable], graph: Graph) -> Tensor:
    op = graph.get_op_by_name(tensor.op.name)
    if op.type == "VariableV2":
        return op.output_tensor(0)
    else:
        return op.output_tensor(tensor.value_index)


@dataclass
class TFTensor:
    op: str
    output_index: int
    contains_index_in_input_name: bool = True


def import_from_graph_def(
    graph_def: Union[tf.GraphDef, str, Path], saver_def: SaverDef = None,
) -> Graph:
    if not isinstance(graph_def, tf.GraphDef):
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(Path(graph_def).read_bytes())
    if saver_def is not None:
        with tf.Graph().as_default() as tf_graph:
            with tf.Session() as session:
                tf.import_graph_def(graph_def, name="")
                return import_from_tf_graph(tf_graph, saver_def, session)
    else:
        graph = Graph()
        tf_graph = tf.Graph()
        name_to_node = {node.name: node for node in graph_def.node}

        def add_op(node: tf.NodeDef):
            if graph.get_op_by_name(node.name) is not None:
                return
            input_tensors: List[TFTensor] = []
            control_input_nodes: List[str] = []
            input: str
            for input in node.input:
                if input.startswith("^"):
                    control_input_nodes.append(input[1:])
                else:
                    names = input.split(":")
                    assert len(names) == 1 or len(names) == 2
                    if len(names) == 1:
                        input_tensors.append(TFTensor(names[0], 0, False))
                    else:
                        input_tensors.append(TFTensor(names[0], int(names[1])))
            for input_tensor in input_tensors:
                add_op(name_to_node[input_tensor.op])
            for control_input_node in control_input_nodes:
                add_op(name_to_node[control_input_node])

            attrs = {
                attr_name: from_attr_proto(node.attr[attr_name])
                for attr_name in node.attr
            }

            op_def = tf_graph._get_op_def(node.op)

            def get_dtype_proto(output_arg):
                if len(output_arg.type_attr) != 0:
                    for attr in op_def.attr:
                        if attr.name == output_arg.type_attr:
                            return node.attr[attr.name].type
                    raise AssertionError(
                        f"cannot find attribute {output_arg.type_attr} "
                        f"in op {node.op} {node.name}"
                    )
                elif len(output_arg.type_list_attr) != 0:
                    for attr in op_def.attr:
                        if attr.name == output_arg.type_list_attr:
                            return list(node.attr[attr.name].list.type)
                    raise AssertionError(
                        f"cannot find attribute {output_arg.type_list_attr} "
                        f"in op {node.op} {node.name}"
                    )
                else:
                    if output_arg.type == types_pb2.DT_INVALID:
                        raise AssertionError(
                            f"No type fields in op {node.op} {node.name}, "
                            f"op_def: {op_def}"
                        )
                    return output_arg.type

            dtypes = [get_dtype_proto(output_arg) for output_arg in op_def.output_arg]
            if len(dtypes) == 1 and isinstance(dtypes[0], list):
                dtypes = dtypes[0]
            dtypes = [tf.as_dtype(dtype) for dtype in dtypes]
            op = Op(
                attrs=dict(
                    name=node.name,
                    type=node.op,
                    device=node.device,
                    __dtypes=dtypes,
                    __contains_index_in_input_name=[
                        input_tensor.contains_index_in_input_name
                        for input_tensor in input_tensors
                    ],
                    **attrs,
                ),
                input_tensors=[
                    graph.get_op_by_name(input_tensor.op).output_tensor(
                        input_tensor.output_index
                    )
                    for input_tensor in input_tensors
                ],
                control_dependencies=[
                    graph.get_op_by_name(control_input_name)
                    for control_input_name in control_input_nodes
                ],
                output_num=len(dtypes),
            )
            if node.HasField("experimental_debug_info"):
                op.attrs["experimental_debug_info"] = node.experimental_debug_info
            graph.add_op(op)

        for node in graph_def.node:
            add_op(node)
        init_graph_attrs(graph)
        return graph


def from_attr_proto(attr_value: tf.AttrValue) -> Any:
    field_name = attr_value.WhichOneof("value")
    if field_name == "s":
        return attr_value.s
    elif field_name == "i":
        return attr_value.i
    elif field_name == "f":
        return attr_value.f
    elif field_name == "b":
        return attr_value.b
    elif field_name == "type":
        return tf.as_dtype(attr_value.type)
    elif field_name == "shape":
        return tensor_shape.as_shape(attr_value.shape)
    elif field_name == "tensor":
        return attr_value.tensor
    elif field_name == "func":
        return attr_value.func
    elif field_name == "placeholder":
        return attr_value.placeholder
    elif field_name == "list":
        list_value = attr_value.list
        if len(list_value.s) != 0:
            return [value for value in list_value.s]
        elif len(list_value.i) != 0:
            return [value for value in list_value.i]
        elif len(list_value.f) != 0:
            return [value for value in list_value.f]
        elif len(list_value.b) != 0:
            return [value for value in list_value.b]
        elif len(list_value.type) != 0:
            return [tf.as_dtype(value) for value in list_value.type]
        elif len(list_value.shape) != 0:
            return [tensor_shape.as_shape(value) for value in list_value.shape]
        elif len(list_value.tensor) != 0:
            return [value for value in list_value.tensor]
        elif len(list_value.func) != 0:
            return [value for value in list_value.func]
        else:
            return []


def import_from_pbtxt(file: Union[str, Path]) -> Graph:
    file = Path(file)
    graph_def = tf.GraphDef()
    google.protobuf.text_format.Parse(file.read_text(), graph_def)
    return import_from_graph_def(graph_def)


def import_from_meta_graph(
    meta_graph: Union[tf.MetaGraphDef, str, Path],
    checkpoint: Union[str, Path] = None,
    session: tf.Session = None,
) -> Graph:
    with tf.Graph().as_default() as tf_graph:
        saver = tf.train.import_meta_graph(meta_graph)
        with ExitStack() as exit_stack:
            if checkpoint is not None:
                if session is None:
                    session = tf.Session()
                    exit_stack.enter_context(session)
                saver.restore(session, str(checkpoint))
            graph = import_from_tf_graph(tf_graph, saver.saver_def, session)
            return graph


def import_from_checkpoint(path: Union[str, Path]) -> Graph:
    path = str(path)
    return import_from_meta_graph(path + ".meta", path)


def export_to_tf_graph(graph: Graph) -> Tuple[tf.Graph, tf.train.Saver, tf.Session]:
    if graph.namespace != tf_namespace():
        graph.to_default_namespace().to_namespace(tf_namespace())
    tf_graph: tf.Graph
    with tf.Graph().as_default() as tf_graph:
        for op in graph.post_order_ops:
            attrs = dict(op.attrs)
            remove_internal_attrs(attrs)
            with tf.control_dependencies(
                [
                    tf_graph.get_operation_by_name(control_input.name)
                    for control_input in op.control_dependencies
                ]
            ):
                tf_op = tf_graph.create_op(
                    op_type=op.type,
                    inputs=[
                        tf_graph.get_tensor_by_name(
                            f"{tensor.op.name}:{tensor.output_index}"
                        )
                        for tensor in op.input_tensors
                    ],
                    dtypes=get_dtypes(op),
                    input_types=None,
                    name=op.name,
                    attrs=to_attrs_proto(tf_graph._get_op_def(op.type), op.type, attrs),
                )
            if "device" in op.attrs:
                tf_op._set_device(op.attrs["device"])
        saver = tf.train.import_meta_graph(graph.attrs[GraphKey.meta_graph])
        variables = {
            variable.name: variable
            for variable in tf_graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        }
        session = tf.Session()
        initializers = [
            variables[name].initializer
            for name, value in graph.attrs[GraphKey.initialized_variables].items()
        ]
        feed_dict = {
            variables[name].initializer.inputs[1]: value
            for name, value in graph.attrs[GraphKey.initialized_variables].items()
        }
        session.run(initializers, feed_dict)
        return tf_graph, saver, session


def remove_internal_attrs(attrs):
    attrs.pop("name", None)
    attrs.pop("type", None)
    attrs.pop("device", None)
    attrs.pop("experimental_debug_info", None)
    attrs.pop("__dtypes", None)
    attrs.pop("__contains_index_in_input_name", None)


def export_to_graph_def(graph: Graph) -> tf.GraphDef:
    tf_graph = tf.Graph()
    graph_def = tf_graph.as_graph_def()
    for op in graph.post_order_ops:
        attrs = dict(op.attrs)
        remove_internal_attrs(attrs)
        node = graph_def.node.add()
        node.name = op.name
        node.op = op.type
        node.device = op.attrs["device"]
        if "experimental_debug_info" in op.attrs:
            node.experimental_debug_info.CopyFrom(op.attrs["experimental_debug_info"])
        for name, attr_value in to_attrs_proto(
            tf_graph._get_op_def(op.type), op.type, attrs
        ).items():
            node.attr[name].CopyFrom(attr_value)
        for index, tensor in enumerate(op.input_tensors):
            input_op = tensor.op
            contains_index_in_name = op.attrs["__contains_index_in_input_name"][index]
            if contains_index_in_name:
                node.input.append(f"{input_op.name}:{tensor.output_index}")
            else:
                node.input.append(input_op.name)
        for control_input in op.control_dependencies:
            node.input.append(f"^{control_input.name}")
    return graph_def


def export_to_pbtxt(graph: Graph, file: Union[str, Path] = None) -> str:
    graph_def = export_to_graph_def(graph)
    graph_pbtxt = google.protobuf.text_format.MessageToString(graph_def)
    if file is not None:
        file = Path(file)
        file.write_text(graph_pbtxt)
    return graph_pbtxt


def export_to_checkpoint(graph: Graph, path: Union[str, Path]) -> None:
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(mode=0o755, parents=True)
    tf_graph, saver, session = export_to_tf_graph(graph)
    with session:
        with tf_graph.as_default():
            if saver is None:
                saver = tf.train.Saver()
            saver.save(session, str(path))


def get_diff_after_conversion(graph_def: tf.GraphDef) -> Dict[str, Any]:
    graph = import_from_graph_def(graph_def)
    new_graph_def = export_to_graph_def(graph)
    return diff_graph_def(graph_def, new_graph_def)


def graph_def_to_dict(graph_def: tf.GraphDef) -> Dict[str, Any]:
    return {
        node.name: json_format.MessageToDict(node, preserving_proto_field_name=True)
        for node in graph_def.node
    }


@contextmanager
def recursionlimit(limit: int):
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(limit)
        yield None
    finally:
        sys.setrecursionlimit(old_limit)


def diff_graph_def(graph_def1: tf.GraphDef, graph_def2: tf.GraphDef) -> Dict[str, Any]:
    with recursionlimit(10000):
        return jsondiff.diff(
            graph_def_to_dict(graph_def1),
            graph_def_to_dict(graph_def2),
            syntax="explicit",
        )


def to_attrs_proto(
    op_def: OpDef, op_type_name: str, attrs: Dict[str, Any],
) -> Dict[str, Any]:
    # Convert attr values to AttrValue protos.
    attr_protos = {}
    attr_defs = {attr_def.name: attr_def for attr_def in op_def.attr}
    for key, value in attrs.items():
        attr_value = tf.AttrValue()
        if key in attr_defs:
            attr_def = attr_defs[key]
        elif value is None:
            attr_protos[key] = attr_value
            continue
        else:
            attr_def = OpDef.AttrDef()
            if isinstance(value, (str, bytes)):
                attr_def.type = "string"
            elif isinstance(value, float):
                attr_def.type = "float"
            elif isinstance(value, bool):
                attr_def.type = "bool"
            # bool is a subclass of int, so we should check bool before checking int
            elif isinstance(value, int):
                attr_def.type = "int"
            elif isinstance(value, tf.DType):
                attr_def.type = "type"
            elif isinstance(value, tf.TensorShape):
                attr_def.type = "shape"
            elif isinstance(value, tensor_pb2.TensorProto):
                attr_def.type = "tensor"
            elif isinstance(value, tf.NameAttrList):
                attr_def.type = "func"
            elif isinstance(value, list) and len(value) == 0:
                attr_value.list.SetInParent()
                attr_protos[key] = attr_value
                continue
            elif isinstance(value, list) and isinstance(value[0], (str, bytes)):
                attr_def.type = "list(string)"
            elif isinstance(value, list) and isinstance(value[0], bool):
                attr_def.type = "list(bool)"
            # bool is a subclass of int, so we should check bool before checking int
            elif isinstance(value, list) and isinstance(value[0], int):
                attr_def.type = "list(int)"
            elif isinstance(value, list) and isinstance(value[0], float):
                attr_def.type = "list(float)"
            elif isinstance(value, list) and isinstance(value[0], tf.DType):
                attr_def.type = "list(type)"
            elif isinstance(value, list) and isinstance(value[0], tf.TensorShape):
                attr_def.type = "list(shape)"
            elif isinstance(value, list) and isinstance(
                value[0], tensor_pb2.TensorProto
            ):
                attr_def.type = "list(tensor)"
            else:
                raise AssertionError(f"{value} has unsupported type")
        if attr_def.HasField("default_value") and value is None:
            attr_value.CopyFrom(attr_def.default_value)
            attr_protos[key] = attr_value
            continue
        if attr_def.type.startswith("list("):
            if not _IsListValue(value):
                raise TypeError("Expected list for attr " + key)
            if attr_def.has_minimum:
                if len(value) < attr_def.minimum:
                    raise ValueError(
                        "Attr '%s' of '%s' Op passed list of length %d "
                        "less than minimum %d."
                        % (key, op_type_name, len(value), attr_def.minimum)
                    )
            attr_value.list.SetInParent()
        if attr_def.type == "string":
            attr_value.s = _MakeStr(value, key)
            if attr_def.HasField("allowed_values"):
                if attr_value.s not in attr_def.allowed_values.list.s:
                    raise ValueError(
                        "Attr '%s' of '%s' Op passed string '%s' not in: \"%s\"."
                        % (
                            key,
                            op_type_name,
                            compat.as_text(attr_value.s),
                            '", "'.join(
                                map(compat.as_text, attr_def.allowed_values.list.s)
                            ),
                        )
                    )
        elif attr_def.type == "list(string)":
            attr_value.list.s.extend([_MakeStr(x, key) for x in value])
            if attr_def.HasField("allowed_values"):
                for x in attr_value.list.s:
                    if x not in attr_def.allowed_values.list.s:
                        raise ValueError(
                            "Attr '%s' of '%s' Op passed string '%s' not in: \"%s\"."
                            % (
                                key,
                                op_type_name,
                                compat.as_text(x),
                                '", "'.join(
                                    map(compat.as_text, attr_def.allowed_values.list.s)
                                ),
                            )
                        )
        elif attr_def.type == "int":
            attr_value.i = _MakeInt(value, key)
            if attr_def.has_minimum:
                if attr_value.i < attr_def.minimum:
                    raise ValueError(
                        "Attr '%s' of '%s' Op passed %d less than minimum %d."
                        % (key, op_type_name, attr_value.i, attr_def.minimum)
                    )
        elif attr_def.type == "list(int)":
            attr_value.list.i.extend([_MakeInt(x, key) for x in value])
        elif attr_def.type == "float":
            attr_value.f = _MakeFloat(value, key)
        elif attr_def.type == "list(float)":
            attr_value.list.f.extend([_MakeFloat(x, key) for x in value])
        elif attr_def.type == "bool":
            attr_value.b = _MakeBool(value, key)
        elif attr_def.type == "list(bool)":
            attr_value.list.b.extend([_MakeBool(x, key) for x in value])
        elif attr_def.type == "type":
            attr_value.type = _MakeType(value, attr_def)
        elif attr_def.type == "list(type)":
            attr_value.list.type.extend([_MakeType(x, attr_def) for x in value])
        elif attr_def.type == "shape":
            attr_value.shape.CopyFrom(_MakeShape(value, key))
        elif attr_def.type == "list(shape)":
            attr_value.list.shape.extend([_MakeShape(x, key) for x in value])
        elif attr_def.type == "tensor":
            attr_value.tensor.CopyFrom(_MakeTensor(value, key))
        elif attr_def.type == "list(tensor)":
            attr_value.list.tensor.extend([_MakeTensor(x, key) for x in value])
        elif attr_def.type == "func":
            if isinstance(value, tf.NameAttrList):
                attr_value.func.CopyFrom(value)
            elif isinstance(value, compat.bytes_or_text_types):
                attr_value.func.name = value
            else:
                value.add_to_graph(tf.get_default_graph())
                attr_value.func.name = value.name
        else:
            raise TypeError("Unrecognized Attr type " + attr_def.type)

        attr_protos[key] = attr_value
    return attr_protos


def get_dtypes(op: Op):
    return op.attrs["__dtypes"]


def get_dtype(tensor: Tensor):
    return get_dtypes(tensor.op)[tensor.output_index]


def import_from_tf_func(tf_func):
    def amanda_func(graph: Graph):
        def func(*args, **kwargs):
            args_dict = dict(enumerate(args))
            tf_args = list(args)
            tf_kwargs = dict(kwargs)
            input_tensors: List[Tensor] = []
            placeholders: List[tf.Tensor] = []

            def to_tf_tensor(tensor: Tensor) -> tf.Tensor:
                placeholder = tf.placeholder(get_dtype(tensor))
                input_tensors.append(tensor)
                placeholders.append(placeholder)
                return placeholder

            def get_args(name: Union[int, str]) -> Dict[Any, Any]:
                return args_dict if isinstance(name, int) else kwargs

            def get_tf_args(name: Union[int, str]) -> Dict[Any, Any]:
                return tf_args if isinstance(name, int) else tf_kwargs

            with tf.Graph().as_default() as tf_graph:
                # mimic tensorflow.python.framework.ops.Graph.unique_name
                tf_graph._names_in_use = {
                    graph.attrs[GraphKey.lower_name_func](name): 1
                    for name in graph.names
                }

                all_args = {**args_dict, **kwargs}
                for name in all_args:
                    arg = get_args(name)[name]
                    if isinstance(arg, Tensor):
                        get_tf_args(name)[name] = to_tf_tensor(arg)
                    if isinstance(arg, list):
                        arg_list = arg
                        for i, arg in enumerate(arg_list):
                            if isinstance(arg, Tensor):
                                get_tf_args(name)[name][i] = to_tf_tensor(arg)
                output_tensor: tf.Tensor = tf_func(*tf_args, **tf_kwargs)
                new_graph = import_from_tf_graph(tf_graph)
                for input, placeholder in zip(input_tensors, placeholders):
                    placeholder_op = new_graph.get_op_by_name(placeholder.op.name)
                    new_graph.replace_tensor(placeholder_op.output_tensor(0), input)
                    new_graph.remove_op(placeholder_op)
                for op in new_graph.ops:
                    graph.add_op(op)

                current_meta_graph = new_graph.attrs[GraphKey.meta_graph]
                if len(current_meta_graph.collection_def) != 0:
                    meta_graph = copy.deepcopy(graph.attrs[GraphKey.meta_graph])
                    meta_graph.collection_def.MergeFrom(
                        current_meta_graph.collection_def
                    )
                    graph.attrs[GraphKey.meta_graph] = meta_graph

                output_op = graph.get_op_by_name(output_tensor.op.name)
                return output_op.output_tensor(output_tensor.value_index)

        return func

    return amanda_func
