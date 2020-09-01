import copy
from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import google.protobuf.text_format
import tensorflow as tf
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

from amanda.conversion.utils import diff_graph_def, to_proto
from amanda.graph import Graph, Op, Tensor
from amanda.namespace import (
    Namespace,
    default_namespace,
    get_global_registry,
    is_qualified,
)
from amanda.rule import OpMapping, Rule, RuleMapper

_namespace = default_namespace() / Namespace("tensorflow")
_internal_namespace = _namespace / Namespace("internal")


def tf_namespace() -> Namespace:
    return _namespace


def tf_internal_namespace() -> Namespace:
    return _internal_namespace


class ToDefaultRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        for key in ["name", "type"]:
            if tf_namespace().qualified(key) in op.attrs:
                op.attrs[key] = op.attrs[tf_namespace().qualified(key)]
        return OpMapping(source_ops=[op], target_ops=[op])


class ToTFRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        for key in ["name", "type"]:
            if default_namespace().qualified(key) in op.attrs:
                op.attrs[key] = op.attrs[default_namespace().qualified(key)]
        return OpMapping(source_ops=[op], target_ops=[op])


_tf_to_default_mapper = RuleMapper(rules=[ToDefaultRule()])
_default_to_tf_mapper = RuleMapper(rules=[ToTFRule()])


def tf_to_default_mapper() -> RuleMapper:
    return _tf_to_default_mapper


def default_to_tf_mapper() -> RuleMapper:
    return _default_to_tf_mapper


get_global_registry().add_mapper(
    tf_namespace(), default_namespace(), tf_to_default_mapper()
)
get_global_registry().add_mapper(
    default_namespace(), tf_namespace(), default_to_tf_mapper()
)


class GraphAttrName:
    meta_graph = tf_internal_namespace().qualified("meta_graph")
    initialized_variables = tf_internal_namespace().qualified("initialized_variables")
    lower_name_func = tf_internal_namespace().qualified("lower_name_func")


class OpAttrName:
    contains_index_in_input_name = tf_internal_namespace().qualified(
        "contains_index_in_input_name"
    )
    experimental_debug_info = tf_internal_namespace().qualified(
        "experimental_debug_info"
    )


def import_from_graph(
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
        node = node_defs[tf_op.name]
        op = Op(
            attrs=attrs,
            input_tensors=[
                from_tf_tensor(input_tensor, graph) for input_tensor in tf_op.inputs
            ],
            control_dependencies=[
                graph.get_op_by_name(control_input_op.name)
                for control_input_op in tf_op.control_inputs
            ],
            output_num=len(tf_op.outputs),
        )
        init_op_attrs(op, node)
        op.name = tf_op.name
        op.type = tf_op.type
        op.attrs["device"] = tf_op.device
        op.attrs[OpAttrName.contains_index_in_input_name] = [
            len(input_name.split(":")) != 1
            for input_name in node.input
            if not input_name.startswith("^")
        ]
        for tf_output_tensor, output_tensor in zip(tf_op.outputs, op.output_tensors):
            output_tensor.attrs["dtype"] = tf_output_tensor.dtype
        graph.add_op(op)

    for tf_op in tf_graph.get_operations():
        add_op(tf_op)
    init_graph_attrs(graph, graph_def)
    meta_graph.graph_def.Clear()
    graph.attrs[GraphAttrName.meta_graph] = meta_graph
    if session is not None:
        update_initialized_variables(graph, tf_graph, session)
    return graph


def init_graph_attrs(graph: Graph, graph_def: tf.GraphDef):
    graph.namespace = tf_namespace()
    for key in ["versions", "library"]:
        if graph_def.HasField(key):
            graph.attrs[key] = getattr(graph_def, key)
    graph.attrs[GraphAttrName.meta_graph] = tf.MetaGraphDef()
    graph.attrs[GraphAttrName.initialized_variables] = {}
    graph.attrs[GraphAttrName.lower_name_func] = lru_cache(maxsize=None)(
        lambda name: name.lower()
    )


def init_op_attrs(op: Op, node: tf.NodeDef):
    op.namespace = tf_namespace()
    if node.HasField("experimental_debug_info"):
        op.attrs[OpAttrName.experimental_debug_info] = node.experimental_debug_info


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
        graph.attrs[GraphAttrName.initialized_variables].update(initialized_variables)


def from_tf_tensor(tensor: Union[tf.Tensor, RefVariable], graph: Graph) -> Tensor:
    op = graph.get_op_by_name(tensor.op.name)
    if op.type == "VariableV2":
        return op.output_tensor(0)
    else:
        return op.output_tensor(tensor.value_index)


@dataclass(frozen=True)
class TFTensor:
    op: str
    output_index: int
    contains_index_in_input_name: bool = True


def import_from_graph_def(
    graph_def: Union[tf.GraphDef, str, bytes, Path], saver_def: SaverDef = None,
) -> Graph:
    graph_def = to_proto(graph_def, tf.GraphDef)
    if saver_def is not None:
        with tf.Graph().as_default() as tf_graph:
            with tf.Session() as session:
                tf.import_graph_def(graph_def, name="")
                return import_from_graph(tf_graph, saver_def, session)
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

            dtypes = get_dtypes(tf_graph, node)
            op = Op(
                attrs=attrs,
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
            init_op_attrs(op, node)
            op.name = node.name
            op.type = node.op
            op.attrs["device"] = node.device
            op.attrs[OpAttrName.contains_index_in_input_name] = [
                input_tensor.contains_index_in_input_name
                for input_tensor in input_tensors
            ]
            for dtype, output_tensor in zip(dtypes, op.output_tensors):
                output_tensor.attrs["dtype"] = dtype
            graph.add_op(op)

        for node in graph_def.node:
            add_op(node)
        init_graph_attrs(graph, graph_def)
        return graph


def get_dtype_proto(node_def, op_def, output_arg):
    def with_number_attr(dtype):
        if len(output_arg.number_attr) != 0:
            for attr in op_def.attr:
                if attr.name == output_arg.number_attr:
                    return [dtype] * node_def.attr[attr.name].i
            raise AssertionError()
        else:
            return dtype

    if len(output_arg.type_attr) != 0:
        for attr in op_def.attr:
            if attr.name == output_arg.type_attr:
                return with_number_attr(node_def.attr[attr.name].type)
        raise AssertionError()
    elif len(output_arg.type_list_attr) != 0:
        for attr in op_def.attr:
            if attr.name == output_arg.type_list_attr:
                return list(node_def.attr[attr.name].list.type)
        raise AssertionError()
    else:
        assert output_arg.type != types_pb2.DT_INVALID
        return with_number_attr(output_arg.type)


def get_dtypes(tf_graph, node_def):
    op_def = tf_graph._get_op_def(node_def.op)
    dtypes = [
        get_dtype_proto(node_def, op_def, output_arg)
        for output_arg in op_def.output_arg
    ]
    if len(dtypes) == 1 and isinstance(dtypes[0], list):
        dtypes = dtypes[0]
    return [tf.as_dtype(dtype) for dtype in dtypes]


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
    meta_graph: Union[tf.MetaGraphDef, str, bytes, Path],
    checkpoint: Union[str, Path] = None,
    session: tf.Session = None,
) -> Graph:
    meta_graph = to_proto(meta_graph, tf.MetaGraphDef)
    with tf.Graph().as_default() as tf_graph, ExitStack() as exit_stack:
        saver = tf.train.import_meta_graph(meta_graph)
        if checkpoint is not None:
            if session is None:
                session = tf.Session()
                exit_stack.enter_context(session)
            saver.restore(session, str(checkpoint))
        graph = import_from_graph(tf_graph, saver.saver_def, session)
        return graph


def import_from_checkpoint(path: Union[str, Path]) -> Graph:
    path = str(path)
    return import_from_meta_graph(path + ".meta", path)


def import_from_saved_model(path: Union[str, Path], tags: List[str]) -> Graph:
    path = str(path)
    with tf.Graph().as_default() as tf_graph:
        with tf.Session() as session:
            meta_graph = tf.saved_model.load(session, tags, path)
            graph = import_from_graph(tf_graph, meta_graph.saver_def, session)
            return graph


@dataclass
class FakeNodeDef:
    op: str
    attr: Dict[str, Any]


def export_to_graph(graph: Graph) -> Tuple[tf.Graph, tf.train.Saver, tf.Session]:
    if graph.namespace != tf_namespace():
        graph = graph.to_default_namespace().to_namespace(tf_namespace())
    tf_graph: tf.Graph
    with tf.Graph().as_default() as tf_graph:
        for op in graph.sorted_ops:
            attrs = without_internal_attrs(op.attrs)
            with tf.control_dependencies(
                [
                    tf_graph.get_operation_by_name(control_input.name)
                    for control_input in op.control_dependencies
                ]
            ):
                attrs_proto = to_attrs_proto(
                    tf_graph._get_op_def(op.type), op.type, attrs
                )
                dtypes = [
                    output_tensor.attrs.get("dtype")
                    for output_tensor in op.output_tensors
                ]
                if None in dtypes:
                    dtypes = get_dtypes(tf_graph, FakeNodeDef(op.type, attrs_proto))
                tf_op = tf_graph.create_op(
                    op_type=op.type,
                    inputs=[
                        tf_graph.get_tensor_by_name(
                            f"{tensor.op.name}:{tensor.output_index}"
                        )
                        for tensor in op.input_tensors
                    ],
                    dtypes=dtypes,
                    input_types=None,
                    name=op.name,
                    attrs=attrs_proto,
                )
            if "device" in op.attrs:
                tf_op._set_device(op.attrs["device"])
        saver = tf.train.import_meta_graph(graph.attrs[GraphAttrName.meta_graph])
        variables = {
            variable.name: variable
            for variable in tf_graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        }
        session = tf.Session()
        initializers = [
            variables[name].initializer
            for name, value in graph.attrs[GraphAttrName.initialized_variables].items()
        ]
        feed_dict = {
            variables[name].initializer.inputs[1]: value
            for name, value in graph.attrs[GraphAttrName.initialized_variables].items()
        }
        session.run(initializers, feed_dict)
        return tf_graph, saver, session


def without_internal_attrs(attrs):
    return {
        name: value
        for name, value in attrs.items()
        if not (is_qualified(name) or name in ["name", "type", "device"])
    }


def export_to_graph_def(graph: Graph) -> tf.GraphDef:
    if graph.namespace != tf_namespace():
        graph = graph.to_default_namespace().to_namespace(tf_namespace())
    tf_graph = tf.Graph()
    graph_def = tf_graph.as_graph_def()
    for key in ["versions", "library"]:
        if key in graph.attrs:
            getattr(graph_def, key).CopyFrom(graph.attrs[key])
    for op in graph.sorted_ops:
        attrs = without_internal_attrs(op.attrs)
        node = graph_def.node.add()
        node.name = op.name
        node.op = op.type
        node.device = op.attrs["device"]
        if OpAttrName.experimental_debug_info in op.attrs:
            node.experimental_debug_info.CopyFrom(
                op.attrs[OpAttrName.experimental_debug_info]
            )
        for name, attr_value in to_attrs_proto(
            tf_graph._get_op_def(op.type), op.type, attrs
        ).items():
            node.attr[name].CopyFrom(attr_value)
        for index, tensor in enumerate(op.input_tensors):
            input_op = tensor.op
            if OpAttrName.contains_index_in_input_name in op.attrs:
                contains_index_in_name = op.attrs[
                    OpAttrName.contains_index_in_input_name
                ][index]
            else:
                contains_index_in_name = True
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
        path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    tf_graph, saver, session = export_to_graph(graph)
    with session, tf_graph.as_default():
        if saver is None:
            saver = tf.train.Saver()
        saver.save(session, str(path))


def export_to_saved_model(
    graph: Graph, path: Union[str, Path], tags: List[str]
) -> None:
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    tf_graph, saver, session = export_to_graph(graph)
    path = str(path)
    builder = tf.saved_model.builder.SavedModelBuilder(path)
    with tf_graph.as_default(), session:
        builder.add_meta_graph_and_variables(
            session, tags, strip_default_attrs=True, saver=saver,
        )
    builder.save()


def get_diff_after_conversion(graph_def: tf.GraphDef) -> Dict[str, Any]:
    graph = import_from_graph_def(graph_def)
    new_graph_def = export_to_graph_def(graph)
    return diff_graph_def(graph_def, new_graph_def)


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


_py_funcs: List[Any] = []


def import_from_tf_func(tf_func):
    def amanda_func(graph: Graph):
        def func(*args, **kwargs):
            if (
                hasattr(tf_func, "_tf_api_names_v1")
                and len(tf_func._tf_api_names_v1) == 1
                and tf_func._tf_api_names_v1[0] == "py_func"
            ):
                # The Global registry for py functions
                # `tensorflow.python.ops.script_ops._py_funcs`
                # is a `WeakValueDictionary`, we keep strong reference to
                # `func` in `tf.py_func` to ensure its lifetime is longer enough when
                # we need to export this graph.
                _py_funcs.append(args[0])
            args_dict = dict(enumerate(args))
            tf_args = list(args)
            tf_kwargs = dict(kwargs)
            input_tensors: List[Tensor] = []
            placeholders: List[tf.Tensor] = []

            def to_tf_tensor(tensor: Tensor) -> tf.Tensor:
                placeholder = tf.placeholder(tensor.attrs["dtype"])
                input_tensors.append(tensor)
                placeholders.append(placeholder)
                return placeholder

            def get_args(name: Union[int, str]) -> Dict[Any, Any]:
                return args_dict if isinstance(name, int) else kwargs

            def get_tf_args(name: Union[int, str]) -> Dict[Any, Any]:
                return tf_args if isinstance(name, int) else tf_kwargs

            with tf.Graph().as_default() as tf_graph:
                # mimic tensorflow.python.framework.ops.Graph.unique_name
                lower_name_func = graph.attrs[GraphAttrName.lower_name_func]
                tf_graph._names_in_use = {
                    lower_name_func(name): 1 for name in graph.names
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
                new_graph = import_from_graph(tf_graph)
                for input, placeholder in zip(input_tensors, placeholders):
                    placeholder_op = new_graph.get_op_by_name(placeholder.op.name)
                    new_graph.replace_tensor(placeholder_op.output_tensor(0), input)
                    new_graph.remove_op(placeholder_op)
                for op in new_graph.ops:
                    graph.add_op(op)

                current_meta_graph = new_graph.attrs[GraphAttrName.meta_graph]
                if len(current_meta_graph.collection_def) != 0:
                    meta_graph = copy.deepcopy(graph.attrs[GraphAttrName.meta_graph])
                    meta_graph.collection_def.MergeFrom(
                        current_meta_graph.collection_def
                    )
                    graph.attrs[GraphAttrName.meta_graph] = meta_graph

                output_op = graph.get_op_by_name(output_tensor.op.name)
                return output_op.output_tensor(output_tensor.value_index)

        return func

    return amanda_func


import_types = {
    "tensorflow_pbtxt": import_from_pbtxt,
    "tensorflow_checkpoint": import_from_checkpoint,
    "tensorflow_saved_model": import_from_saved_model,
}

export_types = {
    "tensorflow_pbtxt": export_to_pbtxt,
    "tensorflow_checkpoint": export_to_checkpoint,
    "tensorflow_saved_model": export_to_saved_model,
}
