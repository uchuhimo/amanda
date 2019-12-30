import copy
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import tensorflow as tf
from tensorflow.core.framework.op_def_pb2 import OpDef
from tensorflow.core.protobuf.saver_pb2 import SaverDef
from tensorflow.python import RefVariable
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

from mmx import Graph, Op, Tensor


class GraphKey:
    meta_graph = "meta_graph"
    initialized_variables = "initialized_variables"
    lower_name_func = "lower_name_func"


def import_from_tf_graph(
    tf_graph: tf.Graph, saver_def: SaverDef = None, session: tf.Session = None,
) -> Graph:
    graph = Graph()

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
        op = Op(
            attrs=dict(
                name=tf_op.name,
                type=tf_op.type,
                device=tf_op.device,
                __dtypes=[output_tensor.dtype for output_tensor in tf_op.outputs],
                **attrs,
            ),
            input_tensors=[
                from_tf_tensor(input_tensor, graph) for input_tensor in tf_op.inputs
            ],
            control_dependencies=[
                graph.get_op_by_name(control_input_op.name)
                for control_input_op in tf_op.control_inputs
            ],
        )
        graph.add_op(op)

    for tf_op in tf_graph.get_operations():
        add_op(tf_op)
    meta_graph: tf.MetaGraphDef = tf.train.export_meta_graph(
        graph=tf_graph, saver_def=saver_def
    )
    meta_graph.graph_def.Clear()
    graph.attrs[GraphKey.meta_graph] = meta_graph
    graph.attrs[GraphKey.initialized_variables] = {}
    if session is not None:
        update_initialized_variables(graph, tf_graph, session)
    graph.attrs[GraphKey.lower_name_func] = lru_cache(maxsize=None)(
        lambda name: name.lower()
    )
    return graph


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


def import_from_graph_def(
    graph_def: Union[tf.GraphDef, str, Path],
    saver_def: SaverDef = None,
    session: tf.Session = None,
) -> Graph:
    if not isinstance(graph_def, tf.GraphDef):
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(Path(graph_def).read_bytes())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
        return import_from_tf_graph(graph, saver_def, session)


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


def export_to_tf_graph(graph: Graph,) -> Tuple[tf.Graph, tf.train.Saver, tf.Session]:
    tf_graph: tf.Graph
    with tf.Graph().as_default() as tf_graph:
        for op in graph.post_order_ops:
            attrs = dict(op.attrs)
            del attrs["name"]
            del attrs["type"]
            del attrs["device"]
            del attrs["__dtypes"]
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


def to_attrs_proto(
    op_def: OpDef, op_type_name: str, attrs: Dict[str, Any],
) -> Dict[str, Any]:
    # Convert attr values to AttrValue protos.
    attr_protos = {}
    for attr_def in op_def.attr:
        key = attr_def.name
        value = attrs[key]
        attr_value = tf.AttrValue()
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


def convert_from_tf_func(tf_func, graph: Graph):
    def new_func(*args, **kwargs):
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
                graph.attrs[GraphKey.lower_name_func](name): 1 for name in graph.names
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
                meta_graph.collection_def.MergeFrom(current_meta_graph.collection_def)
                graph.attrs[GraphKey.meta_graph] = meta_graph

            output_op = graph.get_op_by_name(output_tensor.op.name)
            return output_op.output_tensor(output_tensor.value_index)

    return new_func
