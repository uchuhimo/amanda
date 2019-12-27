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

from mmx import Graph, Op, OutputPort


def import_from_tf_graph(tf_graph: tf.Graph, saver_def: SaverDef = None) -> Graph:
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
            attrs=dict(name=tf_op.name, type=tf_op.type, device=tf_op.device, **attrs),
            inputs=[
                from_tf_tensor(input_tensor, graph) for input_tensor in tf_op.inputs
            ],
            control_inputs=[
                graph.get_op_by_name(control_input_op.name)
                for control_input_op in tf_op.control_inputs
            ],
        )
        graph.add(op)

    for tf_op in tf_graph.get_operations():
        add_op(tf_op)
    meta_graph: tf.MetaGraphDef = tf.train.export_meta_graph(
        graph=tf_graph, saver_def=saver_def
    )
    meta_graph.graph_def.Clear()
    graph.attrs["meta_graph"] = meta_graph
    graph.attrs["variable_values"] = [
        variable.read_value()
        for variable in tf_graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    ]
    return graph


def from_tf_tensor(tensor: Union[tf.Tensor, RefVariable], graph: Graph) -> OutputPort:
    op = graph.get_op_by_name(tensor.op.name)
    if op.type == "VariableV2":
        return op.output(0)
    else:
        return op.output(tensor.value_index)


def import_from_graph_def(
    graph_def: Union[tf.GraphDef, str, Path], saver_def: SaverDef = None,
) -> Graph:
    if not isinstance(graph_def, tf.GraphDef):
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(Path(graph_def).read_bytes())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
        return import_from_tf_graph(graph, saver_def)


def import_from_meta_graph(
    meta_graph: Union[tf.MetaGraphDef, str, Path], checkpoint: Union[str, Path] = None,
) -> Graph:
    with tf.Graph().as_default() as tf_graph:
        saver = tf.train.import_meta_graph(meta_graph)
        if checkpoint is not None:
            with tf.Session() as sess:
                saver.restore(sess, str(checkpoint))
        graph = import_from_tf_graph(tf_graph, saver.saver_def)
        return graph


def import_from_checkpoint(path: Union[str, Path]) -> Graph:
    path = str(path)
    return import_from_meta_graph(path + ".meta", path)


def export_to_tf_graph(graph: Graph) -> Tuple[tf.Graph, tf.train.Saver]:
    tf_graph: tf.Graph
    with tf.Graph().as_default() as tf_graph:
        for op in graph.post_order_ops:
            attrs = dict(op.attrs)
            del attrs["name"]
            del attrs["type"]
            del attrs["device"]
            with tf.control_dependencies(
                [
                    tf_graph.get_operation_by_name(control_input.name)
                    for control_input in op.control_inputs
                ]
            ):
                tf_op = tf_graph.create_op(
                    op_type=op.type,
                    inputs=[
                        tf_graph.get_tensor_by_name(
                            f"{tensor.op.name}:{tensor.output_index}"
                        )
                        for tensor in op.inputs
                    ],
                    dtypes=get_dtypes(op),
                    input_types=None,
                    name=op.name,
                    attrs=to_attrs_proto(tf_graph._get_op_def(op.type), op.type, attrs),
                )
            if "device" in op.attrs:
                tf_op._set_device(op.attrs["device"])
        saver = tf.train.import_meta_graph(graph.attrs["meta_graph"])
        return tf_graph, saver


def export_to_checkpoint(graph: Graph, path: Union[str, Path]) -> None:
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(mode=0o755, parents=True)
    tf_graph, saver = export_to_tf_graph(graph)
    with tf_graph.as_default():
        with tf.Session() as sess:
            if saver is None:
                saver = tf.train.Saver()
            saver.save(sess, str(path))


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
    if "dtype" in op.attrs:
        return op.attrs["dtype"]
    elif "dtypes" in op.attrs:
        return op.attrs["dtypes"]
    elif "Tout" in op.attrs:
        return op.attrs["Tout"]
    elif "T" in op.attrs:
        return op.attrs["T"]
    else:
        return []


def get_dtype(tensor: OutputPort):
    dtypes = get_dtypes(tensor.op)
    assert isinstance(dtypes, (list, tf.DType))
    if isinstance(dtypes, list):
        return dtypes[tensor.output_index]
    else:
        return dtypes


def convert_from_tf_func(tf_func, graph: Graph):
    def new_func(*args, **kwargs):
        args_dict = dict(enumerate(args))
        tf_args = list(args)
        tf_kwargs = dict(kwargs)
        inputs: List[OutputPort] = []
        placeholders: List[tf.Tensor] = []

        def to_tf_tensor(tensor: OutputPort) -> tf.Tensor:
            placeholder = tf.placeholder(get_dtype(tensor))
            inputs.append(tensor)
            placeholders.append(placeholder)
            return placeholder

        def get_args(name: Union[int, str]) -> Dict[Any, Any]:
            return args_dict if isinstance(name, int) else kwargs

        def get_tf_args(name: Union[int, str]) -> Dict[Any, Any]:
            return tf_args if isinstance(name, int) else tf_kwargs

        with tf.Graph().as_default() as tf_graph:
            for name in graph.names:
                tf_graph.unique_name(name, mark_as_used=True)
            all_args = {**args_dict, **kwargs}
            for name in all_args:
                arg = get_args(name)[name]
                if isinstance(arg, OutputPort):
                    get_tf_args(name)[name] = to_tf_tensor(arg)
                if isinstance(arg, list):
                    arg_list = arg
                    for i, arg in enumerate(arg_list):
                        if isinstance(arg, OutputPort):
                            get_tf_args(name)[name][i] = to_tf_tensor(arg)
            output_tensor: tf.Tensor = tf_func(*tf_args, **tf_kwargs)
            new_graph = import_from_tf_graph(tf_graph)
            for input, placeholder in zip(inputs, placeholders):
                placeholder_op = new_graph.get_op_by_name(placeholder.op.name)
                new_graph.replace_tensor(placeholder_op.output(0), input)
                new_graph.remove(placeholder_op)
            for op in new_graph.ops:
                graph.add(op)
            meta_graph = graph.attrs["meta_graph"]
            current_meta_graph = tf.train.export_meta_graph(graph=tf_graph)
            meta_graph.collection_def.MergeFrom(current_meta_graph.collection_def)
            output_op = graph.get_op_by_name(output_tensor.op.name)
            return output_op.output(output_tensor.value_index)

    return new_func
