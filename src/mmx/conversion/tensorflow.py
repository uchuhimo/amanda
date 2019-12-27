from pathlib import Path
from typing import Union

import tensorflow as tf

from mmx import Graph, Op, OutputPort


def import_from_tf_graph(tf_graph: tf.Graph) -> Graph:
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

        def get_output_port(tensor: tf.Tensor) -> OutputPort:
            input_op = tensor.op
            for index, output_tensor in enumerate(input_op.outputs):
                if output_tensor == tensor:
                    return graph.get_op_by_name(input_op.name).output(index)
            raise RuntimeError()

        op = Op(
            attrs=dict(name=tf_op.name, type=tf_op.type, device=tf_op.device, **attrs,),
            inputs=[get_output_port(input_tensor) for input_tensor in tf_op.inputs],
            control_inputs=[
                graph.get_op_by_name(control_input_op.name)
                for control_input_op in tf_op.control_inputs
            ],
        )
        graph.add(op)

    for tf_op in tf_graph.get_operations():
        add_op(tf_op)
    return graph


def import_from_graph_def(graph_def: Union[tf.GraphDef, str, Path]) -> Graph:
    if not isinstance(graph_def, tf.GraphDef):
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(Path(graph_def).read_bytes())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
        return import_from_tf_graph(graph)


def import_from_meta_graph(
    meta_graph: Union[tf.MetaGraphDef, str, Path], checkpoint: Union[str, Path] = None,
) -> Graph:
    if isinstance(meta_graph, tf.MetaGraphDef):
        meta_graph = meta_graph.SerializeToString()
    with tf.Graph().as_default() as graph:
        saver = tf.train.import_meta_graph(meta_graph)
        if checkpoint is not None:
            with tf.Session() as sess:
                saver.restore(sess, str(checkpoint))
        return import_from_tf_graph(graph)


def import_from_checkpoint(path: Union[str, Path]) -> Graph:
    path = str(path)
    return import_from_meta_graph(path + ".meta", path)


def export_to_tf_graph(graph: Graph) -> tf.Graph:
    ...


def export_to_checkpoint(graph: Graph, path: Union[str, Path]) -> None:
    tf_graph = export_to_tf_graph(graph)
    with tf_graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.save(sess, str(path))


def convert_from_tf_func(tf_func, graph: Graph):
    def new_func(*args, **kwargs):
        tf_args = dict(args)
        tf_kwargs = dict(kwargs)
        inputs = []
        placeholders = []
        for index, tf_arg in enumerate(tf_args):
            arg = args[index]
            if isinstance(arg, OutputPort):
                placeholder = tf.placeholder(arg.op.attrs["T"])
                tf_args[index] = placeholder
                inputs.append(arg)
                placeholders.append(placeholder)
            elif isinstance(arg, list):
                arg_list = arg
                for i, arg in enumerate(arg_list):
                    if isinstance(arg, OutputPort):
                        placeholder = tf.placeholder(arg.op.attrs["T"])
                        tf_args[index][i] = placeholder
                        inputs.append(arg)
                        placeholders.append(placeholder)
        for name, tf_arg in tf_kwargs.items():
            arg = kwargs[name]
            if isinstance(arg, OutputPort):
                placeholder = tf.placeholder(arg.op.attrs["T"])
                tf_kwargs[name] = placeholder
                inputs.append(arg)
                placeholders.append(placeholder)
            elif isinstance(arg, list):
                arg_list = arg
                for i, arg in enumerate(arg_list):
                    if isinstance(arg, OutputPort):
                        placeholder = tf.placeholder(arg.op.attrs["T"])
                        tf_kwargs[name][i] = placeholder
                        inputs.append(arg)
                        placeholders.append(placeholder)
        tf_func(*tf_args, **tf_kwargs)

    return new_func
