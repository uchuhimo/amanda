import os
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
from mmdnn.conversion.tensorflow.tensorflow_emitter import TensorflowEmitter
from mmdnn.conversion.tensorflow.tensorflow_parser import TensorflowParser

from mmx import Graph, Op, OutputPort
from mmx.conversion.tensorflow import (
    convert_from_tf_func,
    export_to_checkpoint,
    get_dtype,
    import_from_checkpoint,
    import_from_tf_graph,
)
from mmx.exporter import export_to_protobuf
from mmx.importer import import_from_protobuf

np.random.seed(42)

current_dir = os.path.dirname(os.path.abspath(__file__))
arch_name = "vgg16"


def store_as_numpy(input: np.array, store_dir, file_name):
    np.save(f"{store_dir}/{file_name}", input)
    return input


def modify_graph(graph: Graph):
    store_dir = Path(current_dir).parents[3] / "test_data" / "debug" / arch_name
    for op in graph.clone().post_order_ops:
        for output_port in op.output_ports(graph):
            debug_output: OutputPort[Op] = convert_from_tf_func(tf.py_func, graph)(
                partial(
                    store_as_numpy,
                    store_dir=store_dir,
                    file_name=f"{op.name}:{output_port.output_index}",
                ),
                [output_port],
                get_dtype(output_port),
            )
            for edge in op.output_edges_from_port(output_port, graph):
                edge.dst.inputs[edge.dst_input_index] = debug_output


def modify_model():
    cache_dir = Path(current_dir).parents[3] / "test_data" / "cache"
    meta_file = str(cache_dir / ("imagenet_" + arch_name + ".ckpt.meta"))
    checkpoint_file = str(cache_dir / ("imagenet_" + arch_name + ".ckpt"))
    parser = TensorflowParser(meta_file, checkpoint_file, ["MMdnn_Output"],)
    parser.gen_IR()
    model = parser.IR_graph
    graph = import_from_protobuf(model)
    modify_graph(graph)
    new_model = export_to_protobuf(graph)
    emitter = TensorflowEmitter(new_model)
    emitter.run(
        meta_file.replace("cache", "modified"),
        checkpoint_file.replace("cache", "modified"),
    )


def modify_model_v2():
    prefix_dir = Path(current_dir).parents[3] / "test_data"
    graph = import_from_checkpoint(
        prefix_dir / "cache" / ("imagenet_" + arch_name + ".ckpt")
    )
    modify_graph(graph)
    export_to_checkpoint(
        graph, prefix_dir / "modified" / ("imagenet_" + arch_name + ".ckpt")
    )


def run_model(model_dir, input):
    cache_dir = Path(current_dir).parents[3] / "test_data" / model_dir
    meta_file = str(cache_dir / ("imagenet_" + arch_name + ".ckpt.meta"))
    checkpoint_file = str(cache_dir / ("imagenet_" + arch_name + ".ckpt"))
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, checkpoint_file)
        output = sess.run("MMdnn_Output:0", {"input:0": input})
        return output


def test_control_dependencies():
    with tf.Graph().as_default() as tf_graph:
        x1 = tf.placeholder(tf.float32)
        x2 = tf.placeholder(tf.float32)
        with tf.control_dependencies([x1, x2]):
            tf.placeholder(tf.float32)
        graph = import_from_tf_graph(tf_graph)
        print(graph)


if __name__ == "__main__":
    input = np.random.rand(1, 224, 224, 3)
    output = run_model(model_dir="cache", input=input)
    modify_model_v2()
    new_output = run_model(model_dir="modified", input=input)
    assert np.allclose(output, new_output)
