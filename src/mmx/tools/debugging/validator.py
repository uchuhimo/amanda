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
    import_from_checkpoint,
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
    store_dir = Path(current_dir).parents[3] / "store" / arch_name
    for op in graph.ops:
        output: OutputPort[Op] = convert_from_tf_func(tf.py_func, graph)(
            partial(store_as_numpy, store_dir=store_dir, file_name=op.name),
            [input],
            tf.float32,
        )
        output.op.insert_after(op, graph)


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


def run_model(variant):
    cache_dir = Path(current_dir).parents[3] / "test_data" / variant
    meta_file = str(cache_dir / ("imagenet_" + arch_name + ".ckpt.meta"))
    checkpoint_file = str(cache_dir / ("imagenet_" + arch_name + ".ckpt"))
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, checkpoint_file)
        output = sess.run("MMdnn_Output:0", {"input:0": np.random.rand(1, 224, 224, 3)})
        print(output)


if __name__ == "__main__":
    run_model("cache")
    modify_model_v2()
    run_model("modified")
