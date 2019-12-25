import os
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
from mmdnn.conversion.tensorflow.tensorflow_emitter import TensorflowEmitter
from mmdnn.conversion.tensorflow.tensorflow_parser import TensorflowParser

from mmx import Graph, Op
from mmx.exporter import export_to_protobuf
from mmx.importer import import_from_protobuf

np.random.seed(42)

current_dir = os.path.dirname(os.path.abspath(__file__))
arch_name = "vgg16"


def store_as_numpy(input: np.array, store_dir, file_name):
    np.save(f"{store_dir}/{file_name}", input)
    return input


def new_debug_op(input, store_dir, file_name) -> Op:
    return Op(
        attrs=dict(
            name=f"debug_{file_name}",
            type="PyFunc",
            func=partial(store_as_numpy, store_dir=store_dir, file_name=file_name),
        ),
        inputs=[input],
    )


def modify_graph(graph: Graph):
    store_dir = Path(current_dir).parents[3] / "store" / arch_name
    debug_ops = []
    for op in graph.ops:
        debug_op = new_debug_op(input=op, store_dir=store_dir, file_name=op.name)
        debug_ops.append(debug_op)
        for downstream_op in op.output_ops(graph):
            for index, input_op in enumerate(downstream_op.input_ops):
                if input_op == op:
                    downstream_op.inputs[index] = debug_op.output(0)
    for debug_op in debug_ops:
        graph.add(debug_op)


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


def run_model():
    cache_dir = Path(current_dir).parents[3] / "test_data" / "modified"
    meta_file = str(cache_dir / ("imagenet_" + arch_name + ".ckpt.meta"))
    checkpoint_file = str(cache_dir / ("imagenet_" + arch_name + ".ckpt"))
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, checkpoint_file)
        sess.run("MMdnn_Output", {"input:0": np.random.rand(1, 224, 224, 3)})


if __name__ == "__main__":
    modify_model()
    run_model()
