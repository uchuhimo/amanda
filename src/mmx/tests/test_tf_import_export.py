import os

import numpy as np
import pytest
import tensorflow as tf

from mmx import Graph
from mmx.conversion.tensorflow import (
    convert_from_tf_func,
    export_to_checkpoint,
    import_from_checkpoint,
)
from mmx.tests.utils import root_dir


@pytest.fixture(
    params=[
        "vgg16",
        "vgg19",
        "inception_v1",
        "inception_v3",
        "resnet_v1_50",
        # # "resnet_v1_152",
        "resnet_v2_50",
        "resnet_v2_101",
        # # "resnet_v2_152",
        # # "resnet_v2_200",
        "mobilenet_v1_1.0",
        "mobilenet_v2_1.0_224",
        "inception_resnet_v2",
        "nasnet-a_large",
        "facenet",
        "rnn_lstm_gru_stacked",
    ]
)
def arch_name(request):
    return request.param


def modify_graph(graph: Graph):
    original_graph = graph.clone()
    for op in original_graph.post_order_ops:
        for output_port in op.output_ports(original_graph):
            output_edges = output_port.output_edges(original_graph)
            debug_output = convert_from_tf_func(tf.identity, graph)(output_port)
            for edge in output_edges:
                if edge.dst.type != "Assign":
                    edge.dst.inputs[edge.dst_input_index] = debug_output


def modify_model(arch_name):
    prefix_dir = root_dir() / "tmp"
    graph = import_from_checkpoint(
        tf.train.latest_checkpoint(prefix_dir / "model" / arch_name)
    )
    modify_graph(graph)
    export_to_checkpoint(graph, prefix_dir / "modified_model" / arch_name / arch_name)


def run_model(arch_name, model_dir, input):
    checkpoint_dir = root_dir() / "tmp" / model_dir / arch_name
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"{checkpoint_dir} is not existed")
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_file is None:
        raise FileNotFoundError(
            f"cannot find checkpoint in {checkpoint_dir}, "
            f"only find: {os.listdir(checkpoint_dir)}"
        )
    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(checkpoint_file + ".meta")
            saver.restore(sess, checkpoint_file)
            output = sess.run("MMdnn_Output:0", {"input:0": input})
            return output


input_shapes = {
    "vgg16": (1, 224, 224, 3),
    "vgg19": (1, 224, 224, 3),
    "inception_v1": (1, 224, 224, 3),
    "inception_v3": (1, 299, 299, 3),
    "resnet_v1_50": (1, 224, 224, 3),
    "resnet_v1_152": (1, 224, 224, 3),
    "resnet_v2_50": (1, 299, 299, 3),
    "resnet_v2_101": (1, 299, 299, 3),
    "resnet_v2_152": (1, 299, 299, 3),
    "resnet_v2_200": (1, 299, 299, 3),
    "mobilenet_v1_1.0": (1, 224, 224, 3),
    "mobilenet_v2_1.0_224": (1, 224, 224, 3),
    "inception_resnet_v2": (1, 299, 299, 3),
    "nasnet-a_large": (1, 331, 331, 3),
    "facenet": (1, 160, 160, 3),
    "rnn_lstm_gru_stacked": (1, 150),
}


def test_tf_import_export(arch_name):
    input = np.random.rand(*input_shapes[arch_name])
    output = run_model(arch_name, model_dir="model", input=input)
    modify_model(arch_name)
    new_output = run_model(arch_name, model_dir="modified_model", input=input)
    assert np.allclose(output, new_output)
