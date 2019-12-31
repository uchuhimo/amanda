import json
import os
from pathlib import Path

import google.protobuf.text_format
import numpy as np
import pytest
import tensorflow as tf
from google.protobuf import json_format
from jsondiff import diff

from amanda import Graph
from amanda.conversion.tensorflow import (
    convert_from_tf_func,
    export_to_checkpoint,
    export_to_pbtxt,
    export_to_tf_graph,
    import_from_checkpoint,
    import_from_graph_def,
    import_from_pbtxt,
)
from amanda.tests.utils import root_dir


@pytest.fixture(
    params=[
        "vgg16",
        # "vgg19",
        "inception_v1",
        # "inception_v3",
        # "resnet_v1_50",
        # # "resnet_v1_152",
        "resnet_v2_50",
        # "resnet_v2_101",
        # # "resnet_v2_152",
        # # "resnet_v2_200",
        # "mobilenet_v1_1.0",
        "mobilenet_v2_1.0_224",
        # "inception_resnet_v2",
        # "nasnet-a_large",
        "facenet",
        # "rnn_lstm_gru_stacked",
    ]
)
def arch_name(request):
    return request.param


def modify_graph(graph: Graph):
    original_graph = graph.clone()
    for op in original_graph.ops:
        for tensor in op.output_tensors:
            output_edges = original_graph.edges_from_tensor(tensor)
            debug_output = convert_from_tf_func(tf.identity, graph)(tensor)
            for edge in output_edges:
                if edge.dst_op.type != "Assign":
                    edge.dst_op.input_tensors[edge.dst_input_index] = debug_output


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


def test_tf_import_export_graph_def(arch_name):
    checkpoint_dir = root_dir() / "tmp" / "model" / arch_name
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    input = np.random.rand(*input_shapes[arch_name])
    with tf.Graph().as_default() as tf_graph:
        with tf.Session() as session:
            saver = tf.train.import_meta_graph(checkpoint_file + ".meta")
            saver.restore(session, checkpoint_file)
            output = session.run("MMdnn_Output:0", {"input:0": input})
            graph = import_from_graph_def(tf_graph.as_graph_def(), saver.saver_def)
            graph = graph.to_default_namespace()
    tf_graph, saver, session = export_to_tf_graph(graph)
    with tf_graph.as_default():
        with session:
            saver.restore(session, checkpoint_file)
            new_output = session.run("MMdnn_Output:0", {"input:0": input})
    assert np.allclose(output, new_output)


def test_tf_import_export_partitioned_graph():
    pbtxt_file = root_dir() / "src" / "amanda" / "tests" / "partition-graphs-0.pbtxt"
    graph = import_from_pbtxt(pbtxt_file)
    new_graph_pbtxt = export_to_pbtxt(graph)

    graph_def = tf.GraphDef()
    google.protobuf.text_format.Parse(Path(pbtxt_file).read_text(), graph_def)
    new_graph_def = tf.GraphDef()
    google.protobuf.text_format.Parse(new_graph_pbtxt, new_graph_def)

    ops = {node.name: node for node in graph_def.node}
    new_ops = {node.name: node for node in new_graph_def.node}
    assert len(ops) == len(new_ops)
    ops_diff = {}
    for name in ops:
        assert name in new_ops
        op = ops[name]
        json_str = json_format.MessageToJson(op, preserving_proto_field_name=True)
        op_json = json.loads(json_str)
        new_op = new_ops[name]
        new_json_str = json_format.MessageToJson(
            new_op, preserving_proto_field_name=True
        )
        new_op_json = json.loads(new_json_str)
        op_diff = diff(op_json, new_op_json)
        if len(op_diff) != 0:
            ops_diff[name] = op_diff
    print(new_graph_pbtxt)
    assert ops_diff == {}
