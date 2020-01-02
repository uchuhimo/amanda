import os
from pathlib import Path

import google.protobuf.text_format
import jsondiff
import numpy as np
import pytest
import tensorflow as tf

from amanda import Graph
from amanda.conversion.tensorflow import (
    diff_graph_def,
    export_to_checkpoint,
    export_to_graph_def,
    export_to_pbtxt,
    export_to_tf_graph,
    get_diff_after_conversion,
    import_from_checkpoint,
    import_from_graph_def,
    import_from_pbtxt,
    import_from_tf_func,
)
from amanda.tests.utils import root_dir


@pytest.fixture(
    params=[
        # "vgg16",
        # # "vgg19",
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
            debug_output = import_from_tf_func(tf.identity)(graph)(tensor)
            for edge in output_edges:
                if edge.dst_op.type != "Assign":
                    edge.dst_op.input_tensors[edge.dst_input_index] = debug_output


def modify_model(arch_name):
    prefix_dir = root_dir() / "tmp"
    original_checkpoint = tf.train.latest_checkpoint(prefix_dir / "model" / arch_name)
    print(f">>>>>>>>>>>>>>>> import from the original checkpoint {original_checkpoint}")
    graph = import_from_checkpoint(original_checkpoint)
    modify_graph(graph)

    modified_checkpoint = prefix_dir / "modified_model" / arch_name / arch_name
    export_to_checkpoint(graph, modified_checkpoint)
    print(f">>>>>>>>>>>>>>>> export to the modified checkpoint {modified_checkpoint}")


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
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(checkpoint_file + ".meta")
            saver.restore(sess, checkpoint_file)
            output = sess.run("MMdnn_Output:0", {"input:0": input})
            return output, graph.as_graph_def()


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


def test_tf_modify_graph(arch_name):
    input = np.random.rand(*input_shapes[arch_name])
    output, graph_def = run_model(arch_name, model_dir="model", input=input)
    modify_model(arch_name)
    new_output, new_graph_def = run_model(
        arch_name, model_dir="modified_model", input=input
    )

    graph_diff = diff_graph_def(graph_def, new_graph_def)
    assert jsondiff.delete not in graph_diff
    inserted_ops = graph_diff[jsondiff.insert]
    assert np.all([node["op"] == "Identity" for node in inserted_ops.values()])
    inserted_op_names = {node["name"] for node in inserted_ops.values()}
    updated_ops = graph_diff[jsondiff.update]
    for op_name, updated_op in updated_ops.items():
        assert [jsondiff.update] == list(updated_op.keys())
        assert ["input"] == list(updated_op[jsondiff.update].keys())
        input_diff = updated_op[jsondiff.update]["input"]
        # for data dependency
        if isinstance(input_diff, list):
            for input_name in input_diff:
                assert input_name in inserted_op_names
        # for control dependency
        elif isinstance(input_diff, dict):
            assert jsondiff.update not in input_diff
            assert len(input_diff[jsondiff.insert]) == len(input_diff[jsondiff.delete])
            for index, input_name in input_diff[jsondiff.insert]:
                assert input_name in inserted_op_names
        else:
            raise AssertionError(f"{input_diff} has unknown type {type(input_diff)}")
    assert np.allclose(output, new_output)


def test_tf_import_export_graph_def(arch_name):
    checkpoint_dir = root_dir() / "tmp" / "model" / arch_name
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    with tf.Graph().as_default() as tf_graph:
        tf.train.import_meta_graph(checkpoint_file + ".meta")
    assert get_diff_after_conversion(tf_graph.as_graph_def()) == {}


def test_tf_import_export_graph_def_with_saver(arch_name):
    checkpoint_dir = root_dir() / "tmp" / "model" / arch_name
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    input = np.random.rand(*input_shapes[arch_name])
    with tf.Graph().as_default() as tf_graph:
        with tf.Session() as session:
            saver = tf.train.import_meta_graph(checkpoint_file + ".meta")
            saver.restore(session, checkpoint_file)
            output = session.run("MMdnn_Output:0", {"input:0": input})
            graph_def = tf_graph.as_graph_def()
            graph = import_from_graph_def(graph_def, saver.saver_def)
            graph = graph.to_default_namespace()
    new_tf_graph, saver, session = export_to_tf_graph(graph)
    assert diff_graph_def(graph_def, new_tf_graph.as_graph_def()) == {}
    with new_tf_graph.as_default():
        with session:
            saver.restore(session, checkpoint_file)
            new_output = session.run("MMdnn_Output:0", {"input:0": input})
    assert np.allclose(output, new_output)


def test_tf_import_export_graph_pbtxt(arch_name, tmp_path):
    checkpoint_dir = root_dir() / "tmp" / "model" / arch_name
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    with tf.Graph().as_default() as tf_graph:
        tf.train.import_meta_graph(checkpoint_file + ".meta")
    graph_pbtxt = google.protobuf.text_format.MessageToString(tf_graph.as_graph_def())
    pbtxt_file = tmp_path / "graph.pbtxt"
    pbtxt_file.write_text(graph_pbtxt)
    graph = import_from_pbtxt(pbtxt_file)
    new_graph_pbtxt = export_to_pbtxt(graph)
    graph_def = tf.GraphDef()
    google.protobuf.text_format.Parse(graph_pbtxt, graph_def)
    new_graph_def = tf.GraphDef()
    google.protobuf.text_format.Parse(new_graph_pbtxt, new_graph_def)
    assert diff_graph_def(graph_def, new_graph_def) == {}


def test_tf_import_export_partitioned_graph():
    pbtxt_file = root_dir() / "src" / "amanda" / "tests" / "partition-graphs-0.pbtxt"
    graph_def = tf.GraphDef()
    google.protobuf.text_format.Parse(Path(pbtxt_file).read_text(), graph_def)
    graph = import_from_graph_def(graph_def)
    new_graph_def = export_to_graph_def(graph)
    assert diff_graph_def(graph_def, new_graph_def) == {}


if __name__ == "__main__":
    test_tf_modify_graph(arch_name="vgg16")
