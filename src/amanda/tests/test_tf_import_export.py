import os
import shutil
import tempfile
from pathlib import Path

import google.protobuf.text_format
import jsondiff
import numpy as np
import pytest
import tensorflow as tf
from loguru import logger

import amanda
from amanda import Graph, create_op
from amanda.conversion.tensorflow import (
    after_op_executed_hook,
    before_op_executed_hook,
    export_to_checkpoint,
    export_to_graph,
    export_to_graph_def,
    export_to_pbtxt,
    export_to_saved_model,
    import_from_checkpoint,
    import_from_graph,
    import_from_graph_def,
    import_from_pbtxt,
    import_from_saved_model,
    import_from_tf_func,
)
from amanda.io.file import ensure_dir, root_dir
from amanda.tests.utils import diff_graph_def, get_diff_after_conversion


@pytest.fixture(
    params=[
        "vgg16",
        # "vgg19",
        pytest.param("inception_v1", marks=pytest.mark.slow),
        # "inception_v3",
        # "resnet_v1_50",
        # "resnet_v1_152",
        # "resnet_v2_50",
        pytest.param("resnet_v2_50", marks=pytest.mark.slow),
        # "resnet_v2_101",
        # "resnet_v2_152",
        # "mobilenet_v1_1.0",
        pytest.param("mobilenet_v2_1.0_224", marks=pytest.mark.slow),
        # "inception_resnet_v2",
        # "nasnet-a_large",
        # "facenet",
        # "rnn_lstm_gru_stacked",
    ]
)
def arch_name(request):
    return request.param


def modify_graph_with_primitive_api(graph: Graph):
    for op in graph.ops:
        for output_port in op.output_ports:
            if not output_port.type.raw._is_ref_dtype:
                debug_op = create_op(
                    name=f"debug/{op.name}/{output_port.name}",
                    type="Identity",
                    attrs={"T": output_port.type.raw},
                )
                edges = output_port.out_edges
                graph.add_op(debug_op)
                graph.create_edge(output_port, debug_op.input_port(0))
                for edge in edges:
                    graph.create_edge(debug_op.output_port(0), edge.dst)
                    graph.remove_edge(edge)


def modify_graph_with_tf_func(graph: Graph):
    for op in graph.ops:
        for output_port in op.output_ports:
            if not output_port.type.raw._is_ref_dtype:
                edges = output_port.out_edges
                debug_output = import_from_tf_func(tf.identity)(graph)(output_port)
                for edge in edges:
                    graph.create_edge(debug_output, edge.dst)
                    graph.remove_edge(edge)


def modify_model(arch_name, output_model_dir, modify_graph_func):
    original_checkpoint = tf.train.latest_checkpoint(
        root_dir() / "downloads" / "model" / arch_name
    )
    logger.info(f"import from the original checkpoint {original_checkpoint}")
    graph = import_from_checkpoint(original_checkpoint)
    modify_graph_func(graph)
    modified_checkpoint = root_dir() / "tmp" / output_model_dir / arch_name / arch_name
    export_to_checkpoint(graph, modified_checkpoint)
    logger.info(f"export to the modified checkpoint {modified_checkpoint}")


def run_model(arch_name, model_dir, input):
    checkpoint_dir = root_dir() / model_dir / arch_name
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
            output = sess.run("Output:0", {"input:0": input})
            return output, graph.as_graph_def()


def run_model_with_estimator(arch_name, model_dir, input, tool=None):
    checkpoint_dir = root_dir() / model_dir / arch_name
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"{checkpoint_dir} is not existed")
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_file is None:
        raise FileNotFoundError(
            f"cannot find checkpoint in {checkpoint_dir}, "
            f"only find: {os.listdir(checkpoint_dir)}"
        )

    def model_fn(features, labels, mode):
        tf.train.import_meta_graph(
            checkpoint_file + ".meta", input_map={"input:0": features}
        )
        logits = tf.get_default_graph().get_tensor_by_name("Output:0")
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels
        )
        weight_decay = 1e-4
        loss = cross_entropy + weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
        )
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            learning_rate = 1e-4
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9
            )
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
        else:
            train_op = None
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )

    def input_fn():
        labels = np.array([0])
        dataset = tf.data.Dataset.from_tensor_slices((input.astype(np.float32), labels))
        return dataset.batch(1)

    warm_start = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=checkpoint_file)
    model_dir = tempfile.mkdtemp()
    try:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            warm_start_from=warm_start,
        )
        if tool is not None:
            amanda.apply(estimator, tool)
        else:
            amanda.tensorflow.inject_hook(estimator)
        estimator.train(input_fn=input_fn, steps=1)
    finally:
        shutil.rmtree(model_dir)


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


def test_tf_serialize_graph(arch_name):
    original_checkpoint = tf.train.latest_checkpoint(
        root_dir() / "downloads" / "model" / arch_name
    )
    graph_path = root_dir() / "tmp" / "tf_graph" / arch_name / arch_name
    graph = import_from_checkpoint(original_checkpoint)
    graph_def = export_to_graph_def(graph)
    amanda.io.save_to_proto(graph, graph_path)
    graph = amanda.io.load_from_proto(graph_path)
    amanda.io.save_to_yaml(graph, graph_path)
    new_graph = amanda.io.load_from_yaml(graph_path)
    new_graph_def = export_to_graph_def(new_graph)
    assert diff_graph_def(graph_def, new_graph_def) == {}


def test_tf_modify_graph(arch_name):
    input = np.random.rand(*input_shapes[arch_name])
    output, graph_def = run_model(arch_name, model_dir="downloads/model", input=input)
    modify_model(arch_name, "modified_graph", modify_graph_with_primitive_api)
    new_output, new_graph_def = run_model(
        arch_name, model_dir="tmp/modified_graph", input=input
    )
    check_modified_graph(graph_def, new_graph_def)
    np.testing.assert_allclose(output, new_output)


def test_tf_modify_graph_with_tf_func(arch_name):
    input = np.random.rand(*input_shapes[arch_name])
    output, graph_def = run_model(arch_name, model_dir="downloads/model", input=input)
    modify_model(arch_name, "modified_graph_with_tf_func", modify_graph_with_tf_func)
    new_output, new_graph_def = run_model(
        arch_name, model_dir="tmp/modified_graph_with_tf_func", input=input
    )
    check_modified_graph(graph_def, new_graph_def)
    np.testing.assert_allclose(output, new_output)


def test_tf_modify_graph_with_hook(arch_name):
    from amanda.tools.debugging.insert_debug_op_tensorflow import DebuggingTool

    input = np.random.rand(*input_shapes[arch_name])
    run_model(arch_name, model_dir="downloads/model", input=input)
    store_dir = root_dir() / "tmp" / "debug_info_with_hook" / arch_name
    ensure_dir(store_dir, is_file=False)
    run_model_with_estimator(
        arch_name,
        model_dir="downloads/model",
        input=input,
        tool=DebuggingTool(store_dir),
    )


class TestTool(amanda.Tool):
    def __init__(self, store_dir):
        super(TestTool, self).__init__(namespace="amanda/tensorflow")
        self.register_event(amanda.event.before_op_added, self.before_op_added)
        self.register_event(amanda.event.after_op_added, self.after_op_added)
        self.register_event(
            amanda.event.before_op_executed, self.test_before_op_executed
        )
        self.register_event(amanda.event.after_op_executed, self.test_after_op_executed)
        self.register_event(
            amanda.event.before_backward_op_executed,
            self.test_before_backward_op_executed,
        )
        self.register_event(
            amanda.event.after_backward_op_executed,
            self.test_after_backward_op_executed,
        )
        self.store_dir = store_dir
        self.before_added_ops = set()
        self.after_added_ops = set()
        self.before_executed_ops = set()
        self.after_executed_ops = set()

    def before_op_added(self, context: amanda.EventContext):
        op = context["op"]
        self.before_added_ops.add(op.name)
        input_with_index = [
            (index, input)
            for index, input in enumerate(context["inputs"])
            if not input.dtype._is_ref_dtype
        ]
        inputs = [input for _, input in input_with_index]
        input_indices = [index for index, _ in input_with_index]
        input_types = op._input_types
        input_types = [input_types[index] for index in input_indices]
        with tf.control_dependencies(op.control_inputs):
            new_inputs = tf.py_function(
                before_op_executed_hook(context),
                inputs,
                input_types,
                name=f"{op.name}_before_op_executed",
            )
        if len(inputs) == 0:
            new_op = new_inputs
            op._add_control_input(new_op)
            new_inputs = []
        for index, new_input in zip(input_indices, new_inputs):
            context["inputs"][index] = new_input

    def after_op_added(self, context: amanda.EventContext):
        op = context["op"]
        self.after_added_ops.add(op.name)
        output_with_index = [
            (index, output)
            for index, output in enumerate(context["outputs"])
            if not output.dtype._is_ref_dtype
        ]
        outputs = [output for _, output in output_with_index]
        output_indices = [index for index, _ in output_with_index]
        output_types = op._output_types
        output_types = [output_types[index] for index in output_indices]
        new_outputs = tf.py_function(
            after_op_executed_hook(context),
            outputs,
            output_types,
            name=f"{op.name}_after_op_executed",
        )
        if len(outputs) == 0:
            new_op = new_outputs
            new_outputs = []
        else:
            new_op = new_outputs[0].op
        for control_output in op._control_outputs:
            control_output._add_control_input(new_op)
        if len(outputs) == 0:
            new_op._add_control_input(op)
        for index, new_output in zip(output_indices, new_outputs):
            context["outputs"][index] = new_output

    def test_before_op_executed(self, context: amanda.EventContext):
        op = context["op"]
        self.before_executed_ops.add(op.name)
        file_name = op.name.replace("/", "_")
        # print(
        #     "before",
        #     op.type,
        #     [input.dtype for input in context["inputs"]],
        #     op.name
        # )
        store_dir = self.store_dir / "before_op_executed" / file_name
        ensure_dir(store_dir, is_file=False)
        for index, input in enumerate(context["inputs"]):
            np.save(f"{store_dir}/{index}", input)

    def test_after_op_executed(self, context: amanda.EventContext):
        op = context["op"]
        self.after_executed_ops.add(op.name)
        file_name = op.name.replace("/", "_")
        # print(
        #     "after",
        #     op.type,
        #     # [input.dtype for input in context["inputs"]],
        #     [output.dtype for output in context["outputs"]],
        #     op.name,
        # )
        store_dir = self.store_dir / "after_op_executed" / file_name
        ensure_dir(store_dir, is_file=False)
        # for index, input in enumerate(context["inputs"]):
        #     np.save(f"{store_dir}/input_{index}", input)
        for index, output in enumerate(context["outputs"]):
            np.save(f"{store_dir}/{index}", output)

    def test_before_backward_op_executed(self, context: amanda.EventContext):
        op = context["op"]
        file_name = op.name.replace("/", "_")
        # backward_op = context["backward_op"]
        # print(
        #     "before_bw",
        #     op.type,
        #     backward_op.type,
        #     [input.dtype for input in context["outputs"]],
        #     [input.dtype for input in context["grad_outputs"]],
        #     op.name,
        #     backward_op.name
        # )
        store_dir = self.store_dir / "before_backward_op_executed" / file_name
        ensure_dir(store_dir, is_file=False)
        # for index, input in enumerate(context["outputs"]):
        #     np.save(f"{store_dir}/{index}", input)
        for index, input in enumerate(context["grad_outputs"]):
            np.save(f"{store_dir}/grad_{index}", input)

    def test_after_backward_op_executed(self, context: amanda.EventContext):
        op = context["op"]
        file_name = op.name.replace("/", "_")
        # backward_op = context["backward_op"]
        # print(
        #     "after_bw",
        #     op.type,
        #     backward_op.type,
        #     [input.dtype for input in context["inputs"]],
        #     [input.dtype for input in context["grad_inputs"]],
        #     op.name,
        #     backward_op.name,
        # )
        store_dir = self.store_dir / "after_backward_op_executed" / file_name
        ensure_dir(store_dir, is_file=False)
        # for index, output in enumerate(context["inputs"]):
        #     np.save(f"{store_dir}/{index}", output)
        for index, output in enumerate(context["grad_inputs"]):
            np.save(f"{store_dir}/grad_{index}", output)


def test_tf_modify_graph_with_new_hook(arch_name):
    input = np.random.rand(*input_shapes[arch_name])
    run_model(arch_name, model_dir="downloads/model", input=input)
    store_dir = root_dir() / "tmp" / "debug_info_with_new_hook" / arch_name
    tool = TestTool(store_dir)
    with amanda.tool.apply(tool):
        run_model_with_estimator(
            arch_name,
            model_dir="downloads/model",
            input=input,
        )
    assert tool.before_added_ops == tool.after_added_ops
    assert tool.before_executed_ops == tool.after_executed_ops


def check_modified_graph(graph_def, new_graph_def):
    graph_diff = diff_graph_def(graph_def, new_graph_def)
    assert [jsondiff.update] == list(graph_diff.keys())
    assert {"node"} == set(graph_diff[jsondiff.update].keys())
    node_diff = graph_diff[jsondiff.update]["node"]
    assert jsondiff.delete not in node_diff
    inserted_ops = node_diff[jsondiff.insert]
    assert np.all([node["op"] == "Identity" for node in inserted_ops.values()])
    inserted_op_names = {node["name"] for node in inserted_ops.values()}
    updated_ops = node_diff[jsondiff.update]
    for updated_op in updated_ops.values():
        assert [jsondiff.update] == list(updated_op.keys())
        assert ["input"] == list(updated_op[jsondiff.update].keys())
        input_diff = updated_op[jsondiff.update]["input"]
        if isinstance(input_diff, list):
            for input_name in input_diff:
                assert input_name in inserted_op_names or input_name.startswith("^")
        elif isinstance(input_diff, dict):
            assert jsondiff.update not in input_diff
            assert len(input_diff[jsondiff.insert]) == len(input_diff[jsondiff.delete])
            for (insert_index, input_name), delete_index in zip(
                sorted(input_diff[jsondiff.insert], key=lambda x: x[0]),
                sorted(input_diff[jsondiff.delete]),
            ):
                assert insert_index == delete_index
                assert input_name in inserted_op_names or input_name.startswith("^")
        else:
            raise AssertionError(f"{input_diff} has unknown type {type(input_diff)}")


def test_tf_import_export_graph_def(arch_name):
    checkpoint_dir = root_dir() / "downloads" / "model" / arch_name
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    with tf.Graph().as_default() as tf_graph:
        tf.train.import_meta_graph(checkpoint_file + ".meta")
    assert get_diff_after_conversion(tf_graph.as_graph_def()) == {}


def test_tf_import_export_graph(arch_name):
    checkpoint_dir = root_dir() / "downloads" / "model" / arch_name
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    with tf.Graph().as_default() as tf_graph:
        tf.train.import_meta_graph(checkpoint_file + ".meta")
    graph = import_from_graph(tf_graph)
    new_tf_graph, _, _ = export_to_graph(graph)
    assert diff_graph_def(tf_graph.as_graph_def(), new_tf_graph.as_graph_def()) == {}


def test_tf_import_export_saved_model(arch_name, tmp_path):
    checkpoint_dir = root_dir() / "downloads" / "model" / arch_name
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    input = np.random.rand(*input_shapes[arch_name])
    path = str(tmp_path / arch_name)
    new_path = str(tmp_path / f"new_{arch_name}")
    with tf.Graph().as_default() as tf_graph:
        with tf.Session() as session:
            saver = tf.train.import_meta_graph(checkpoint_file + ".meta")
            saver.restore(session, checkpoint_file)
            output = session.run("Output:0", {"input:0": input})
            builder = tf.saved_model.builder.SavedModelBuilder(path)
            builder.add_meta_graph_and_variables(
                session,
                [tf.saved_model.tag_constants.SERVING],
                strip_default_attrs=True,
                saver=saver,
            )
            builder.save()
            graph = import_from_saved_model(
                path, tags=[tf.saved_model.tag_constants.SERVING]
            )
    export_to_saved_model(graph, new_path, tags=[tf.saved_model.tag_constants.SERVING])
    with tf.Graph().as_default() as new_tf_graph:
        with tf.Session() as session:
            tf.saved_model.load(
                session, [tf.saved_model.tag_constants.SERVING], new_path
            )
            assert (
                diff_graph_def(tf_graph.as_graph_def(), new_tf_graph.as_graph_def())
                == {}
            )
            new_output = session.run("Output:0", {"input:0": input})
    np.testing.assert_allclose(output, new_output)


def test_tf_import_export_graph_pbtxt(arch_name, tmp_path):
    checkpoint_dir = root_dir() / "downloads" / "model" / arch_name
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


@pytest.fixture(params=["partition-graphs-0.pbtxt", "placer_input_1.pbtxt"])
def partitioned_graph_file(request):
    return request.param


def test_tf_import_export_partitioned_graph(partitioned_graph_file):
    pbtxt_file = root_dir() / "src" / "amanda" / "tests" / partitioned_graph_file
    graph_def = tf.GraphDef()
    google.protobuf.text_format.Parse(Path(pbtxt_file).read_text(), graph_def)
    graph = import_from_graph_def(graph_def)
    new_graph_def = export_to_graph_def(graph)
    assert diff_graph_def(graph_def, new_graph_def) == {}


if __name__ == "__main__":
    test_tf_modify_graph(arch_name="vgg16")
