from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import load_library

import amanda
from amanda.tests.download_model import download_tf_model
from amanda.tests.test_tf_import_export import run_model
from amanda.tests.utils import root_dir
from amanda.tool import Tool

store_tensor_to_file_ops = load_library.load_op_library(
    str(
        root_dir()
        / "cc/tensorflow/store_tensor_to_file/ops/store_tensor_to_file_ops.so"
    )
)
store_tensor_to_file = store_tensor_to_file_ops.store_tensor_to_file

arch_name = "vgg16"
original_checkpoint_dir = tf.train.latest_checkpoint(
    root_dir() / "downloads" / "model" / arch_name
)
assert original_checkpoint_dir is not None
modified_checkpoint_dir = root_dir() / "tmp" / "modified_model" / arch_name / arch_name
store_dir = root_dir() / "tmp" / "debug_info" / arch_name

if not Path(store_dir).exists():
    Path(store_dir).mkdir(mode=0o755, parents=True, exist_ok=True)

input = np.random.rand(1, 224, 224, 3)


def run_original_model():
    output, _ = run_model(arch_name, model_dir="downloads/model", input=input)
    return output


def run_modified_model():
    new_output, _ = run_model(arch_name, model_dir="tmp/modified_model", input=input)
    return new_output


def verify_output(output, new_output):
    np.testing.assert_allclose(output, new_output, atol=1.0e-5)


def modify_graph(graph: amanda.Graph):
    for op in graph.ops:
        for tensor in op.output_tensors:
            if not tensor.attrs["dtype"]._is_ref_dtype:
                op_name = tensor.op.attrs["name"]
                debug_op = amanda.create_op(
                    attrs={
                        "name": f"debug/{op_name}/{tensor.output_index}",
                        "type": "StoreTensorToFile",
                        "T": tensor.attrs["dtype"],
                    },
                    input_tensors=[tensor],
                    control_dependencies=[],
                    output_num=1,
                )

                for output_op in graph.ops:
                    for index, input_tensor in enumerate(output_op.input_tensors):
                        if tensor == input_tensor:
                            output_op.update_input_tensor(
                                index, debug_op.output_tensors[0]
                            )
                graph.add_op(debug_op)


class DebuggingTool(Tool):
    def instrument(self, graph: amanda.Graph) -> amanda.Graph:
        modify_graph(graph)
        return graph


def main():
    download_tf_model(arch_name, model_dir="model")
    output = run_original_model()

    graph = amanda.tensorflow.import_from_checkpoint(original_checkpoint_dir)
    modify_graph(graph)
    amanda.tensorflow.export_to_checkpoint(graph, modified_checkpoint_dir)

    new_output = run_modified_model()
    verify_output(output, new_output)


if __name__ == "__main__":
    main()
