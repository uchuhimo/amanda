import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import load_library

import amanda
from amanda.io.file import root_dir
from amanda.tests.test_tf_import_export import run_model
from amanda.tools.debugging.insert_debug_op_adhoc import modify_graph

store_tensor_to_file_ops = load_library.load_op_library(
    os.path.dirname(os.path.abspath(__file__)) + "/store_tensor_to_file_op.so"
)
store_tensor_to_file = store_tensor_to_file_ops.store_tensor_to_file

arch_name = "vgg16"
original_checkpoint_dir = tf.train.latest_checkpoint(
    root_dir() / "downloads" / "model" / arch_name
)
assert original_checkpoint_dir is not None
modified_checkpoint_dir = root_dir() / "tmp" / "modified_model" / arch_name / arch_name


def run_original_model(input):
    output, _ = run_model(arch_name, model_dir="downloads/model", input=input)
    return output


def run_modified_model(input):
    new_output, _ = run_model(arch_name, model_dir="tmp/modified_model", input=input)
    return new_output


def verify_output(output, new_output):
    np.testing.assert_allclose(output, new_output, atol=1.0e-5)


def main():
    input = np.random.rand(1, 224, 224, 3)
    output = run_original_model(input)

    graph = amanda.tensorflow.import_from_checkpoint(original_checkpoint_dir)
    new_graph = modify_graph(graph)
    amanda.tensorflow.export_to_checkpoint(new_graph, modified_checkpoint_dir)

    new_output = run_modified_model(input)
    verify_output(output, new_output)


if __name__ == "__main__":
    main()
