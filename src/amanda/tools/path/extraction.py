# type: ignore
import pickle

import numpy as np
import tensorflow as tf

import amanda
from amanda.tests.utils import root_dir
from amanda.tools.path.instrumentation import modify_graph

arch_name = "vgg16"

inputs = [np.random.rand(1, 224, 224, 3) for _ in range(1000)]


def insert_path_extraction_code(arch_name, output_model_dir):
    original_checkpoint = tf.train.latest_checkpoint(
        root_dir() / "downloads" / "model" / arch_name
    )
    graph = amanda.tensorflow.import_from_checkpoint(original_checkpoint)
    path_op_names = modify_graph(graph)
    modified_checkpoint = root_dir() / "tmp" / output_model_dir / arch_name / arch_name
    amanda.tensorflow.export_to_checkpoint(graph, modified_checkpoint)
    return [op_name + ":0" for op_name in path_op_names]


def extract_paths(arch_name, path_tensor_names, model_dir, input):
    checkpoint_dir = root_dir() / "tmp" / model_dir / arch_name
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(checkpoint_file + ".meta")
            saver.restore(sess, checkpoint_file)
            for input in inputs:
                paths = sess.run(path_tensor_names, {"input:0": input})
            return paths


def main(arch_name):
    path_tensor_names = insert_path_extraction_code(
        arch_name, output_model_dir="model_with_path_extraction"
    )
    input = np.random.rand(1, 224, 224, 3)
    paths = extract_paths(
        arch_name,
        path_tensor_names,
        model_dir="model_with_path_extraction",
        input=input,
    )
    pickle.dump(paths, root_dir() / "effective_paths" / arch_name)


if __name__ == "__main__":
    main("vgg16")
