from functools import partial

import numpy as np
import tensorflow as tf

from amanda import Graph, Tensor
from amanda.conversion.tensorflow import (
    export_to_checkpoint,
    get_dtype,
    import_from_checkpoint,
    import_from_tf_func,
)
from amanda.tests.test_tf_import_export import run_model
from amanda.tests.utils import root_dir

arch_name = "vgg16"


def store_as_numpy(input: np.array, store_dir, file_name):
    np.save(f"{store_dir}/{file_name}", input)
    return input


def modify_graph(graph: Graph):
    store_dir = root_dir() / "tmp" / "debug_info" / arch_name
    original_graph = graph.clone()
    for op in original_graph.ops:
        for tensor in op.output_tensors:
            output_edges = original_graph.edges_from_tensor(tensor)
            debug_output: Tensor = import_from_tf_func(tf.py_func)(graph)(
                partial(
                    store_as_numpy,
                    store_dir=store_dir,
                    file_name=f"{op.name}:{tensor.output_index}",
                ),
                [tensor],
                get_dtype(tensor),
            )
            # debug_output = convert_from_tf_func(tf.identity, graph)(tensor)
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


def main(arch_name):
    input = np.random.rand(1, 224, 224, 3)
    output, _ = run_model(arch_name, model_dir="model", input=input)
    modify_model(arch_name)
    new_output, _ = run_model(arch_name, model_dir="modified_model", input=input)
    assert np.allclose(output, new_output)


if __name__ == "__main__":
    main("vgg16")
