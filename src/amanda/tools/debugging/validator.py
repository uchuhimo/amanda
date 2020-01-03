from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf

from amanda import Graph, Tensor
from amanda.conversion.tensorflow import get_dtype, import_from_tf_func
from amanda.tests.test_tf_import_export import modify_model, run_model
from amanda.tests.utils import root_dir

arch_name = "vgg16"


def store_as_numpy(input: np.array, store_dir: str, file_name: str):
    if not Path(store_dir).exists():
        Path(store_dir).mkdir(mode=0o755, parents=True, exist_ok=True)
    file_name = file_name.replace("/", "_")
    np.save(f"{store_dir}/{file_name}", input)
    return input


def modify_graph(graph: Graph):
    store_dir = root_dir() / "tmp" / "debug_info" / arch_name
    original_graph = graph.copy()
    for op in original_graph.ops:
        for tensor in op.output_tensors:
            output_edges = original_graph.edges_from_tensor(tensor)
            debug_output: Tensor = import_from_tf_func(tf.py_func)(graph)(
                partial(
                    store_as_numpy,
                    store_dir=store_dir,
                    file_name=f"{op.name}_{tensor.output_index}",
                ),
                [tensor],
                get_dtype(tensor),
            )
            for edge in output_edges:
                if edge.dst_op.type != "Assign":
                    edge.dst_op.input_tensors[edge.dst_input_index] = debug_output


def main(arch_name):
    input = np.random.rand(1, 224, 224, 3)
    output, _ = run_model(arch_name, model_dir="model", input=input)
    modify_model(arch_name, "modified_model_with_py_func", modify_graph)
    new_output, _ = run_model(
        arch_name, model_dir="modified_model_with_py_func", input=input
    )
    assert np.allclose(output, new_output, atol=1.0e-5)


if __name__ == "__main__":
    main("vgg16")
    # test_tf_modify_graph(arch_name="facenet")
