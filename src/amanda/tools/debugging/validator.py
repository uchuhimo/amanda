from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.script_ops import _py_funcs

import amanda
from amanda import Graph, OutputPort
from amanda.conversion.tensorflow import import_from_tf_func
from amanda.io.file import root_dir
from amanda.tests.test_tf_import_export import modify_model, run_model

arch_name = "vgg16"


def store_as_numpy(input: np.array, store_dir: str, file_name: str):
    if not Path(store_dir).exists():
        Path(store_dir).mkdir(mode=0o755, parents=True, exist_ok=True)
    file_name = file_name.replace("/", "_")
    np.save(f"{store_dir}/{file_name}", input)
    return input


def modify_graph(graph: Graph):
    store_dir = root_dir() / "tmp" / "validation_info" / arch_name
    for op in graph.ops:
        for output_port in op.output_ports:
            if not output_port.type.raw._is_ref_dtype:
                edges = output_port.out_edges
                full_name = f"{op.name}_{output_port.name}"
                func = partial(
                    store_as_numpy,
                    store_dir=store_dir,
                    file_name=full_name,
                )
                token = _py_funcs.insert(func)
                debug_op = amanda.create_op(
                    type="PyFunc",
                    name=full_name,
                    inputs=["input/0"],
                    outputs=["output/0"],
                    attrs=dict(
                        token=token,
                        Tin=[output_port.type.raw],
                        Tout=[output_port.type.raw],
                    ),
                )
                graph.add_op(debug_op)
                debug_output = debug_op.output_port(0)
                graph.create_edge(output_port, debug_op.input_port(0))
                for edge in edges:
                    graph.create_edge(debug_output, edge.dst)
                    graph.remove_edge(edge)


def modify_graph_with_tf_func(graph: Graph):
    store_dir = root_dir() / "tmp" / "validation_info" / arch_name
    for op in graph.ops:
        for output_port in op.output_ports:
            if not output_port.type.raw._is_ref_dtype:
                edges = output_port.out_edges
                full_name = f"{op.name}_{output_port.name}"
                func = partial(
                    store_as_numpy,
                    store_dir=store_dir,
                    file_name=full_name,
                )
                debug_output: OutputPort = import_from_tf_func(tf.py_func)(graph)(
                    func,
                    [output_port],
                    output_port.type.raw,
                )
                for edge in edges:
                    graph.create_edge(debug_output, edge.dst)
                    graph.remove_edge(edge)


def main(arch_name):
    input = np.random.rand(1, 224, 224, 3)
    output, _ = run_model(arch_name, model_dir="downloads/model", input=input)
    modify_model(arch_name, "modified_model_with_py_func", modify_graph)
    new_output, _ = run_model(
        arch_name, model_dir="tmp/modified_model_with_py_func", input=input
    )
    np.testing.assert_allclose(output, new_output, atol=1.0e-5)


if __name__ == "__main__":
    main("vgg16")
    # modify_model("facenet", "modified_graph", modify_graph_with_primitive_api)
    # modify_model("nasnet-a_large", "modified_graph", modify_graph_with_primitive_api)
