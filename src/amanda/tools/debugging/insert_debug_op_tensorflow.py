from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import load_library

import amanda
from amanda import create_op
from amanda.tests.test_tf_import_export import run_model as run_model_from
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


def run_original_model(input):
    output, _ = run_model_from(arch_name, model_dir="downloads/model", input=input)
    return output


def run_modified_model(input):
    new_output, _ = run_model_from(
        arch_name, model_dir="tmp/modified_model", input=input
    )
    return new_output


def verify_output(output, new_output):
    np.testing.assert_allclose(output, new_output, atol=1.0e-5)


def modify_graph(graph: amanda.Graph):
    for op in graph.ops:
        for output_port in op.output_ports:
            if not output_port.type.raw._is_ref_dtype:
                debug_op = create_op(
                    name=f"debug/{op.name}/{output_port.name}",
                    type="StoreTensorToFile",
                    attrs={
                        "store_dir": str(store_dir),
                        "file_name": f"{op.name}_{output_port.name}".replace("/", "_"),
                        "T": output_port.type.raw,
                    },
                )
                edges = output_port.out_edges
                graph.add_op(debug_op)
                graph.create_edge(output_port, debug_op.input_port(0))
                for edge in edges:
                    graph.create_edge(debug_op.output_port(0), edge.dst)
                    graph.remove_edge(edge)


class DebuggingTool(Tool):
    def instrument(self, graph: amanda.Graph) -> amanda.Graph:
        modify_graph(graph)
        return graph


@click.command()
@click.option(
    "--model-dir",
    "-m",
    required=True,
    type=click.Path(),
    help="Directory of the model.",
)
def run_model(model_dir):
    input = np.random.rand(1, 224, 224, 3)
    output, _ = run_model_from("", model_dir=model_dir, input=input)
    return output


def main():
    input = np.random.rand(1, 224, 224, 3)
    output = run_original_model(input)

    graph = amanda.tensorflow.import_from_checkpoint(original_checkpoint_dir)
    modify_graph(graph)
    amanda.tensorflow.export_to_checkpoint(graph, modified_checkpoint_dir)

    new_output = run_modified_model(input)
    verify_output(output, new_output)


if __name__ == "__main__":
    main()
