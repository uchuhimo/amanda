import tensorflow as tf

from amanda import Op
from amanda.conversion.tensorflow import export_to_checkpoint, import_from_checkpoint
from amanda.tests.utils import root_dir

arch_name = "vgg16"
prefix_dir = root_dir() / "tmp"
original_checkpoint_dir = tf.train.latest_checkpoint(prefix_dir / "model" / arch_name)
modified_checkpoint_dir = prefix_dir / "modified_model" / arch_name / arch_name
store_dir = root_dir() / "tmp" / "debug_info" / arch_name


def modify_graph(graph):
    for op in graph.ops:
        for tensor in op.output_tensors:
            debug_op = Op(
                attrs={
                    "name": "debug" + op.attrs["name"],
                    "type": "StoreTensorToFile",
                    "store_dir": store_dir,
                    "file_name": op.attrs["name"] + ":" + str(tensor.output_index),
                },
                input_tensors=[tensor],
                control_dependencies=[],
                output_num=1,
            )
            graph.add_op(debug_op)
            for output_op in graph.ops:
                for index, input_tensor in enumerate(output_op.input_tensors):
                    if tensor == input_tensor:
                        output_op.update_input_tensor(index, debug_op.output_tensors[0])


def main():
    graph = import_from_checkpoint(original_checkpoint_dir)
    modify_graph(graph)
    export_to_checkpoint(graph, modified_checkpoint_dir)


if __name__ == "__main__":
    main()
