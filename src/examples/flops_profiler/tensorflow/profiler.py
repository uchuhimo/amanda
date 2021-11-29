import amanda
import tensorflow as tf

from examples.flops_profiler.tensorflow.map import tf_map
from examples.flops_profiler.utils.node import Node


class Profiler(amanda.Tool):
    def __init__(self):
        super().__init__(namespace="tensorflow")
        self.add_inst_for_op(
            self.forward_instrumentation,
            require_outputs=True,
        )

    def forward_instrumentation(self, context: amanda.OpContext):
        op = context.get_op()
        node = Node(op.type)

        context.insert_before_op(
            self.record_before_op,
            inputs=[
                index
                for index, tensor in enumerate(context.get_inputs())
                if not tensor.dtype._is_ref_dtype
            ],
            node=node,
        )
        context.insert_after_op(
            self.record_after_op,
            outputs=[
                index
                for index, tensor in enumerate(context.get_outputs())
                if not tensor.dtype._is_ref_dtype
            ],
            node=node,
        )

    def record_before_op(self, *inputs, node):
        def extract_fn(*inputs):
            node.inputs = inputs
            return inputs

        new_inputs = tf.py_function(
            extract_fn,
            inputs,
            [tensor.dtype for tensor in inputs],
            name="before_" + node.name,
        )
        return new_inputs

    def record_after_op(self, *outputs, node):
        def extract_fn(*outputs):
            node.outputs = outputs
            for key, kernel in tf_map:
                if isinstance(key, str):
                    key = [key]
                if node.name in key:
                    if kernel:
                        print(node.name, kernel(node))
                    else:
                        print(node.name, 0)
                    return outputs
            print(node.name, "no impl.")
            return outputs

        new_outputs = tf.py_function(
            extract_fn,
            outputs,
            [tensor.dtype for tensor in outputs],
            name="after_" + node.name,
        )
        return new_outputs
