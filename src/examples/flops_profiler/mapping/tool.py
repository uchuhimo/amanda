import amanda
import tensorflow as tf
from amanda.tools.mapping import MappingTool

import examples.flops_profiler.mapping.tf_rule as tf_rule
import examples.flops_profiler.mapping.torch_rule as torch_rule
from examples.flops_profiler.pytorch.map import pytorch_map
from examples.flops_profiler.utils.node import Node


class FlopsProfileTool(amanda.Tool):
    def __init__(self, namespace="FlopsProfiler"):
        super().__init__(namespace=namespace)

        self.depends_on(
            MappingTool(
                rules=[
                    ["pytorch", torch_rule.torch_get_shape],
                    ["pytorch", torch_rule.torch_type],
                    ["tensorflow", tf_rule.tf_get_shape],
                    ["tensorflow", tf_rule.tf_type],
                ]
            )
        )
        if namespace == "pytorch":
            self.add_inst_for_op(self.forward_instrumentation)
        if namespace == "tensorflow":
            self.add_inst_for_op(self.forward_instrumentation, require_outputs=True)

    def forward_instrumentation(self, context: amanda.OpContext):

        node = Node(context["type"])

        if context.namespace == "pytorch":
            context.insert_before_op(
                self.torch_profile_inputs,
                node=node,
            )
            context.insert_after_op(
                self.torch_calculate_flops,
                node=node,
            )

        if context.namespace == "tensorflow":
            if context.get_inputs():
                context.insert_before_op(
                    self.tf_profile_inputs,
                    inputs=[
                        index
                        for index, tensor in enumerate(context.get_inputs())
                        if not tensor.dtype._is_ref_dtype
                    ],
                    node=node,
                )
            if context.get_outputs():
                context.insert_after_op(
                    self.tf_calculate_flops,
                    outputs=[
                        index
                        for index, tensor in enumerate(context.get_outputs())
                        if not tensor.dtype._is_ref_dtype
                    ],
                    node=node,
                )

    def tf_calculate_flops(self, *outputs, node):
        def op(*outputs):
            node.outputs = outputs
            cal_node_flops(node, outputs)
            return outputs

        new_outputs = tf.py_function(
            op,
            outputs,
            [tensor.dtype for tensor in outputs],
            name="after_" + node.name,
        )
        return new_outputs

    def torch_calculate_flops(self, *outputs, node):
        cal_node_flops(node, outputs)

    def tf_profile_inputs(self, *inputs, node):
        def op(*inputs):
            node.inputs = inputs
            return inputs

        new_inputs = tf.py_function(
            op,
            inputs,
            [tensor.dtype for tensor in inputs],
            name="before_" + node.name,
        )
        return new_inputs

    def torch_profile_inputs(self, *inputs, node):
        node.inputs = inputs


def cal_node_flops(node, outputs):
    node.outputs = outputs
    for key, kernel in pytorch_map:
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
