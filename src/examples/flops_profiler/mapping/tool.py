import amanda
from amanda.tools.mapping import MappingTool

import examples.flops_profiler.mapping.torch_rule as torch_rule
from examples.flops_profiler.pytorch.map import pytorch_map
from examples.flops_profiler.utils.node import Node


class FlopsProfileTool(amanda.Tool):
    def __init__(self):
        super().__init__(namespace="FlopsProfiler")

        self.depends_on(
            MappingTool(
                rules=[
                    ["pytorch", torch_rule.torch_get_shape],
                    ["pytorch", torch_rule.torch_type],
                ]
            )
        )
        self.add_inst_for_op(self.forward_instrumentation)

    def forward_instrumentation(self, context: amanda.OpContext):

        node = Node(context["type"])

        context.insert_before_op(
            self.profile_inputs,
            node=node,
        )

        context.insert_after_op(
            self.calculate_flops,
            node=node,
        )

    def calculate_flops(self, *outputs, node):
        node.outputs = outputs
        for key, kernel in pytorch_map:
            if isinstance(key, str):
                key = [key]
            if node.name in key:
                if kernel:
                    print(node.name, kernel(node))
                else:
                    print(node.name, 0)
                return
        print(node.name, "no impl.")

    def profile_inputs(self, *inputs, node):
        node.inputs = inputs
