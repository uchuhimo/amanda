import amanda

from examples.flops_profiler.pytorch.map import pytorch_map
from examples.flops_profiler.utils.node import Node


class AmandaFlopsPytorch(amanda.Tool):
    def __init__(self):
        super().__init__(namespace="pytorch")
        # after_forward_op
        self.add_inst_for_op(self.forward_inst)

    def forward_inst(self, context: amanda.OpContext):
        op = context.get_op()

        node = Node(op.__name__)

        context.insert_before_op(
            self.get_inputs,
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

    def get_inputs(self, *inputs, node):
        node.inputs = inputs
