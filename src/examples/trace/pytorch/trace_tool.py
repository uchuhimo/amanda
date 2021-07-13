import torch

import amanda


class TraceTool(amanda.Tool):
    def __init__(self):
        super(TraceTool, self).__init__(namespace="amanda/pytorch")
        self.add_inst_for_op(self.forward_instrumentation)
        # self.add_inst_for_backward_op(
        #     self.backward_instrumentation,
        #     require_grad_inputs=True,
        # )

    def forward_instrumentation(self, context: amanda.OpContext):
        print(context["op"].__name__)
