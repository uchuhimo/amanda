import amanda


class ErrorInjectionTool(amanda.Tool):
    def __init__(self, filter_fn, modify_fn):
        super(ErrorInjectionTool, self).__init__(namespace="amanda/pytorch")

        self.add_inst_for_op(self.forward_injection, require_outputs=True)

        self.injection_filter = filter_fn
        self.injection_flipper = modify_fn

    def forward_injection(self, context):
        op_name = context["op"].__name__

        if self.injection_filter(op_name):
            print(f"injecting op: {op_name}")
            context["inputs"][0].data = self.injection_flipper(context["inputs"][0])
        else:
            print(f"skipping op: {op_name}")
