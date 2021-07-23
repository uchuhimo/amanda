import amanda


class ErrorInjectionTool(amanda.Tool):
    def __init__(self, filter_fn, modify_fn):
        super(ErrorInjectionTool, self).__init__(namespace="amanda/pytorch")

        self.add_inst_for_op(self.forward_injection, require_outputs=False)

        self.injection_filter = filter_fn
        self.injection_flipper = modify_fn

    def forward_injection(self, context):
        op_name = context["op"].__name__

        if self.injection_filter(op_name):
            print(f"injecting op: {op_name}")
            context.insert_before_op(
                self.error_injection_op,
                inputs=[0],
            )
        else:
            print(f"skipping op: {op_name}")

    def error_injection_op(self, feat_map):
        return self.injection_flipper(feat_map)
