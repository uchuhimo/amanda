import amanda


class ErrorInjectionTool(amanda.Tool):
    def __init__(self, filter_fn, modify_fn):
        super(ErrorInjectionTool, self).__init__(namespace="amanda/pytorch")
        self.register_event(amanda.event.before_op_executed, self.forward_injection)

        self.register_event(amanda.event.after_op_executed, self.test_after_hook)

        self.injection_filter = filter_fn
        self.injection_flipper = modify_fn

    def forward_injection(self, context):
        op_name = context["op"].__name__

        if self.injection_filter(op_name):
            print(f"injecting op: {op_name}")
            context["args"][0].data = self.injection_flipper(context["args"][0])
        else:
            print(f"skipping op: {op_name}")

    def test_after_hook(self, context):
        print(context["output"])
