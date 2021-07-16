import amanda
from amanda.io.file import ensure_dir


class TraceTool(amanda.Tool):
    def __init__(self, file_name="tmp/trace_resnet50/tracetool.txt"):
        super(TraceTool, self).__init__(namespace="amanda/pytorch")
        self.add_inst_for_op(self.forward_instrumentation)
        self.add_inst_for_backward_op(
            self.backward_instrumentation,
            require_grad_inputs=True,
        )

        self.output_file = open(ensure_dir(file_name), "w")

    def forward_instrumentation(self, context: amanda.OpContext):
        self.output_file.write(f"fw: {context['op'].__name__}\n")

    def backward_instrumentation(self, context: amanda.OpContext):
        self.output_file.write(
            f"fw: {context['op'].__name__}, bw: {context['backward_op'].__name__}\n"
        )
