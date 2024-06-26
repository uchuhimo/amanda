import amanda
from amanda.io.file import ensure_dir


class TraceTool(amanda.Tool):
    def __init__(self, output_dir="tmp"):
        super(TraceTool, self).__init__(namespace="amanda/tensorflow")
        self.add_inst_for_op(self.forward_instrumentation)
        self.add_inst_for_op(
            self.backward_instrumentation,
            backward=True,
            require_outputs=False,
        )

        self.output_file = open(ensure_dir(output_dir), "w")

    def forward_instrumentation(self, context: amanda.OpContext):
        op = context.get_op()

        context.insert_before_op(self.dump_forward_op, op_name=op.name)

    def backward_instrumentation(self, context: amanda.OpContext):
        fw_op = context.get_op()
        bw_op = context.get_backward_op()
        context.insert_before_backward_op(
            self.dump_backward_op,
            fw_op_name=fw_op.name if fw_op is not None else None,
            bw_op_name=bw_op.name,
        )

    def dump_forward_op(self, *inputs, op_name):
        self.output_file.write(f"fw_op: {op_name}\n")

    def dump_backward_op(self, *inputs, fw_op_name, bw_op_name):
        self.output_file.write(f"bw: {bw_op_name}, fw: {fw_op_name}\n")
