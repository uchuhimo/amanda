import amanda

from examples.effective_path.graph import Graph, Op


class TraceEffectivePathTool(amanda.Tool):
    def __init__(self):
        super(TraceEffectivePathTool, self).__init__(namespace="amanda/pytorch")
        self.add_inst_for_op(self.trace_forward_graph)

        self.graph = Graph()

    def trace_forward_graph(self, context):
        op = Op(raw_op=context.get_op(), id=context.get_op_id())
        print(f"forward tracing: {op.raw_op.__name__}")
        self.graph.ops[context.get_op_id()] = op
        context.insert_before_op(
            self.extract_inputs,
            op=op,
        )
        context.insert_after_op(
            self.extract_outputs,
            op=op,
        )

    def extract_inputs(self, *inputs, op):
        op.inputs = inputs

    def extract_outputs(self, *outputs, op):
        op.outputs = outputs

    def unpack_inputs(self, inputs):
        def _unpack_inputs(inputs):
            for input in inputs:
                if type(input).__name__ == "Tensor":
                    all_inputs.append(input)
                elif isinstance(input, list) or isinstance(input, set):
                    _unpack_inputs(input)

        all_inputs = []
        _unpack_inputs(inputs)
        return all_inputs
