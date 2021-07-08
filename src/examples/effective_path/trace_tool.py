from utils import Graph, Op

import amanda


class TraceEffectivePathTool(amanda.Tool):
    def __init__(self):
        super(TraceEffectivePathTool, self).__init__(namespace="amanda/pytorch")
        self.register_event(amanda.event.after_op_executed, self.trace_forward_graph)

        self.graph = Graph()
        self.tensor_output_ops = dict()

    def trace_forward_graph(self, context):
        op = Op(
            raw_op=context["op"],
            input_ops=[
                self.tensor_output_ops[i]
                for i in self.unpack_inputs(context["args"])
                if i.__hash__ and i in self.tensor_output_ops
            ],
            inputs=context["args"],
            outputs=context["output"],
        )
        print(f"forward tracing: {op.raw_op.__name__}")
        self.graph.ops.append(op)
        self.tensor_output_ops[op.outputs] = op

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
