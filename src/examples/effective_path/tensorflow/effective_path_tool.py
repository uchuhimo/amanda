import tensorflow as tf

import amanda
from examples.effective_path.graph import Graph, Op


class EffectivePathTool(amanda.Tool):
    def __init__(self):
        super().__init__(namespace="amanda/tensorflow")
        self.add_inst_for_op(self.forward_instrumentation, require_outputs=True)
        self.graph = Graph()
        self.raw_to_op = {}

    def forward_instrumentation(self, context: amanda.OpContext):
        raw_op = context.get_op()
        input_index = []
        inputs = []
        for index, input in enumerate(raw_op.inputs):
            if not input.dtype._is_ref_dtype:
                input_index.append(index)
                inputs.append(input)
        output_index = [
            index
            for index, output in enumerate(raw_op.outputs)
            if not output.dtype._is_ref_dtype
        ]
        op = Op(
            raw_op=raw_op,
            input_ops=[self.raw_to_op[input.op] for input in inputs],
        )
        self.graph.ops.append(op)
        self.raw_to_op[raw_op] = op
        context.insert_before_op(self.extract_inputs, inputs=input_index, op=op)
        context.insert_after_op(self.extract_outputs, outputs=output_index, op=op)

    def extract_inputs(self, *inputs, op):
        def extract_fn(*np_inputs):
            op.inputs = np_inputs
            return np_inputs

        if len(inputs) == 0:
            return
        new_inputs = tf.numpy_function(
            extract_fn,
            inputs,
            [tensor.dtype for tensor in inputs],
        )
        return new_inputs

    def extract_outputs(self, *outputs, op):
        def extract_fn(*np_outputs):
            op.outputs = np_outputs
            return np_outputs

        if len(outputs) == 0:
            return
        new_outputs = tf.numpy_function(
            extract_fn,
            outputs,
            [tensor.dtype for tensor in outputs],
        )
        return new_outputs
