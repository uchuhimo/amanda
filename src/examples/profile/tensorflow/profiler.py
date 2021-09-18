import json
import os
import time

import amanda
import tensorflow as tf
from amanda.io.file import ensure_dir

from examples.profile.tensorflow.utils import from_attr_proto


class Profiler(amanda.Tool):
    def __init__(self):
        super().__init__(namespace="tensorflow")
        self.add_inst_for_op(
            self.forward_instrumentation,
            require_outputs=True,
        )
        self.add_inst_for_op(
            self.backward_instrumentation,
            backward=True,
            require_outputs=True,
        )
        self.start_time = None
        self.trace = []

    def export_chrome_trace(self, path):
        with open(ensure_dir(path), "w") as file:
            json.dump(self.trace, file)

    def forward_instrumentation(self, context: amanda.OpContext):
        op = context.get_op()
        if op.type == "Identity":
            return
        event = {
            "name": op.type,
            "args": {
                key: from_attr_proto(value) for key, value in op.node_def.attr.items()
            },
        }
        event["args"]["id"] = op.name
        context.insert_before_op(
            self.record_before_op,
            inputs=[
                index
                for index, tensor in enumerate(context.get_inputs())
                if not tensor.dtype._is_ref_dtype
            ],
            event=event,
            op=op,
        )
        context.insert_after_op(
            self.record_after_op,
            outputs=[
                index
                for index, tensor in enumerate(context.get_outputs())
                if not tensor.dtype._is_ref_dtype
            ],
            event=event,
            op=op,
        )

    def backward_instrumentation(self, context: amanda.OpContext):
        bw_op = context.get_backward_op()
        event = {
            "name": bw_op.type,
            "args": {
                key: from_attr_proto(value)
                for key, value in bw_op.node_def.attr.items()
            },
        }
        event["args"]["id"] = bw_op.name
        context.insert_before_backward_op(
            self.record_before_op,
            grad_outputs=[
                index
                for index, tensor in enumerate(context.get_grad_outputs())
                if not tensor.dtype._is_ref_dtype
            ],
            event=event,
            op=bw_op,
        )
        context.insert_after_backward_op(
            self.record_after_op,
            grad_inputs=[
                index
                for index, tensor in enumerate(context.get_grad_inputs())
                if not tensor.dtype._is_ref_dtype
            ],
            event=event,
            op=bw_op,
        )

    def record_before_op(self, *inputs, event, op):
        def extract_fn(*inputs):
            if self.start_time is None:
                self.start_time = time.perf_counter() * 1000
            event["pid"] = os.getpid()
            event["ph"] = "X"
            event["ts"] = time.perf_counter() * 1000 - self.start_time
            return inputs

        new_inputs = tf.py_function(
            extract_fn,
            inputs,
            [tensor.dtype for tensor in inputs],
            name="before_" + op.name,
        )
        return new_inputs

    def record_after_op(self, *outputs, event, op):
        def extract_fn(*outputs):
            if "ts" in event:
                event["dur"] = (
                    time.perf_counter() * 1000 - event["ts"] - self.start_time
                )
                self.trace.append(event)
            return outputs

        with tf.control_dependencies([op]):
            new_outputs = tf.py_function(
                extract_fn,
                outputs,
                [tensor.dtype for tensor in outputs],
                name="after_" + op.name,
            )
        return new_outputs
