import json
import os
import time

import amanda
from amanda.io.file import ensure_dir


class Profiler(amanda.Tool):
    def __init__(self):
        super().__init__(namespace="pytorch")
        self.add_inst_for_op(self.forward_instrumentation)
        self.add_inst_for_op(
            self.backward_instrumentation,
            backward=True,
        )
        self.start_time = None
        self.trace = []

    def export_chrome_trace(self, path):
        with open(ensure_dir(path), "w") as file:
            json.dump(self.trace, file)

    def forward_instrumentation(self, context: amanda.OpContext):
        op = context.get_op()
        event = {"name": op.__name__}
        context.insert_before_op(
            self.record_before_op,
            event=event,
        )
        context.insert_after_op(
            self.record_after_op,
            event=event,
        )

    def backward_instrumentation(self, context: amanda.OpContext):
        bw_op = context.get_backward_op()
        event = {"name": bw_op.__name__}
        context.insert_before_backward_op(
            self.record_before_op,
            event=event,
        )
        context.insert_after_backward_op(
            self.record_after_op,
            event=event,
        )

    def record_before_op(self, *inputs, event):
        if self.start_time is None:
            self.start_time = time.perf_counter() * 1000
        event["pid"] = os.getpid()
        event["ph"] = "X"
        event["ts"] = time.perf_counter() * 1000 - self.start_time

    def record_after_op(self, *outputs, event):
        event["dur"] = time.perf_counter() * 1000 - event["ts"] - self.start_time
        self.trace.append(event)
