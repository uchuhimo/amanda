import amanda


class DummyTool(amanda.Tool):
    def __init__(self):
        super(DummyTool, self).__init__(namespace="amanda/tensorflow")
        self.register_event(
            amanda.event.before_op_executed,
            self.test_before_op_executed
        )
        self.register_event(
            amanda.event.after_backward_op_executed,
            self.test_after_backward_op_executed
        )

    def test_before_op_executed(self, context: amanda.EventContext):
        return

    def test_after_backward_op_executed(self, context: amanda.EventContext):
        return
