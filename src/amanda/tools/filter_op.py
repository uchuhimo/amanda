from typing import Callable

from amanda.event import (
    Event,
    EventContext,
    after_backward_op_added,
    after_op_added,
    before_backward_op_added,
    before_op_added,
)
from amanda.tool import Tool


class FilterOpTool(Tool):
    def __init__(self, filter_fn: Callable[[Event, EventContext], bool]):
        super().__init__()
        self.register_event(before_op_added, self.before_op_added)
        self.register_event(after_op_added, self.after_op_added)
        self.register_event(before_backward_op_added, self.before_backward_op_added)
        self.register_event(after_backward_op_added, self.after_backward_op_added)
        self.filter_fn = filter_fn

    def before_op_added(self, context: EventContext):
        if self.filter_fn(before_op_added, context):
            context["hook_before_op_added"] = True
        else:
            if "hook_before_op_added" not in context:
                context["hook_before_op_added"] = False

    def after_op_added(self, context: EventContext):
        if self.filter_fn(after_op_added, context):
            context["hook_after_op_added"] = True
        else:
            if "hook_after_op_added" not in context:
                context["hook_after_op_added"] = False

    def before_backward_op_added(self, context: EventContext):
        if self.filter_fn(before_backward_op_added, context):
            context["hook_before_backward_op_added"] = True
        else:
            if "hook_before_backward_op_added" not in context:
                context["hook_before_backward_op_added"] = False

    def after_backward_op_added(self, context: EventContext):
        if self.filter_fn(after_backward_op_added, context):
            context["hook_after_backward_op_added"] = True
        else:
            if "hook_after_backward_op_added" not in context:
                context["hook_after_backward_op_added"] = False
