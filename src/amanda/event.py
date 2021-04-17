from dataclasses import dataclass
from typing import Callable, Iterable
from enum import Enum, auto
import logging

@dataclass(frozen=True)
class Event:
    name: str


on_graph_loaded = Event("on_graph_loaded")
before_graph_executed = Event("before_graph_executed")
after_graph_executed = Event("after_graph_executed")
after_backward_graph_executed = Event("after_backward_graph_executed")
before_subgraph_executed = Event("before_subgraph_executed")
after_subgraph_executed = Event("after_subgraph_executed")
after_backward_subgraph_executed = Event("after_backward_subgraph_executed")
before_op_executed = Event("before_op_executed")
after_op_executed = Event("after_op_executed")
before_backward_op_executed = Event("before_backward_op_executed")
after_backward_op_executed = Event("after_backward_op_executed")

EventCallback = Callable[["EventContext"], None]

class UpdateStatus(Enum):
    No_Grad = auto()
    Not_Updated = auto()
    Is_Updated = auto()

class GradStatue():
    def __init__(self, event: Event, status: UpdateStatus) -> None:
        self.event = event
        self.status = status


class EventContext(dict):
    from amanda.tool import Tool

    def __init__(self, tools: Iterable[Tool]):
        super(EventContext, self).__init__()
        from amanda.tool import Tool

        self.tools: Iterable[Tool] = tools

    def trigger(self, event: Event, **kwargs) -> None:
        self.update(**kwargs)
        for tool in self.tools:
            if tool and tool.is_registered(event):
                tool.get_callback(event)(self)

    def is_registered(self, event: Event) -> bool:
        for tool in self.tools:
            if tool.is_registered(event):
                return True
        return False

    '''EventContext.registry_bw_events()
    registry the backward hooks for before/after backward op events.
    before_bw_op_event is hooked on output tensor.
    after_bw_op_event is hooked on backward function of output tensor.
        it is the output.grad_fn.
    note that this method is coupled with pytorch and need to be refactored
    '''
    def registry_bw_events(self, output):
        '''
        hook wrappers triggers the event with context and remove the hook.
        note that is a hook is not triggered, then it will not be collected!
        this may caused by ops only existed only in forward phase.
        '''
        def before_bw_op_hook(context, output):
            context.trigger(before_backward_op_executed, output_grad=output)
            before_bw_op_hook_handle.remove()
        
        def after_bw_op_hook(context, input, output):
            context.trigger(after_backward_op_executed, input_grad=input, output_grad=output)
            after_bw_op_hook_handle.remove()

        if hasattr(output, 'register_hook') and output.requires_grad:
            before_bw_op_hook_handle = output.register_hook(lambda output: before_bw_op_hook(self, output))

        if hasattr(output, 'grad_fn') and output.grad_fn:
            after_bw_op_hook_handle = output.grad_fn.register_hook(lambda input,output: after_bw_op_hook(self, input, output))
            