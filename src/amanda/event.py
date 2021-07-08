from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterable, List, Set


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
before_op_added = Event("before_op_added")
after_op_added = Event("after_op_added")
before_backward_op_added = Event("before_backward_op_added")
after_backward_op_added = Event("after_backward_op_added")
after_graph_constructed = Event("after_graph_constructed")
on_op_call = Event("on_op_call")
on_backward_op_call = Event("on_backward_op_call")
before_forward = Event("before_forward")
after_forward = Event("before_forward")
before_backward = Event("before_backward")
after_backward = Event("after_backward")

EventCallback = Callable[["EventContext"], None]
OpCallback = Callable[["OpContext"], None]


class UpdateStatus(Enum):
    No_Grad = auto()
    Not_Updated = auto()
    Is_Updated = auto()


class GradStatue:
    def __init__(self, event: Event, status: UpdateStatus) -> None:
        self.event = event
        self.status = status


@dataclass(frozen=True)
class Action:
    type: str
    func: Callable[..., Any]
    kwargs: Dict[str, Any]
    inputs: List[int] = None
    outputs: List[int] = None


class OpContext(dict):
    from amanda.tool import Tool

    def __init__(self, tools: Iterable[Tool]):
        super().__init__()
        from amanda.tool import Tool

        self.tools: Iterable[Tool] = tools
        self.actions: List[Action] = []

    def trigger(self, event: Event, **kwargs) -> None:
        self.update(**kwargs)
        triggered_tools: Set[str] = set()
        for tool in self.tools:
            if tool and tool.is_registered(event):
                tool.trigger(event, self, triggered_tools)

    def is_registered(self, event: Event) -> bool:
        for tool in self.tools:
            if tool.is_registered(event):
                return True
        return False

    @property
    def op(self):
        return self.get_op()

    @property
    def inputs(self):
        return self.get_inputs()

    @property
    def backward_op(self):
        return self.get_backward_op()

    @property
    def grad_outputs(self):
        return self.get_grad_outputs()

    def get_op(self):
        return self["op"]

    def get_inputs(self):
        return self["inputs"]

    def get_backward_op(self):
        return self["backward_op"]

    def get_grad_outputs(self):
        return self["grad_outputs"]

    def insert_before_op(self, func, inputs: List[int] = None, **kwargs):
        self.actions.append(
            Action(
                type="insert_before_op",
                func=func,
                inputs=inputs,
                kwargs=kwargs,
            )
        )

    def insert_after_op(self, func, outputs: List[int] = None, **kwargs):
        self.actions.append(
            Action(
                type="insert_after_op",
                func=func,
                outputs=outputs,
                kwargs=kwargs,
            )
        )

    def insert_before_backward_op(self, func, grad_outputs: List[int] = None, **kwargs):
        self.actions.append(
            Action(
                type="insert_before_op",
                func=func,
                inputs=grad_outputs,
                kwargs=kwargs,
            )
        )

    def insert_after_backward_op(self, func, grad_inputs: List[int] = None, **kwargs):
        self.actions.append(
            Action(
                type="insert_after_op",
                func=func,
                outputs=grad_inputs,
                kwargs=kwargs,
            )
        )

    def replace_op(self, func, inputs: List[int] = None, **kwargs):
        self.actions.append(
            Action(
                type="replace_op",
                func=func,
                inputs=inputs,
                kwargs=kwargs,
            )
        )

    def replace_backward_op(self, func, grad_outputs: List[int] = None, **kwargs):
        self.actions.append(
            Action(
                type="replace_op",
                func=func,
                inputs=grad_outputs,
                kwargs=kwargs,
            )
        )


class EventContext(dict):
    from amanda.tool import Tool

    def __init__(self, tools: Iterable[Tool]):
        super(EventContext, self).__init__()
        from amanda.tool import Tool

        self.tools: Iterable[Tool] = tools

    def trigger(self, event: Event, **kwargs) -> None:
        self.update(**kwargs)
        triggered_tools: Set[str] = set()
        for tool in self.tools:
            if tool and tool.is_registered(event):
                tool.trigger(event, self, triggered_tools)

    def is_registered(self, event: Event) -> bool:
        for tool in self.tools:
            if tool.is_registered(event):
                return True
        return False

    """EventContext.registry_bw_events()
    registry the backward hooks for before/after backward op events.
    before_bw_op_event is hooked on output tensor.
    after_bw_op_event is hooked on backward function of output tensor.
        it is the output.grad_fn.
    note that this method is coupled with pytorch and need to be refactored
    """

    def registry_bw_events(self, output):
        """
        hook wrapper= triggers the event with context and remove the hook.
        note that if a hook is not triggered, then it will not be collected!
        this may caused by ops only existed only in forward phase.
        """

        def before_bw_op_hook(context, output):
            context.trigger(before_backward_op_executed, output_grad=output)
            before_bw_op_hook_handle.remove()

        def after_bw_op_hook(context, input, output):
            context.trigger(
                after_backward_op_executed, input_grad=input, output_grad=output
            )
            after_bw_op_hook_handle.remove()

        if hasattr(output, "register_hook") and output.requires_grad:
            before_bw_op_hook_handle = output.register_hook(
                lambda output: before_bw_op_hook(self, output)
            )

        if hasattr(output, "grad_fn"):  # skip ops with non-tensor output
            if (
                output.grad_fn.__class__.__name__ == "UnsafeViewBackward"
            ):  # check for broadcast case
                """check for broadcast case
                not considering broadcast of input tensors now
                """
                # print(f"add backward with broadcast")
                if output.grad_fn.next_functions[0][0]:
                    after_bw_op_hook_handle = output.grad_fn.next_functions[0][
                        0
                    ].register_hook(
                        lambda input, output: after_bw_op_hook(
                            self, input, self["output_grad"]
                        )
                    )
            elif output.grad_fn.__class__.__name__ == "AddBackward0":  # check for bias
                # print(f"add backward with bias founded")
                if output.grad_fn.next_functions[0][0]:
                    after_bw_op_hook_handle = output.grad_fn.next_functions[0][
                        0
                    ].register_hook(
                        lambda input, output: after_bw_op_hook(
                            self, input, self["output_grad"]
                        )
                    )
            else:
                if output.grad_fn:
                    after_bw_op_hook_handle = output.grad_fn.register_hook(
                        lambda input, output: after_bw_op_hook(self, input, output)
                    )
        else:
            pass

    """EventContext.register_bw_events_recursively()
    same functionality as register_bw_events() with subgraph matching,
    in this manner, a EventContext in bw phase have "op", "bw_op" two context,
    either of them may be None or not exists, denoting only exists in fw or bw,
    """

    def register_bw_events_recursively(self, output, input_grad_fns):
        def before_bw_op_hook(context, bw_op, output):
            context.trigger(
                before_backward_op_executed, bw_op=bw_op, output_grad=output
            )
            before_bw_op_hook_handle.remove()

        def _register_bw_events(grad_fn):
            def after_bw_op_hook(context, bw_op, input, output):
                context.trigger(
                    after_backward_op_executed,
                    bw_op=bw_op,
                    input_grad=input,
                    output_grad=output,
                )
                after_bw_op_hook_handle.remove()

            if grad_fn and grad_fn not in input_grad_fns:
                # print(f"registering after event for: {grad_fn}")
                after_bw_op_hook_handle = grad_fn.register_hook(
                    lambda input, output: after_bw_op_hook(self, grad_fn, input, output)
                )
            else:
                return
            for next_grad_fn, next_input_pos in grad_fn.next_functions:
                # including AccumulateGrad in backward graph
                if next_grad_fn and next_grad_fn not in input_grad_fns:
                    # if next_grad_fn and
                    #     next_grad_fn not in input_grad_fns
                    #     and type(next_grad_fn).__name__ != "AccumulateGrad":
                    _register_bw_events(next_grad_fn)

        if hasattr(output, "register_hook") and output.requires_grad:
            # print(f"registering before event for: {output.shape}")
            before_bw_op_hook_handle = output.register_hook(
                lambda output: before_bw_op_hook(self, output.grad_fn, output)
            )

        if hasattr(output, "grad_fn"):
            _register_bw_events(output.grad_fn)
