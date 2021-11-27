import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterable, List


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
after_op_call = Event("after_op_call")
after_backward_op_call = Event("after_backward_op_call")
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
    trackback: Any = None


class OpContext(dict):
    from amanda.tool import Tool

    def __init__(self, tools: Iterable[Tool], namespace: str):
        super().__init__()
        from amanda.tool import Tool

        self.namespace = namespace
        self.tools: Iterable[Tool] = tools
        self.actions: List[Action] = []
        self.is_op_replaced = False
        self.is_backward_op_replaced = False

    def trigger(self, event: Event, **kwargs) -> None:
        self.update(**kwargs)
        for tool in self.tools:
            if tool and tool.is_registered(event):
                tool.trigger(event, self)

    def is_registered(self, event: Event) -> bool:
        for tool in self.tools:
            if tool.is_registered(event):
                return True
        return False

    def inherite(self):
        new_context = OpContext(self.tools, self.namespace)
        for key, value in self.items():
            new_context[key] = value
        return new_context

    @property
    def op(self):
        return self.get_op()

    @property
    def op_id(self):
        return self.get_op_id()

    @property
    def backward_op(self):
        return self.get_backward_op()

    @property
    def backward_op_id(self):
        return self.get_backward_op_id()

    @property
    def inputs(self):
        return self.get_inputs()

    @property
    def outputs(self):
        return self.get_outputs()

    @property
    def grad_outputs(self):
        return self.get_grad_outputs()

    @property
    def grad_inputs(self):
        return self.get_grad_inputs()

    def get_op(self):
        return self["op"]

    def get_op_id(self):
        return self["op_id"] if "op_id" in self else None

    def get_inputs(self):
        return self["inputs"]

    def get_outputs(self):
        return self["outputs"]

    def get_backward_op(self):
        return self["backward_op"]

    def get_backward_op_id(self):
        return self["backward_op_id"]

    def get_grad_outputs(self):
        return self["grad_outputs"]

    def get_grad_inputs(self):
        return self["grad_inputs"]

    def is_forward(self) -> bool:
        return "backward_op" not in self

    def is_backward(self) -> bool:
        return "backward_op" in self

    def insert_before_op(self, func, inputs: List[int] = None, **kwargs):
        self.actions.append(
            Action(
                type="insert_before_op",
                func=func,
                inputs=inputs,
                kwargs=kwargs,
                trackback=sys.exc_info()[2],
            )
        )

    def insert_after_op(self, func, outputs: List[int] = None, **kwargs):
        self.actions.append(
            Action(
                type="insert_after_op",
                func=func,
                outputs=outputs,
                kwargs=kwargs,
                trackback=sys.exc_info()[2],
            )
        )

    def insert_before_backward_op(self, func, grad_outputs: List[int] = None, **kwargs):
        self.actions.append(
            Action(
                type="insert_before_backward_op",
                func=func,
                inputs=grad_outputs,
                kwargs=kwargs,
                trackback=sys.exc_info()[2],
            )
        )

    def insert_after_backward_op(self, func, grad_inputs: List[int] = None, **kwargs):
        self.actions.append(
            Action(
                type="insert_after_backward_op",
                func=func,
                outputs=grad_inputs,
                kwargs=kwargs,
                trackback=sys.exc_info()[2],
            )
        )

    def replace_op(self, func, inputs: List[int] = None, **kwargs):
        if self.is_op_replaced:
            raise RuntimeError("cannot replace op twice")
        else:
            self.is_op_replaced = True
        self.actions.append(
            Action(
                type="replace_op",
                func=func,
                inputs=inputs,
                kwargs=kwargs,
                trackback=sys.exc_info()[2],
            )
        )

    def replace_backward_op(self, func, grad_outputs: List[int] = None, **kwargs):
        if self.is_backward_op_replaced:
            raise RuntimeError("cannot replace backward op twice")
        else:
            self.is_backward_op_replaced = True
        self.actions.append(
            Action(
                type="replace_backward_op",
                func=func,
                inputs=grad_outputs,
                kwargs=kwargs,
                trackback=sys.exc_info()[2],
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
        for tool in self.tools:
            if tool and tool.is_registered(event):
                tool.trigger(event, self)

    def is_registered(self, event: Event) -> bool:
        for tool in self.tools:
            if tool.is_registered(event):
                return True
        return False
