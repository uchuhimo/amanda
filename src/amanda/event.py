from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class Event:
    name: str


on_graph_loaded = Event("on_graph_loaded")
before_subgraph_executed = Event("before_subgraph_executed")
after_subgraph_executed = Event("after_subgraph_executed")
before_op_executed = Event("before_op_executed")
after_op_executed = Event("after_op_executed")

EventCallback = Callable[["EventContext"], None]


class EventContext(dict):
    from amanda.tool import Tool

    def __init__(self, tools: Iterable[Tool]):
        super(EventContext, self).__init__()
        from amanda.tool import Tool

        self.tools: Iterable[Tool] = tools

    def trigger(self, event: Event, **kwargs) -> None:
        self.update(**kwargs)
        for tool in self.tools:
            if tool.is_registered(event):
                tool.get_callback(event)(self)

    def is_registered(self, event: Event) -> bool:
        for tool in self.tools:
            if tool.is_registered(event):
                return True
        return False
