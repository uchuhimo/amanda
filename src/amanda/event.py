from dataclasses import dataclass
from typing import Callable, Dict


@dataclass(frozen=True)
class Event:
    name: str


on_graph_loaded = Event("on_graph_loaded")
before_op_executed = Event("before_op_executed")
update_graph = Event("update_graph")
update_op = Event("update_op")

EventCallback = Callable[["EventContext"], None]


class EventContext(dict):
    def __init__(self):
        super(EventContext, self).__init__()
        self._event_to_callback: Dict[Event, EventCallback] = {}

    def trigger(self, event: Event) -> None:
        from amanda.tool import get_tools

        if event in self._event_to_callback:
            return self.get_callback(event)(self)
        for tool in get_tools():
            if tool.is_registered(event):
                return tool.get_callback(event)(self)

    def register_event(self, event: Event, callback: EventCallback) -> None:
        self._event_to_callback[event] = callback

    def get_callback(self, event: Event) -> EventCallback:
        return self._event_to_callback[event]

    def is_registered(self, event: Event) -> bool:
        from amanda.tool import get_tools

        if event in self._event_to_callback:
            return True
        for tool in get_tools():
            if tool.is_registered(event):
                return True
        return False
