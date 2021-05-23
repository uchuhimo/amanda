from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List

from amanda.event import Event, EventCallback
from amanda.threading import ThreadLocalStack


@dataclass
class Tool:
    namespace: str
    _event_to_callback: Dict[Event, EventCallback] = field(default_factory=dict)

    def register_event(self, event: Event, callback: EventCallback) -> None:
        self._event_to_callback[event] = callback

    def unregister_event(self, event: Event) -> None:
        if self.is_registered(event):
            del self._event_to_callback[event]

    def get_callback(self, event: Event) -> EventCallback:
        return self._event_to_callback[event]

    def is_registered(self, event: Event) -> bool:
        return event in self._event_to_callback


_tools = ThreadLocalStack()


@contextmanager
def apply(tool: Tool):
    _tools.push(tool)
    yield tool
    _tools.pop()


def get_tools() -> List[Tool]:
    return list(_tools)
