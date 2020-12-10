import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List

from amanda.event import Event, EventCallback


@dataclass
class Tool:
    _event_to_callback: Dict[Event, EventCallback] = field(default_factory=dict)

    def register_event(self, event: Event, callback: EventCallback) -> None:
        self._event_to_callback[event] = callback

    def get_callback(self, event: Event) -> EventCallback:
        return self._event_to_callback[event]

    def is_registered(self, event: Event) -> bool:
        return event in self._event_to_callback


class ThreadLocalStack(threading.local):
    """A thread-local stack."""

    def __init__(self):
        super(ThreadLocalStack, self).__init__()
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        self.stack.pop()

    def __iter__(self):
        return reversed(self.stack).__iter__()


_tools = ThreadLocalStack()


@contextmanager
def apply(tool: Tool):
    _tools.push(tool)
    yield tool
    _tools.pop()


def get_tools() -> List[Tool]:
    return list(_tools)
