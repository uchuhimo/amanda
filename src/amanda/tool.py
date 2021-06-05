from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Set

from amanda.event import Event, EventCallback
from amanda.threading import ThreadLocalStack


@dataclass
class Tool:
    namespace: str = None
    _event_to_callback: Dict[Event, EventCallback] = field(default_factory=dict)
    dependencies: List["Tool"] = field(default_factory=list)

    def register_event(self, event: Event, callback: EventCallback) -> None:
        self._event_to_callback[event] = callback

    def unregister_event(self, event: Event) -> None:
        if event in self._event_to_callback:
            del self._event_to_callback[event]

    def get_callback(self, event: Event) -> EventCallback:
        return self._event_to_callback[event]

    def is_registered(self, event: Event) -> bool:
        if event in self._event_to_callback:
            return True
        for dependency in self.dependencies:
            if dependency.is_registered(event):
                return True
        return False

    def depends_on(self, *tools: "Tool") -> None:
        self.dependencies.extend(tools)

    def get_id(self) -> str:
        return None

    def trigger(self, event: Event, context, triggered_tools: Set[str] = None) -> None:
        triggered_tools = triggered_tools or set()
        tool_id = self.get_id()
        if tool_id is not None and tool_id in triggered_tools:
            return
        if tool_id is not None:
            triggered_tools.add(tool_id)
        for dependency in self.dependencies:
            dependency.trigger(event, context, triggered_tools)
        if event in self._event_to_callback:
            self.get_callback(event)(context)


_tools = ThreadLocalStack()


@contextmanager
def apply(*tools: Tool):
    for tool in tools:
        _tools.push(tool)
    yield
    for _ in tools:
        _tools.pop()


def get_tools() -> List[Tool]:
    return list(_tools)
