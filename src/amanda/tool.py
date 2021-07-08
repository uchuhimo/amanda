from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set, Union

from amanda.event import (
    Event,
    EventCallback,
    OpCallback,
    on_backward_op_call,
    on_op_call,
)
from amanda.threading import ThreadLocalStack

ToolCallback = Union[EventCallback, OpCallback]


@dataclass
class Tool:
    namespace: str = None
    _event_to_callback: Dict[Event, ToolCallback] = field(default_factory=dict)
    dependencies: List["Tool"] = field(default_factory=list)

    def register_event(self, event: Event, callback: ToolCallback) -> None:
        self._event_to_callback[event] = callback

    def unregister_event(self, event: Event) -> None:
        if event in self._event_to_callback:
            del self._event_to_callback[event]

    def add_inst_for_op(self, callback: OpCallback) -> None:
        self._event_to_callback[on_op_call] = callback

    def add_inst_for_backward_op(self, callback: OpCallback) -> None:
        self._event_to_callback[on_backward_op_call] = callback

    def get_callback(self, event: Event) -> ToolCallback:
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


@dataclass
class ApplyScope:
    cleanup_tasks: List[Callable[[], None]] = field(default_factory=list)


_tools = ThreadLocalStack()
_apply_scopes = ThreadLocalStack()


@contextmanager
def apply(*tools: Tool):
    for tool in tools:
        _tools.push(tool)
    _apply_scopes.push(ApplyScope())
    yield
    scope = _apply_scopes.pop()
    for task in scope.cleanup_tasks:
        task()
    for _ in tools:
        _tools.pop()


class Handler:
    def __init__(self, scope, task) -> None:
        self.scope = scope
        self.task = task

    def unregister(self):
        self.scope.cleanup_tasks.remove(self.task)


def register_cleanup_task(task: Callable[[], None]) -> Handler:
    _apply_scopes.top().cleanup_tasks.append(task)
    return Handler(_apply_scopes.top(), task)


def get_tools() -> List[Tool]:
    return list(_tools)
