import itertools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Set, Union

from amanda.event import (
    Event,
    EventCallback,
    OpCallback,
    after_backward_op_call,
    after_op_call,
    on_backward_op_call,
    on_op_call,
)
from amanda.lang import Handler, register_handler
from amanda.threading import ThreadLocalStack

ToolCallback = Union[EventCallback, OpCallback]


@dataclass
class Tool:
    namespace: str = None
    _event_to_callback: Dict[Event, ToolCallback] = field(default_factory=dict)
    _mappings: List[Mapping] = field(default_factory=list)
    dependencies: List["Tool"] = field(default_factory=list)

    def register_event(self, event: Event, callback: ToolCallback) -> None:
        self._event_to_callback[event] = callback

    def unregister_event(self, event: Event) -> None:
        if event in self._event_to_callback:
            del self._event_to_callback[event]

    def register_mapping(self, mapping: Mapping):
        self._mappings.append(mapping)

    def add_inst_for_op(
        self,
        callback: OpCallback,
        backward: bool = False,
        require_outputs: bool = False,
    ) -> None:
        if backward:
            if require_outputs:
                self._event_to_callback[after_backward_op_call] = callback
            else:
                self._event_to_callback[on_backward_op_call] = callback
        else:
            if require_outputs:
                self._event_to_callback[after_op_call] = callback
            else:
                self._event_to_callback[on_op_call] = callback

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
    tools: List[Tool] = field(default_factory=list)


_apply_scopes = ThreadLocalStack()


@contextmanager
def apply(*tools: Tool):
    apply_scope = ApplyScope()
    for tool in tools:
        apply_scope.tools.append(tool)
    _apply_scopes.push(apply_scope)
    yield
    _apply_scopes.pop()
    for task in apply_scope.cleanup_tasks:
        task()


def register_cleanup_task(task: Callable[[], None]) -> Handler:
    return register_handler(_apply_scopes.top().cleanup_tasks, task)


def get_tools() -> List[Tool]:
    if len(_apply_scopes) == 1:
        return _apply_scopes.top().tools
    elif len(_apply_scopes) == 0:
        return []
    else:
        return list(itertools.chain([scope.tools for scope in _apply_scopes]))
