import itertools
from collections import defaultdict
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
    _event_to_callbacks: Dict[Event, List[ToolCallback]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _event_to_all_callbacks: Dict[Event, List[ToolCallback]] = None
    _mappings: List[Mapping] = field(default_factory=list)
    dependencies: List["Tool"] = field(default_factory=list)

    def collect_callbacks(self):
        if self._event_to_all_callbacks is not None:
            return
        self._event_to_all_callbacks = defaultdict(list)
        callback_set: Set[int] = set()
        for event, callbacks in self._event_to_callbacks.items():
            for callback in callbacks:
                if id(callback) not in callback_set:
                    self._event_to_all_callbacks[event].append(callback)
                    callback_set.add(id(callback))
        for dependency in self.dependencies:
            dependency.collect_callbacks()
            for event, callbacks in dependency._event_to_all_callbacks.items():
                for callback in callbacks:
                    if id(callback) not in callback_set:
                        self._event_to_all_callbacks[event].append(callback)
                        callback_set.add(id(callback))

    def register_event(self, event: Event, callback: ToolCallback) -> None:
        self._event_to_callbacks[event].append(callback)

    def register_mapping(self, mapping: Mapping):
        self._mappings.append(mapping)

    def load_mapping(self, file):
        ...

    def add_inst_for_op(
        self,
        callback: OpCallback,
        backward: bool = False,
        require_outputs: bool = False,
    ) -> None:
        if backward:
            if require_outputs:
                self._event_to_callbacks[after_backward_op_call].append(callback)
            else:
                self._event_to_callbacks[on_backward_op_call].append(callback)
        else:
            if require_outputs:
                self._event_to_callbacks[after_op_call].append(callback)
            else:
                self._event_to_callbacks[on_op_call].append(callback)

    def is_registered(self, event: Event) -> bool:
        if self._event_to_all_callbacks is None:
            self.collect_callbacks()
        return event in self._event_to_all_callbacks

    def depends_on(self, *tools: "Tool") -> None:
        self.dependencies.extend(tools)

    def get_id(self) -> str:
        return None

    def trigger(self, event: Event, context) -> None:
        if self._event_to_all_callbacks is None:
            self.collect_callbacks()
        if event in self._event_to_all_callbacks:
            for callback in self._event_to_all_callbacks[event]:
                callback(context)


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
    try:
        yield
    finally:
        _apply_scopes.pop()
        for task in apply_scope.cleanup_tasks:
            task()


def get_apply_scope():
    return _apply_scopes.top()


def register_cleanup_task(task: Callable[[], None]) -> Handler:
    return register_handler(_apply_scopes.top().cleanup_tasks, task)


def get_tools() -> List[Tool]:
    if len(_apply_scopes) == 1:
        return _apply_scopes.top().tools
    elif len(_apply_scopes) == 0:
        return []
    else:
        return list(itertools.chain([scope.tools for scope in _apply_scopes]))
