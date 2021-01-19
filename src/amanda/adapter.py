from dataclasses import dataclass, field
from typing import Any, Dict, Type, TypeVar

from amanda.event import EventContext
from amanda.tool import Tool

T = TypeVar("T")


@dataclass
class Adapter:
    namespace: str

    def apply(self, target: Any, context: EventContext) -> None:
        ...


@dataclass
class AdapterRegistry:
    _type_to_adapter: Dict[Type, Adapter] = field(default_factory=dict)

    def register_adapter(self, type: Type, adapter: Adapter) -> None:
        self._type_to_adapter[type] = adapter

    def dispatch_type(self, type: Type) -> Adapter:
        for clazz in type.mro():
            if clazz in self._type_to_adapter:
                return self._type_to_adapter[clazz]
        raise RuntimeError(f"There is no adapter for type {type}")


_registry = AdapterRegistry()


def get_adapter_registry() -> AdapterRegistry:
    return _registry


def apply(target: T, *tools: Tool) -> None:
    return (
        get_adapter_registry()
        .dispatch_type(type(target))
        .apply(target, EventContext(tools=tools))
    )
