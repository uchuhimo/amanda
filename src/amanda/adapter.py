from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Generic, Type, TypeVar

from amanda.event import EventContext

T = TypeVar("T")


class Adapter(ABC, Generic[T]):
    @abstractmethod
    def adapt(self, target: T, context: EventContext) -> T:
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


def adapt(target: T, context: EventContext = None) -> T:
    return (
        get_adapter_registry()
        .dispatch_type(type(target))
        .adapt(target, context or EventContext())
    )
