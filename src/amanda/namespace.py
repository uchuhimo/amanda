from abc import ABC, abstractmethod
from typing import Dict, Optional


class Namespace:
    def __init__(self, name: str):
        self.name = name


class Mapper(ABC):
    @abstractmethod
    def map(self, graph, namespace: Namespace):
        ...


class Registry:
    def __init__(self):
        self.mappers: Dict[str, Dict[str, Mapper]] = {}

    def get_mapper(self, source: Namespace, target: Namespace) -> Mapper:
        return self.mappers[source.name][target.name]

    def add_mapper(self, source: Namespace, target: Namespace, mapper: Mapper):
        if source.name not in self.mappers:
            self.mappers[source.name] = {}
        self.mappers[source.name][target.name] = mapper

    def remove_mapper(self, source: Namespace, target: Namespace) -> Optional[Mapper]:
        if source.name not in self.mappers:
            return None
        elif target.name not in self.mappers[source.name]:
            return None
        else:
            mapper = self.mappers[source.name][target.name]
            del self.mappers[source.name][target.name]
            return mapper


_global_registry = Registry()


def get_global_registry() -> Registry:
    return _global_registry


_default_namespace = Namespace(name="amanda")


def default_namespace() -> Namespace:
    return _default_namespace


_internal_namespace = Namespace(name="amanda/internal")


def internal_namespace() -> Namespace:
    return _internal_namespace
