from abc import ABC, abstractmethod
from typing import Dict, Optional, Union


class Namespace:
    def __init__(self, full_name: str):
        assert full_name != ""
        self.full_name = full_name

    def qualified(self, name: str) -> str:
        return f"/{self.full_name}/{get_base_name(name)}"

    def belong_to(self, namespace: "Namespace") -> bool:
        return self.full_name == namespace.full_name or self.full_name.startswith(
            namespace.full_name + "/"
        )

    def __truediv__(self, other: Union["Namespace", str]) -> "Namespace":
        if isinstance(other, str):
            full_name = other
        else:
            full_name = other.full_name
        return Namespace(f"{self.full_name}/{full_name}")

    def __eq__(self, other):
        if isinstance(other, Namespace) and self.full_name == other.full_name:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.full_name)

    def __repr__(self) -> str:
        return f"Namespace({self.full_name})"


def is_qualified(name: str) -> bool:
    return name.startswith("/")


def get_namespace(name: str) -> str:
    if is_qualified(name):
        return name[1 : name.rfind("/")]
    else:
        return ""


def get_base_name(name: str) -> str:
    if is_qualified(name):
        return name[name.rfind("/") + 1 :]
    else:
        return name


def map_namespace(name: str, source: Namespace, target: Namespace) -> str:
    if is_qualified(name):
        if get_namespace(name) == target.full_name:
            return get_base_name(name)
        else:
            return name
    else:
        return source.qualified(name)


_default_namespace = Namespace("amanda")
_internal_namespace = _default_namespace / Namespace("internal")


def default_namespace() -> Namespace:
    return _default_namespace


def internal_namespace() -> Namespace:
    return _internal_namespace


def exp(expression: str):
    ...


class Mapper(ABC):
    @abstractmethod
    def map(self, graph, namespace: Namespace):
        ...


class DumpMapper:
    def add_rule(self, src_op_list, dst_op_list, map_func, variant=None):
        ...

    def insert_rule(
        self,
        src_op=None,
        src_attr_name=None,
        src_attr_value=None,
        dst_op=None,
        dst_attr_name=None,
        dst_attr_value=None,
        dst_value=None,
        tag=None,
    ):
        ...


class Registry:
    def __init__(self):
        self.mappers: Dict[str, Dict[str, Mapper]] = {}

    def get_mapper(self, source: Namespace, target: Namespace) -> Mapper:
        return self.mappers[source.full_name][target.full_name]

    def add_mapper(self, source: Namespace, target: Namespace, mapper: Mapper):
        if source.full_name not in self.mappers:
            self.mappers[source.full_name] = {}
        self.mappers[source.full_name][target.full_name] = mapper

    def remove_mapper(self, source: Namespace, target: Namespace) -> Optional[Mapper]:
        if source.full_name not in self.mappers:
            return None
        elif target.full_name not in self.mappers[source.full_name]:
            return None
        else:
            mapper = self.mappers[source.full_name][target.full_name]
            del self.mappers[source.full_name][target.full_name]
            return mapper


_global_registry = Registry()


def get_global_registry() -> Registry:
    return _global_registry


def get_mapper(source: Namespace, target: Namespace) -> DumpMapper:
    return DumpMapper()


def get_mapping_table(src: str, dst: str) -> DumpMapper:
    return DumpMapper()
