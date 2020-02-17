from abc import ABC, abstractmethod
from typing import Dict, Optional, Union


class Namespace:
    def __init__(self, namespace: str):
        assert namespace != ""
        self.namespace = namespace

    def qualified(self, name: str) -> str:
        return f"/{self.namespace}/{get_base_name(name)}"

    def belong_to(self, namespace: "Namespace") -> bool:
        return self.namespace.startswith(namespace.namespace + "/")

    def __truediv__(self, other: Union["Namespace", str]) -> "Namespace":
        if isinstance(other, str):
            namespace = other
        else:
            namespace = other.namespace
        return Namespace(f"{self.namespace}/{namespace}")

    def __eq__(self, other):
        if isinstance(other, Namespace) and self.namespace == other.namespace:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.namespace)

    def __repr__(self) -> str:
        return f"Namespace({self.namespace})"


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
        if get_namespace(name) == target.namespace:
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
        src_op,
        src_attr_name,
        src_attr_value,
        dst_op,
        dst_attr_name,
        dst_attr_value,
        tag=None,
    ):
        ...


class Registry:
    def __init__(self):
        self.mappers: Dict[str, Dict[str, Mapper]] = {}

    def get_mapper(self, source: Namespace, target: Namespace) -> Mapper:
        return self.mappers[source.namespace][target.namespace]

    def add_mapper(self, source: Namespace, target: Namespace, mapper: Mapper):
        if source.namespace not in self.mappers:
            self.mappers[source.namespace] = {}
        self.mappers[source.namespace][target.namespace] = mapper

    def remove_mapper(self, source: Namespace, target: Namespace) -> Optional[Mapper]:
        if source.namespace not in self.mappers:
            return None
        elif target.namespace not in self.mappers[source.namespace]:
            return None
        else:
            mapper = self.mappers[source.namespace][target.namespace]
            del self.mappers[source.namespace][target.namespace]
            return mapper


_global_registry = Registry()


def get_global_registry() -> Registry:
    return _global_registry


def get_mapper(source: Namespace, target: Namespace) -> DumpMapper:
    return DumpMapper()


def get_mapping_table(src: str, dst: str) -> DumpMapper:
    return DumpMapper()
