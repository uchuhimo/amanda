from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set

from amanda.exception import OpMappingError
from amanda.graph import Graph, Op
from amanda.namespace import Mapper, Namespace, map_namespace


@dataclass
class OpMapping:
    source_ops: List[Op]
    target_ops: List[Op]


class Rule(ABC):
    @abstractmethod
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        ...


class NoopRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        return OpMapping(source_ops=[op], target_ops=[op])


class RuleMapper(Mapper):
    def __init__(self, rules: List[Rule]):
        self.rules: List[Rule] = rules

    def add_rule(self, rule: Rule):
        self.rules.insert(0, rule)

    def remove_rule(self, rule: Rule):
        self.rules.remove(rule)

    def map(self, graph: Graph, namespace: Namespace) -> Graph:
        source_namespace = graph.namespace
        if namespace == source_namespace:
            return graph
        new_graph = graph.copy()
        for name, value in new_graph.attrs.items():
            new_name = map_namespace(name, source_namespace, namespace)
            if name != new_name:
                new_graph.attrs[new_name] = value
                del new_graph.attrs[name]
        new_graph.namespace = namespace
        for op in new_graph.ops:
            for name, value in op.attrs.items():
                new_name = map_namespace(name, source_namespace, namespace)
                if name != new_name:
                    op.attrs[new_name] = value
                    del op.attrs[name]
            op.namespace = namespace
        mapped_ops: Set[Op] = set()
        for op in new_graph.sorted_ops:
            if op not in mapped_ops:
                has_matched_rule = False
                for rule in self.rules:
                    mapping = rule.apply(new_graph, op)
                    if len(mapping.source_ops) != 0:
                        mapped_ops.update(mapping.source_ops)
                        for target_op in mapping.target_ops:
                            if target_op not in new_graph:
                                new_graph.add_op(target_op)
                        has_matched_rule = True
                        break
                if not has_matched_rule:
                    raise OpMappingError(new_graph, op)
        return new_graph


class RawRuleMapper(Mapper):
    def __init__(self, rules: List[Rule]):
        self.rules: List[Rule] = rules

    def add_rule(self, rule: Rule):
        self.rules.insert(0, rule)

    def remove_rule(self, rule: Rule):
        self.rules.remove(rule)

    def map(self, graph: Graph, namespace: Namespace) -> Graph:
        source_namespace = graph.namespace
        if namespace == source_namespace:
            return graph
        new_graph = graph.copy()
        new_graph.namespace = namespace
        if len(self.rules) == 0:
            return new_graph
        mapped_ops: Set[Op] = set()
        for op in new_graph.sorted_ops:
            if op not in mapped_ops:
                has_matched_rule = False
                for rule in self.rules:
                    mapping = rule.apply(new_graph, op)
                    if len(mapping.source_ops) != 0:
                        mapped_ops.update(mapping.source_ops)
                        for target_op in mapping.target_ops:
                            if target_op not in new_graph:
                                new_graph.add_op(target_op)
                        has_matched_rule = True
                        break
                if not has_matched_rule:
                    raise OpMappingError(new_graph, op)
        return new_graph
