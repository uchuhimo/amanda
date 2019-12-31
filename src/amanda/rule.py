from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set

from amanda.exception import OpMappingError
from amanda.graph import Graph, Op
from amanda.namespace import Mapper, Namespace


@dataclass
class OpMapping:
    source_ops: List[Op]
    target_ops: List[Op]


class Rule(ABC):
    @abstractmethod
    def apply(self, source: Graph, target: Graph, op: Op) -> OpMapping:
        ...


class NoopRule(Rule):
    def apply(self, source: Graph, target: Graph, op: Op) -> OpMapping:
        return OpMapping(source_ops=[op], target_ops=[op])


class RuleMapper(Mapper):
    def __init__(self, rules: List[Rule]):
        self.rules: List[Rule] = rules

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def remove_rule(self, rule: Rule):
        self.rules.remove(rule)

    def map(self, graph: Graph, namespace: Namespace) -> Graph:
        new_graph = Graph(attrs=dict(graph.attrs))
        mapped_ops: Set[Op] = set()
        for op in graph.post_order_ops:
            if op not in mapped_ops:
                has_matched_rule = False
                for rule in self.rules:
                    mapping = rule.apply(graph, new_graph, op)
                    if len(mapping.source_ops) != 0:
                        mapped_ops.update(mapping.source_ops)
                        for target_op in mapping.target_ops:
                            if target_op not in new_graph:
                                new_graph.add_op(target_op)
                        has_matched_rule = True
                        break
                if not has_matched_rule:
                    raise OpMappingError(graph, new_graph, op)
        new_graph.namespace = namespace
        return new_graph
