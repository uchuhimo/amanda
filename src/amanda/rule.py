from abc import ABC, abstractmethod
from typing import List, Set, Tuple

from amanda.exception import OpMappingError
from amanda.graph import Graph, Op
from amanda.namespace import Mapper


class Rule(ABC):
    @abstractmethod
    def apply(self, source: Graph, target: Graph, op: Op) -> Tuple[List[Op], List[Op]]:
        ...


class RuleMapper(Mapper):
    def __init__(self, rules: List[Rule]):
        self.rules: List[Rule] = rules

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def remove_rule(self, rule: Rule):
        self.rules.remove(rule)

    def mapping(self, graph: Graph) -> Graph:
        new_graph = Graph()
        mapped_ops: Set[Op] = set()
        for op in graph.post_order_ops:
            if op not in mapped_ops:
                for rule in self.rules:
                    source_ops, target_ops = rule.apply(graph, new_graph, op)
                    mapped_ops.update(source_ops)
                    for target_op in target_ops:
                        if target_op not in new_graph:
                            new_graph.add_op(target_op)
                    break
                raise OpMappingError(graph, new_graph, op)
        return new_graph
