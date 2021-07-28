from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from amanda.event import OpContext

RuleCallback = Callable[[OpContext], OpContext]


@dataclass
class Mapping:
    _namespace_to_rule: Dict[Tuple[str, str, bool, bool], RuleCallback] = field(
        default_factory=dict
    )
    _mappings: List["Mapping"] = field(default_factory=list)

    def register_rule(
        self,
        source: str,
        target: str,
        func: RuleCallback,
        backward: bool = None,
        require_outputs: bool = None,
    ):
        self._namespace_to_rule[(source, target, backward, require_outputs)] = func

    def use(self, other: "Mapping"):
        self._mappings.append(other)
