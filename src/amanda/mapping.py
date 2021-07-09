from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

from amanda.event import OpContext

RuleCallback = Callable[[OpContext], OpContext]


@dataclass
class Mapping:
    _namespace_to_rule: Dict[Tuple[str, str], RuleCallback] = field(
        default_factory=dict
    )

    def register_rule(self, source: str, target: str, func: RuleCallback):
        self._namespace_to_rule[(source, target)] = func
