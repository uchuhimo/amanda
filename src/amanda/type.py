from dataclasses import dataclass, field
from typing import Any

from amanda.attributes import Attributes
from amanda.namespace import Namespace, default_namespace


@dataclass
class DataType:
    namespace: Namespace
    name: str
    attrs: Attributes = field(default_factory=Attributes)
    _raw: Any = field(default=None, repr=False, hash=False, compare=False)

    @property
    def raw(self) -> Any:
        if self._raw is None:
            from amanda.io.serde import deserialize_type

            self._raw = deserialize_type(self)
        return self._raw


unknown_type = DataType(
    namespace=default_namespace(),
    name="Unknown",
)
