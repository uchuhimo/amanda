from dataclasses import dataclass, field
from typing import Any

from amanda.attributes import Attributes
from amanda.namespace import Namespace, default_namespace


@dataclass
class DataType:
    namespace: Namespace
    name: str
    attrs: Attributes = field(default_factory=Attributes)
    raw: Any = None


unknown_type = DataType(namespace=default_namespace(), name="Unknown")
