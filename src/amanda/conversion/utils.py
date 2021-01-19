import sys
import typing
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar, Union

from google.protobuf import json_format
from google.protobuf.message import Message

from amanda.namespace import is_qualified

T = TypeVar("T", bound=Message)


def to_proto(proto: Union[T, str, bytes, Path], message_type: Type[T]) -> T:
    if not isinstance(proto, message_type):
        if isinstance(proto, bytes):
            proto_bytes = proto
        else:
            if isinstance(proto, str):
                proto = Path(proto)
            proto_bytes = typing.cast(Path, proto).read_bytes()
        proto = message_type()
        proto.ParseFromString(proto_bytes)
    return proto


def without_internal_attrs(attrs: Dict[str, Any], names: List[str] = []):
    return {
        name: value
        for name, value in attrs.items()
        if not (
            is_qualified(name)
            or name in names
            or name.startswith("output_port/")
            or name.startswith("input_port/")
        )
    }


def repeated_fields_to_dict(repeated_fields) -> Dict[str, Any]:
    return {
        proto_def.name: json_format.MessageToDict(
            proto_def, preserving_proto_field_name=True
        )
        for proto_def in repeated_fields
    }


@contextmanager
def recursionlimit(limit: int):
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(limit)
        yield None
    finally:
        sys.setrecursionlimit(old_limit)
