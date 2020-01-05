import typing
from pathlib import Path
from typing import Type, TypeVar, Union

from google.protobuf.message import Message

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
