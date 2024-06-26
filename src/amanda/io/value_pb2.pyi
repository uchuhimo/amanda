# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
    EnumDescriptor as google___protobuf___descriptor___EnumDescriptor,
    FileDescriptor as google___protobuf___descriptor___FileDescriptor,
)

from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer as google___protobuf___internal___containers___RepeatedCompositeFieldContainer,
)

from google.protobuf.internal.enum_type_wrapper import (
    _EnumTypeWrapper as google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    Iterable as typing___Iterable,
    Mapping as typing___Mapping,
    MutableMapping as typing___MutableMapping,
    NewType as typing___NewType,
    Optional as typing___Optional,
    Text as typing___Text,
    cast as typing___cast,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int


DESCRIPTOR: google___protobuf___descriptor___FileDescriptor = ...

NullValueValue = typing___NewType('NullValueValue', builtin___int)
type___NullValueValue = NullValueValue
NullValue: _NullValue
class _NullValue(google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper[NullValueValue]):
    DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
    NULL_VALUE = typing___cast(NullValueValue, 0)
NULL_VALUE = typing___cast(NullValueValue, 0)
type___NullValue = NullValue

class Value(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    null_value: type___NullValueValue = ...
    bool_value: builtin___bool = ...
    int_value: builtin___int = ...
    double_value: builtin___float = ...
    string_value: typing___Text = ...
    bytes_value: builtin___bytes = ...

    @property
    def struct_value(self) -> type___Struct: ...

    @property
    def list_value(self) -> type___ListValue: ...

    @property
    def serialized_value(self) -> type___SerializedValue: ...

    @property
    def type_value(self) -> type___DataTypeValue: ...

    @property
    def ref_value(self) -> type___RefValue: ...

    def __init__(self,
        *,
        struct_value : typing___Optional[type___Struct] = None,
        list_value : typing___Optional[type___ListValue] = None,
        serialized_value : typing___Optional[type___SerializedValue] = None,
        null_value : typing___Optional[type___NullValueValue] = None,
        bool_value : typing___Optional[builtin___bool] = None,
        int_value : typing___Optional[builtin___int] = None,
        double_value : typing___Optional[builtin___float] = None,
        string_value : typing___Optional[typing___Text] = None,
        bytes_value : typing___Optional[builtin___bytes] = None,
        type_value : typing___Optional[type___DataTypeValue] = None,
        ref_value : typing___Optional[type___RefValue] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"bool_value",b"bool_value",u"bytes_value",b"bytes_value",u"double_value",b"double_value",u"int_value",b"int_value",u"kind",b"kind",u"list_value",b"list_value",u"null_value",b"null_value",u"ref_value",b"ref_value",u"serialized_value",b"serialized_value",u"string_value",b"string_value",u"struct_value",b"struct_value",u"type_value",b"type_value"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"bool_value",b"bool_value",u"bytes_value",b"bytes_value",u"double_value",b"double_value",u"int_value",b"int_value",u"kind",b"kind",u"list_value",b"list_value",u"null_value",b"null_value",u"ref_value",b"ref_value",u"serialized_value",b"serialized_value",u"string_value",b"string_value",u"struct_value",b"struct_value",u"type_value",b"type_value"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions___Literal[u"kind",b"kind"]) -> typing_extensions___Literal["struct_value","list_value","serialized_value","null_value","bool_value","int_value","double_value","string_value","bytes_value","type_value","ref_value"]: ...
type___Value = Value

class DataTypeValue(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    namespace: typing___Text = ...
    name: typing___Text = ...

    @property
    def attrs(self) -> type___Struct: ...

    def __init__(self,
        *,
        namespace : typing___Optional[typing___Text] = None,
        name : typing___Optional[typing___Text] = None,
        attrs : typing___Optional[type___Struct] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"attrs",b"attrs"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"attrs",b"attrs",u"name",b"name",u"namespace",b"namespace"]) -> None: ...
type___DataTypeValue = DataTypeValue

class Struct(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class FieldsEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: typing___Text = ...

        @property
        def value(self) -> type___Value: ...

        def __init__(self,
            *,
            key : typing___Optional[typing___Text] = None,
            value : typing___Optional[type___Value] = None,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions___Literal[u"value",b"value"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...
    type___FieldsEntry = FieldsEntry


    @property
    def fields(self) -> typing___MutableMapping[typing___Text, type___Value]: ...

    def __init__(self,
        *,
        fields : typing___Optional[typing___Mapping[typing___Text, type___Value]] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"fields",b"fields"]) -> None: ...
type___Struct = Struct

class ListValue(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def values(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[type___Value]: ...

    def __init__(self,
        *,
        values : typing___Optional[typing___Iterable[type___Value]] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"values",b"values"]) -> None: ...
type___ListValue = ListValue

class SerializedValue(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def type(self) -> type___DataTypeValue: ...

    @property
    def value(self) -> type___Value: ...

    def __init__(self,
        *,
        type : typing___Optional[type___DataTypeValue] = None,
        value : typing___Optional[type___Value] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"type",b"type",u"value",b"value"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"type",b"type",u"value",b"value"]) -> None: ...
type___SerializedValue = SerializedValue

class RefValue(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    schema: typing___Text = ...
    value: typing___Text = ...

    def __init__(self,
        *,
        schema : typing___Optional[typing___Text] = None,
        value : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"schema",b"schema",u"value",b"value"]) -> None: ...
type___RefValue = RefValue
