# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: amanda/io/graph.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from amanda.io import value_pb2 as amanda_dot_io_dot_value__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='amanda/io/graph.proto',
  package='amanda',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x15\x61manda/io/graph.proto\x12\x06\x61manda\x1a\x15\x61manda/io/value.proto\"<\n\x07PortDef\x12\x0c\n\x04name\x18\x01 \x01(\t\x12#\n\x04type\x18\x02 \x01(\x0b\x32\x15.amanda.DataTypeValue\"t\n\x07\x45\x64geDef\x12$\n\x03src\x18\x01 \x01(\x0b\x32\x17.amanda.SerializedValue\x12$\n\x03\x64st\x18\x02 \x01(\x0b\x32\x17.amanda.SerializedValue\x12\x1d\n\x05\x61ttrs\x18\x03 \x01(\x0b\x32\x0e.amanda.Struct\"\xbc\x02\n\x07NodeDef\x12+\n\tnode_kind\x18\x01 \x01(\x0e\x32\x18.amanda.NodeDef.NodeKind\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x0c\n\x04type\x18\x03 \x01(\t\x12\x11\n\tnamespace\x18\x04 \x01(\t\x12$\n\x0binput_ports\x18\x05 \x03(\x0b\x32\x0f.amanda.PortDef\x12%\n\x0coutput_ports\x18\x06 \x03(\x0b\x32\x0f.amanda.PortDef\x12\x1c\n\x03ops\x18\x07 \x03(\x0b\x32\x0f.amanda.NodeDef\x12\x1e\n\x05\x65\x64ges\x18\x08 \x03(\x0b\x32\x0f.amanda.EdgeDef\x12\x1d\n\x05\x61ttrs\x18\t \x01(\x0b\x32\x0e.amanda.Struct\"+\n\x08NodeKind\x12\x06\n\x02OP\x10\x00\x12\t\n\x05GRAPH\x10\x01\x12\x0c\n\x08SUBGRAPH\x10\x02\x62\x06proto3'
  ,
  dependencies=[amanda_dot_io_dot_value__pb2.DESCRIPTOR,])



_NODEDEF_NODEKIND = _descriptor.EnumDescriptor(
  name='NodeKind',
  full_name='amanda.NodeDef.NodeKind',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='OP', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='GRAPH', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SUBGRAPH', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=510,
  serialized_end=553,
)
_sym_db.RegisterEnumDescriptor(_NODEDEF_NODEKIND)


_PORTDEF = _descriptor.Descriptor(
  name='PortDef',
  full_name='amanda.PortDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='amanda.PortDef.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='type', full_name='amanda.PortDef.type', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=56,
  serialized_end=116,
)


_EDGEDEF = _descriptor.Descriptor(
  name='EdgeDef',
  full_name='amanda.EdgeDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='src', full_name='amanda.EdgeDef.src', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dst', full_name='amanda.EdgeDef.dst', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='attrs', full_name='amanda.EdgeDef.attrs', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=118,
  serialized_end=234,
)


_NODEDEF = _descriptor.Descriptor(
  name='NodeDef',
  full_name='amanda.NodeDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='node_kind', full_name='amanda.NodeDef.node_kind', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='name', full_name='amanda.NodeDef.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='type', full_name='amanda.NodeDef.type', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='namespace', full_name='amanda.NodeDef.namespace', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='input_ports', full_name='amanda.NodeDef.input_ports', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='output_ports', full_name='amanda.NodeDef.output_ports', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ops', full_name='amanda.NodeDef.ops', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='edges', full_name='amanda.NodeDef.edges', index=7,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='attrs', full_name='amanda.NodeDef.attrs', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _NODEDEF_NODEKIND,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=237,
  serialized_end=553,
)

_PORTDEF.fields_by_name['type'].message_type = amanda_dot_io_dot_value__pb2._DATATYPEVALUE
_EDGEDEF.fields_by_name['src'].message_type = amanda_dot_io_dot_value__pb2._SERIALIZEDVALUE
_EDGEDEF.fields_by_name['dst'].message_type = amanda_dot_io_dot_value__pb2._SERIALIZEDVALUE
_EDGEDEF.fields_by_name['attrs'].message_type = amanda_dot_io_dot_value__pb2._STRUCT
_NODEDEF.fields_by_name['node_kind'].enum_type = _NODEDEF_NODEKIND
_NODEDEF.fields_by_name['input_ports'].message_type = _PORTDEF
_NODEDEF.fields_by_name['output_ports'].message_type = _PORTDEF
_NODEDEF.fields_by_name['ops'].message_type = _NODEDEF
_NODEDEF.fields_by_name['edges'].message_type = _EDGEDEF
_NODEDEF.fields_by_name['attrs'].message_type = amanda_dot_io_dot_value__pb2._STRUCT
_NODEDEF_NODEKIND.containing_type = _NODEDEF
DESCRIPTOR.message_types_by_name['PortDef'] = _PORTDEF
DESCRIPTOR.message_types_by_name['EdgeDef'] = _EDGEDEF
DESCRIPTOR.message_types_by_name['NodeDef'] = _NODEDEF
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PortDef = _reflection.GeneratedProtocolMessageType('PortDef', (_message.Message,), {
  'DESCRIPTOR' : _PORTDEF,
  '__module__' : 'amanda.io.graph_pb2'
  # @@protoc_insertion_point(class_scope:amanda.PortDef)
  })
_sym_db.RegisterMessage(PortDef)

EdgeDef = _reflection.GeneratedProtocolMessageType('EdgeDef', (_message.Message,), {
  'DESCRIPTOR' : _EDGEDEF,
  '__module__' : 'amanda.io.graph_pb2'
  # @@protoc_insertion_point(class_scope:amanda.EdgeDef)
  })
_sym_db.RegisterMessage(EdgeDef)

NodeDef = _reflection.GeneratedProtocolMessageType('NodeDef', (_message.Message,), {
  'DESCRIPTOR' : _NODEDEF,
  '__module__' : 'amanda.io.graph_pb2'
  # @@protoc_insertion_point(class_scope:amanda.NodeDef)
  })
_sym_db.RegisterMessage(NodeDef)


# @@protoc_insertion_point(module_scope)