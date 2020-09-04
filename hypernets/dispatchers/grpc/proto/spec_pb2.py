# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: hypernets/dispatchers/grpc/proto/spec.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='hypernets/dispatchers/grpc/proto/spec.proto',
  package='hypernets.dispatchers.proto',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n+hypernets/dispatchers/grpc/proto/spec.proto\x12\x1bhypernets.dispatchers.proto\"\x17\n\x07RpcCode\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\"\"\n\x0bNextRequest\x12\x13\n\x0b\x65xecutor_id\x18\x01 \x01(\t\"6\n\tTrailItem\x12\x10\n\x08trail_no\x18\x01 \x01(\x05\x12\x17\n\x0fspace_file_path\x18\x02 \x01(\t\"=\n\x0bTrailStatus\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\x10\n\x08trail_no\x18\x02 \x01(\x05\x12\x0e\n\x06reward\x18\x03 \x01(\x02\x32\xc6\x01\n\x0cSearchDriver\x12Z\n\x04next\x12(.hypernets.dispatchers.proto.NextRequest\x1a&.hypernets.dispatchers.proto.TrailItem\"\x00\x12Z\n\x06report\x12(.hypernets.dispatchers.proto.TrailStatus\x1a$.hypernets.dispatchers.proto.RpcCode\"\x00\x62\x06proto3'
)




_RPCCODE = _descriptor.Descriptor(
  name='RpcCode',
  full_name='hypernets.dispatchers.proto.RpcCode',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='code', full_name='hypernets.dispatchers.proto.RpcCode.code', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=76,
  serialized_end=99,
)


_NEXTREQUEST = _descriptor.Descriptor(
  name='NextRequest',
  full_name='hypernets.dispatchers.proto.NextRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='executor_id', full_name='hypernets.dispatchers.proto.NextRequest.executor_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=101,
  serialized_end=135,
)


_TRAILITEM = _descriptor.Descriptor(
  name='TrailItem',
  full_name='hypernets.dispatchers.proto.TrailItem',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='trail_no', full_name='hypernets.dispatchers.proto.TrailItem.trail_no', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='space_file_path', full_name='hypernets.dispatchers.proto.TrailItem.space_file_path', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=137,
  serialized_end=191,
)


_TRAILSTATUS = _descriptor.Descriptor(
  name='TrailStatus',
  full_name='hypernets.dispatchers.proto.TrailStatus',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='code', full_name='hypernets.dispatchers.proto.TrailStatus.code', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='trail_no', full_name='hypernets.dispatchers.proto.TrailStatus.trail_no', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='reward', full_name='hypernets.dispatchers.proto.TrailStatus.reward', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=193,
  serialized_end=254,
)

DESCRIPTOR.message_types_by_name['RpcCode'] = _RPCCODE
DESCRIPTOR.message_types_by_name['NextRequest'] = _NEXTREQUEST
DESCRIPTOR.message_types_by_name['TrailItem'] = _TRAILITEM
DESCRIPTOR.message_types_by_name['TrailStatus'] = _TRAILSTATUS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RpcCode = _reflection.GeneratedProtocolMessageType('RpcCode', (_message.Message,), {
  'DESCRIPTOR' : _RPCCODE,
  '__module__' : 'hypernets.dispatchers.grpc.proto.spec_pb2'
  # @@protoc_insertion_point(class_scope:hypernets.dispatchers.proto.RpcCode)
  })
_sym_db.RegisterMessage(RpcCode)

NextRequest = _reflection.GeneratedProtocolMessageType('NextRequest', (_message.Message,), {
  'DESCRIPTOR' : _NEXTREQUEST,
  '__module__' : 'hypernets.dispatchers.grpc.proto.spec_pb2'
  # @@protoc_insertion_point(class_scope:hypernets.dispatchers.proto.NextRequest)
  })
_sym_db.RegisterMessage(NextRequest)

TrailItem = _reflection.GeneratedProtocolMessageType('TrailItem', (_message.Message,), {
  'DESCRIPTOR' : _TRAILITEM,
  '__module__' : 'hypernets.dispatchers.grpc.proto.spec_pb2'
  # @@protoc_insertion_point(class_scope:hypernets.dispatchers.proto.TrailItem)
  })
_sym_db.RegisterMessage(TrailItem)

TrailStatus = _reflection.GeneratedProtocolMessageType('TrailStatus', (_message.Message,), {
  'DESCRIPTOR' : _TRAILSTATUS,
  '__module__' : 'hypernets.dispatchers.grpc.proto.spec_pb2'
  # @@protoc_insertion_point(class_scope:hypernets.dispatchers.proto.TrailStatus)
  })
_sym_db.RegisterMessage(TrailStatus)



_SEARCHDRIVER = _descriptor.ServiceDescriptor(
  name='SearchDriver',
  full_name='hypernets.dispatchers.proto.SearchDriver',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=257,
  serialized_end=455,
  methods=[
  _descriptor.MethodDescriptor(
    name='next',
    full_name='hypernets.dispatchers.proto.SearchDriver.next',
    index=0,
    containing_service=None,
    input_type=_NEXTREQUEST,
    output_type=_TRAILITEM,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='report',
    full_name='hypernets.dispatchers.proto.SearchDriver.report',
    index=1,
    containing_service=None,
    input_type=_TRAILSTATUS,
    output_type=_RPCCODE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_SEARCHDRIVER)

DESCRIPTOR.services_by_name['SearchDriver'] = _SEARCHDRIVER

# @@protoc_insertion_point(module_scope)
