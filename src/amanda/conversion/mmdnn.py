from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Union

import numpy as np
from mmdnn.conversion.common.IR import graph_pb2

from amanda.conversion.utils import to_proto
from amanda.graph import Graph, Op
from amanda.namespace import is_qualified


def export_to_graph_def(graph: Graph) -> graph_pb2.GraphDef:
    graph_def = graph_pb2.GraphDef()
    for op in graph.ops:
        node = graph_def.node.add()
        for port in op.input_tensors:
            if port.output_index == 0:
                port_string = port.op.name
            else:
                port_string = port.op.name + ":" + str(port.output_index)
            node.input.append(port_string)

        node.op = op.type
        node.name = op.name
        # set attrs for node

        for key in without_internal_attrs(op.attrs):
            ir_value = node.attr[key]
            value = op.attrs[key]
            if type(value) == bytes:
                ir_value.s = value
            elif type(value) == int:
                ir_value.i = value
            elif type(value) == float:
                ir_value.f = value
            elif type(value) == bool:
                ir_value.b = value
            elif type(value) == DataType:
                ir_value.type = value.value
            elif type(value) == TensorShape:
                ir_value.shape.CopyFrom(value.export_to_ir())
            elif type(value) == LiteralTensor:
                ir_value.tensor.CopyFrom(value.export_to_ir())
            elif type(value) == list:
                ir_value.list.SetInParent()
                if len(value) > 0:
                    elem = value[0]
                    if type(elem) == bytes:
                        ir_value.list.s.extend(value)
                    elif type(elem) == int:
                        ir_value.list.i.extend(value)
                    elif type(elem) == float:
                        ir_value.list.f.extend(value)
                    elif type(elem) == bool:
                        ir_value.list.b.extend(value)
                    elif type(elem) == DataType:
                        ir_value.list.type.extend(value)
                    elif type(elem) == TensorShape:
                        for elem_ in value:
                            ir_value.list.shape.append(elem_.export_to_ir())
                    elif type(elem) == LiteralTensor:
                        for elem_ in value:
                            ir_value.list.tensor.append(elem_.export_to_ir())

    return graph_def


def without_internal_attrs(attrs):
    return {
        name: value
        for name, value in attrs.items()
        if not (is_qualified(name) or name in ["name", "type"])
    }


@dataclass
class MmdnnTensor:
    op: str
    output_index: int


def import_from_graph_def(
    graph_def: Union[graph_pb2.GraphDef, str, bytes, Path]
) -> Graph:
    graph_def = to_proto(graph_def, graph_pb2.GraphDef)
    graph = Graph()
    name_to_node = {node.name: node for node in graph_def.node}

    def add_op(node):
        if graph.get_op_by_name(node.name) is not None:
            return
        input_tensors: List[MmdnnTensor] = []
        input: str
        for input in node.input:
            names = input.split(":")
            assert len(names) == 1 or len(names) == 2
            if len(names) == 1:
                input_tensors.append(MmdnnTensor(names[0], 0))
            else:
                input_tensors.append(MmdnnTensor(names[0], int(names[1])))
        for input_tensor in input_tensors:
            add_op(name_to_node[input_tensor.op])
        op = Op(
            input_tensors=[
                graph.get_op_by_name(input_tensor.op).output_tensor(
                    input_tensor.output_index
                )
                for input_tensor in input_tensors
            ]
        )
        op.type = node.op
        op.name = node.name
        #  add attrs into op
        for key in node.attr:
            ir_attr_value = node.attr[key]
            field_set = ir_attr_value.WhichOneof("value")
            if field_set == "type":
                op.attrs[key] = DataType(ir_attr_value.type)
            elif field_set == "shape":
                op.attrs[key] = TensorShape(ir_attr_value.shape)
            elif field_set == "tensor":
                op.attrs[key] = LiteralTensor(ir_attr_value.tensor)
            elif field_set == "s":
                op.attrs[key] = ir_attr_value.s
            elif field_set == "i":
                op.attrs[key] = ir_attr_value.i
            elif field_set == "f":
                op.attrs[key] = ir_attr_value.f
            elif field_set == "b":
                op.attrs[key] = ir_attr_value.b
            elif field_set == "list":
                ir_list = ir_attr_value.list
                attr_value_list = []
                for value in ir_list.s:
                    attr_value_list.append(value)
                for value in ir_list.i:
                    attr_value_list.append(value)
                for value in ir_list.f:
                    attr_value_list.append(value)
                for value in ir_list.b:
                    attr_value_list.append(value)
                for value in ir_list.type:
                    attr_value_list.append(DataType(value))
                for value in ir_list.shape:
                    attr_value_list.append(TensorShape(value))
                for value in ir_list.tensor:
                    attr_value_list.append(LiteralTensor(value))
                op.attrs[key] = attr_value_list
            else:
                print("unknown field met")
                assert False

        graph.add_op(op)

    for ir_node in graph_def.node:
        add_op(ir_node)

    return graph


class DataType(Enum):
    DT_UNDEFINED = 0
    DT_INT8 = 1
    DT_INT16 = 2
    DT_INT32 = 3
    DT_INT64 = 4
    DT_UINT8 = 5
    DT_UINT16 = 6
    DT_UINT32 = 7
    DT_UINT64 = 8
    DT_FLOAT16 = 9
    DT_FLOAT32 = 10
    DT_FLOAT64 = 11
    DT_COMPLEX64 = 12
    DT_COMPLEX128 = 13
    DT_BOOL = 14
    DT_STRING = 15


@dataclass
class Dim:
    size: int
    name: str


@dataclass
class TensorShape:
    dim: List["Dim"]
    unknown_rank: bool

    def __init__(self, ir_tensor_shape: graph_pb2.TensorShape):
        self.unknown_rank = ir_tensor_shape.unknown_rank
        self.dim = []
        for dim_ in ir_tensor_shape.dim:
            self.dim.append(Dim(dim_.size, dim_.name))

    def export_to_ir(self):
        ir_shape = graph_pb2.TensorShape()
        ir_shape.unknown_rank = self.unknown_rank
        for dim_ in self.dim:
            ir_shape.dim.add(size=dim_.size, name=dim_.name)
        return ir_shape


@dataclass
class LiteralTensor:
    data_type: DataType
    tensor_shape: TensorShape
    version_number: int
    tensor_content: np.array

    def __init__(self, ir_literal_tensor: graph_pb2.LiteralTensor):
        self.tensor_shape = TensorShape(ir_literal_tensor.tensor_shape)
        self.version_number = ir_literal_tensor.version_number
        self.data_type = DataType(ir_literal_tensor.dtype)
        # fill tensor_content according to data_type
        switch = {
            DataType.DT_UNDEFINED: ir_literal_tensor.tensor_content,
            DataType.DT_INT8: ir_literal_tensor.int_val,
            DataType.DT_INT16: ir_literal_tensor.int_val,
            DataType.DT_INT32: ir_literal_tensor.int_val,
            DataType.DT_UINT8: ir_literal_tensor.uint_val,
            DataType.DT_UINT16: ir_literal_tensor.uint_val,
            DataType.DT_UINT32: ir_literal_tensor.uint_val,
            DataType.DT_INT64: ir_literal_tensor.int64_val,
            DataType.DT_UINT64: ir_literal_tensor.uint64_val,
            DataType.DT_FLOAT16: ir_literal_tensor.float_val,
            DataType.DT_FLOAT32: ir_literal_tensor.float_val,
            DataType.DT_FLOAT64: ir_literal_tensor.double_val,
            DataType.DT_COMPLEX64: ir_literal_tensor.double_val,
            DataType.DT_COMPLEX128: ir_literal_tensor.double_val,
            DataType.DT_BOOL: ir_literal_tensor.bool_val,
            DataType.DT_STRING: ir_literal_tensor.string_val,
        }
        ir_tensor_content = switch[self.data_type]
        for element in ir_tensor_content:
            self.tensor_content.append(element)

    def export_to_ir(self):
        ir_literal_tensor = graph_pb2.LiteralTensor()
        ir_literal_tensor.dtype = self.data_type
        ir_literal_tensor.tensor_shape.CopyFrom(self.tensor_shape.export_to_ir())
        ir_literal_tensor.version_number = self.version_number
        # fill tensor_content according to data_type
        switch = {
            DataType.DT_UNDEFINED: ir_literal_tensor.tensor_content,
            DataType.DT_INT8: ir_literal_tensor.int_val,
            DataType.DT_INT16: ir_literal_tensor.int_val,
            DataType.DT_INT32: ir_literal_tensor.int_val,
            DataType.DT_UINT8: ir_literal_tensor.uint_val,
            DataType.DT_UINT16: ir_literal_tensor.uint_val,
            DataType.DT_UINT32: ir_literal_tensor.uint_val,
            DataType.DT_INT64: ir_literal_tensor.int64_val,
            DataType.DT_UINT64: ir_literal_tensor.uint64_val,
            DataType.DT_FLOAT16: ir_literal_tensor.float_val,
            DataType.DT_FLOAT32: ir_literal_tensor.float_val,
            DataType.DT_FLOAT64: ir_literal_tensor.double_val,
            DataType.DT_COMPLEX64: ir_literal_tensor.double_val,
            DataType.DT_COMPLEX128: ir_literal_tensor.double_val,
            DataType.DT_BOOL: ir_literal_tensor.bool_val,
            DataType.DT_STRING: ir_literal_tensor.string_val,
        }
        ir_tensor_content = switch[self.data_type]
        for element in self.tensor_content:
            ir_tensor_content.append(element)
