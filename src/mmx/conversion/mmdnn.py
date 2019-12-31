from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import DefaultDict, List

import numpy as np
from mmdnn.conversion.common.IR import graph_pb2

from mmx.graph import Graph, Op, Tensor


def export_to_protobuf(graph: Graph):
    IR_graph = graph_pb2.GraphDef()
    for op in graph.ops:
        IR_graph_node = IR_graph.node.add()
        for port in op.input_tensors:
            if port.output_index == 0:
                port_string = port.op.name
            else:
                port_string = port.op.name + ":" + str(port.output_index)
            IR_graph_node.input.append(port_string)

        IR_graph_node.op = op.type
        IR_graph_node.name = op.name
        # set attrs for IR_graph_node

        for key in op.attrs:
            if key == "name" or key == "type":
                continue
            IR_value = IR_graph_node.attr[key]
            value = op.attrs[key]
            if type(value) == bytes:
                IR_value.s = value
            elif type(value) == int:
                IR_value.i = value
            elif type(value) == float:
                IR_value.f = value
            elif type(value) == bool:
                IR_value.b = value
            elif type(value) == DataType:
                IR_value.type = value.value
            elif type(value) == TensorShape:
                IR_value.shape.CopyFrom(value.export_to_IR())
            elif type(value) == LiteralTensor:
                IR_value.tensor.CopyFrom(value.export_to_IR())
            elif type(value) == list:
                IR_value.list.SetInParent()
                if len(value) > 0:
                    elem = value[0]
                    if type(elem) == bytes:
                        IR_value.list.s.extend(value)
                    elif type(elem) == int:
                        IR_value.list.i.extend(value)
                    elif type(elem) == float:
                        IR_value.list.f.extend(value)
                    elif type(elem) == bool:
                        IR_value.list.b.extend(value)
                    elif type(elem) == DataType:
                        IR_value.list.type.extend(value)
                    elif type(elem) == TensorShape:
                        for elem_ in value:
                            IR_value.list.shape.append(elem_.export_to_IR())
                    elif type(elem) == LiteralTensor:
                        for elem_ in value:
                            IR_value.list.tensor.append(elem_.export_to_IR())

    return IR_graph


def import_from_protobuf(model) -> Graph:
    IR_graph = model
    graph = Graph()
    name_to_op = {}
    inputs_dict: DefaultDict[Op, list] = defaultdict(list)
    for IR_node in IR_graph.node:
        op = Op()
        op.type = IR_node.op
        op.name = IR_node.name
        name_to_op[op.name] = op
        for input_string in IR_node.input:
            inputs_dict[op].append(input_string)
        #  add attrs into op
        for key in IR_node.attr:
            IR_attr_value = IR_node.attr[key]
            field_set = IR_attr_value.WhichOneof("value")
            if field_set == "type":
                op.attrs[key] = DataType(IR_attr_value.type)
            elif field_set == "shape":
                op.attrs[key] = TensorShape(IR_attr_value.shape)
            elif field_set == "tensor":
                op.attrs[key] = LiteralTensor(IR_attr_value.tensor)
            elif field_set == "s":
                op.attrs[key] = IR_attr_value.s
            elif field_set == "i":
                op.attrs[key] = IR_attr_value.i
            elif field_set == "f":
                op.attrs[key] = IR_attr_value.f
            elif field_set == "b":
                op.attrs[key] = IR_attr_value.b
            elif field_set == "list":
                IR_list = IR_attr_value.list
                attr_value_list = []
                for value in IR_list.s:
                    attr_value_list.append(value)
                for value in IR_list.i:
                    attr_value_list.append(value)
                for value in IR_list.f:
                    attr_value_list.append(value)
                for value in IR_list.b:
                    attr_value_list.append(value)
                for value in IR_list.type:
                    attr_value_list.append(DataType(value))
                for value in IR_list.shape:
                    attr_value_list.append(TensorShape(value))
                for value in IR_list.tensor:
                    attr_value_list.append(LiteralTensor(value))
                op.attrs[key] = attr_value_list
            else:
                print("unknown field met")
                assert False

        graph.add_op(op)

    # second pass: set the inputs of ops
    for op in graph.ops:
        for input_string in inputs_dict[op]:
            input_port_tmp = input_string.split(":")
            input_op = name_to_op[input_port_tmp[0]]
            if len(input_port_tmp) < 2:
                input_op_output_index = 0
            else:
                input_op_output_index = int(input_port_tmp[1])
            input_port = Tensor(input_op, input_op_output_index)
            op.input_tensors.append(input_port)

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

    def __init__(self, IR_tensor_shape: graph_pb2.TensorShape):
        self.unknown_rank = IR_tensor_shape.unknown_rank
        self.dim = []
        for dim_ in IR_tensor_shape.dim:
            self.dim.append(Dim(dim_.size, dim_.name))

    def export_to_IR(self):
        IR_shape = graph_pb2.TensorShape()
        IR_shape.unknown_rank = self.unknown_rank
        for dim_ in self.dim:
            IR_shape.dim.add(size=dim_.size, name=dim_.name)
        return IR_shape


@dataclass
class LiteralTensor:
    data_type: DataType
    tensor_shape: TensorShape
    version_number: int
    tensor_content: np.array

    def __init__(self, IR_literal_tensor: graph_pb2.LiteralTensor):
        self.tensor_shape = TensorShape(IR_literal_tensor.tensor_shape)
        self.version_number = IR_literal_tensor.version_number
        self.data_type = DataType(IR_literal_tensor.dtype)
        # fill tensor_content according to data_type
        switch = {
            DataType.DT_UNDEFINED: IR_literal_tensor.tensor_content,
            DataType.DT_INT8: IR_literal_tensor.int_val,
            DataType.DT_INT16: IR_literal_tensor.int_val,
            DataType.DT_INT32: IR_literal_tensor.int_val,
            DataType.DT_UINT8: IR_literal_tensor.uint_val,
            DataType.DT_UINT16: IR_literal_tensor.uint_val,
            DataType.DT_UINT32: IR_literal_tensor.uint_val,
            DataType.DT_INT64: IR_literal_tensor.int64_val,
            DataType.DT_UINT64: IR_literal_tensor.uint64_val,
            DataType.DT_FLOAT16: IR_literal_tensor.float_val,
            DataType.DT_FLOAT32: IR_literal_tensor.float_val,
            DataType.DT_FLOAT64: IR_literal_tensor.double_val,
            DataType.DT_COMPLEX64: IR_literal_tensor.double_val,
            DataType.DT_COMPLEX128: IR_literal_tensor.double_val,
            DataType.DT_BOOL: IR_literal_tensor.bool_val,
            DataType.DT_STRING: IR_literal_tensor.string_val,
        }
        IR_tensor_content = switch[self.data_type]
        for element in IR_tensor_content:
            self.tensor_content.append(element)

    def export_to_IR(self):
        IR_literal_tensor = graph_pb2.LiteralTensor()
        IR_literal_tensor.dtype = self.data_type
        IR_literal_tensor.tensor_shape.CopyFrom(self.tensor_shape.export_to_IR())
        IR_literal_tensor.version_number = self.version_number
        # fill tensor_content according to data_type
        switch = {
            DataType.DT_UNDEFINED: IR_literal_tensor.tensor_content,
            DataType.DT_INT8: IR_literal_tensor.int_val,
            DataType.DT_INT16: IR_literal_tensor.int_val,
            DataType.DT_INT32: IR_literal_tensor.int_val,
            DataType.DT_UINT8: IR_literal_tensor.uint_val,
            DataType.DT_UINT16: IR_literal_tensor.uint_val,
            DataType.DT_UINT32: IR_literal_tensor.uint_val,
            DataType.DT_INT64: IR_literal_tensor.int64_val,
            DataType.DT_UINT64: IR_literal_tensor.uint64_val,
            DataType.DT_FLOAT16: IR_literal_tensor.float_val,
            DataType.DT_FLOAT32: IR_literal_tensor.float_val,
            DataType.DT_FLOAT64: IR_literal_tensor.double_val,
            DataType.DT_COMPLEX64: IR_literal_tensor.double_val,
            DataType.DT_COMPLEX128: IR_literal_tensor.double_val,
            DataType.DT_BOOL: IR_literal_tensor.bool_val,
            DataType.DT_STRING: IR_literal_tensor.string_val,
        }
        IR_tensor_content = switch[self.data_type]
        for element in self.tensor_content:
            IR_tensor_content.append(element)
