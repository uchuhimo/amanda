from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import DefaultDict, List

import numpy as np

from mmx import Graph, Op, OutputPort


# The following classes are identical to graph.proto
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

    @classmethod
    def from_pb(cls, pb2_shape):
        return cls(dim=pb2_shape.dim, unknown_rank=pb2_shape.unknown_rank)


@dataclass
class LiteralTensor:
    dtype: DataType
    tensor_shape: TensorShape
    version_number: int
    tensor_content: str
    val: np.array
    int_val: List[int]
    uint_val: List[int]
    int64_val: List[int]
    uint64_val: List[int]
    float_val: List[float]
    double_val: List[float]
    bool_val: List[bool]
    string_val: List[str]


@dataclass
class ListValue:
    s: List[str]
    i: List[int]
    f: List[float]
    b: List[bool]
    type_val: List["DataType"]
    shape: List["TensorShape"]
    tensor: List["LiteralTensor"]


class AttrValue:
    # TODO: This is a oneof class in protobuf, only one of the following will be set
    field_set: str
    list_val: List["ListValue"]
    s: str
    i: int
    f: float
    b: bool
    type_val: DataType
    shape: TensorShape
    tensor: LiteralTensor


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
        # TODO: add attrs into op
        graph.add(op)

    # second pass: set the inputs of ops
    for op in graph.ops:
        for input_string in inputs_dict[op]:
            input_port_tmp = input_string.split(":")
            input_op = name_to_op[input_port_tmp[0]]
            if len(input_port_tmp) < 2:
                input_op_output_index = 0
            else:
                input_op_output_index = int(input_port_tmp[1])
            input_port = OutputPort(input_op, input_op_output_index)
            op.inputs.append(input_port)

    return graph


def print_graph(graph: Graph):
    for op in graph.ops:
        for input_op in op.input_ops:
            print("input: ", input_op.name)
        print("op: ", op.type)
        print("name: ", op.name)
        print("")
        # TODO: print attrs
    return
