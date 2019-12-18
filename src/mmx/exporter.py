from typing import List

from mmdnn.conversion.common.IR import graph_pb2

from mmx import Graph, Op
from mmx.importer import DataType, LiteralTensor, TensorShape


class Exporter:
    op_list: List["Op"]
    graph: Graph

    def __init__(self, graph: Graph, op_list: List["Op"]):
        self.graph = graph
        self.op_list = op_list
        self.op_unused = Op()  # this is to avoid error of flake8

    def export_to_protobuf(self):
        IR_graph = graph_pb2.GraphDef()
        for op in self.op_list:
            IR_graph_node = IR_graph.node.add()
            for port in op.inputs:
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
