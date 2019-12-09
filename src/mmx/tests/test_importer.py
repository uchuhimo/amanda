from mmdnn.conversion.common.IR import graph_pb2
from mmdnn.conversion.common.IR.IR_graph import load_protobuf_from_file

from mmx.importer import import_from_protobuf, print_graph


def test_importer():
    path = "../../../test_data/tensorflow_inception_resnet_v2_converted.pb"
    model = graph_pb2.GraphDef()
    load_protobuf_from_file(model, path)
    graph = import_from_protobuf(model)
    print_graph(graph)
