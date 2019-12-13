import os

import pytest
from mmdnn.conversion.common.IR import graph_pb2
from mmdnn.conversion.common.IR.IR_graph import load_protobuf_from_file

from mmx.importer import import_from_protobuf, print_graph

from pathlib import Path


def test_importer():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = Path(current_dir).parents[2] / "test_data" / "tensorflow_inception_resnet_v2_converted.pb"
    model = graph_pb2.GraphDef()
    load_protobuf_from_file(model, str(path))
    graph = import_from_protobuf(model)
    print_graph(graph)
