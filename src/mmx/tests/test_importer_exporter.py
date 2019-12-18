import os
from pathlib import Path

import google.protobuf.json_format as json_format
import json_tools
from mmdnn.conversion.common.IR import graph_pb2
from mmdnn.conversion.common.IR.IR_graph import load_protobuf_from_file

from mmx.exporter import Exporter
from mmx.importer import Importer


def test_importer_exporter():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = (
        Path(current_dir).parents[2]
        / "test_data"
        / "tensorflow_inception_resnet_v2_converted.pb"
    )
    model = graph_pb2.GraphDef()
    load_protobuf_from_file(model, str(path))
    importer = Importer()
    importer.import_from_protobuf(model)
    exporter = Exporter(importer.graph, importer.op_list)
    new_model = exporter.export_to_protobuf()
    json_str = json_format.MessageToJson(new_model, preserving_proto_field_name=True)
    new_json_str = json_format.MessageToJson(
        new_model, preserving_proto_field_name=True
    )
    result = json_tools.diff(json_str, new_json_str)
    assert result == []
