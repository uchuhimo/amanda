import json
import os
from pathlib import Path

import google.protobuf.json_format as json_format
from jsondiff import diff
from mmdnn.conversion.examples.tensorflow.extractor import tensorflow_extractor
from mmdnn.conversion.tensorflow.tensorflow_parser import TensorflowParser

from mmx.exporter import export_to_protobuf
from mmx.importer import import_from_protobuf


# this function tests transformation between IR of MMdnn and IR of mmx
def test_importer_exporter():
    # download models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cachedir = Path(current_dir).parents[2] / "test_data" / "cache"
    cachedir = str(cachedir) + "/"
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    # for a complete list of architecture name supported, see
    # mmdnn/conversion/examples/tensorflow/extractor.py
    architecture_name = "vgg16"
    tensorflow_extractor.download(architecture_name, cachedir)
    # convert downloaded model to MMdnn IR
    parser = TensorflowParser(
        cachedir + "imagenet_" + architecture_name + ".ckpt.meta",
        cachedir + "imagenet_" + architecture_name + ".ckpt",
        ["MMdnn_Output"],
    )
    parser.gen_IR()
    model = parser.IR_graph
    # check transformation between MMdnn IR and mmx IR
    graph = import_from_protobuf(model)
    new_model = export_to_protobuf(graph)
    op_dict = {}
    for node in model.node:
        op_dict[node.name] = node
    new_op_dict = {}
    for node in new_model.node:
        new_op_dict[node.name] = node
    assert len(op_dict) == len(new_op_dict)

    for name in op_dict:
        assert name in new_op_dict
        op = op_dict[name]
        json_str = json_format.MessageToJson(op, preserving_proto_field_name=True)
        json_object = json.loads(json_str)
        new_op = new_op_dict[name]
        new_json_str = json_format.MessageToJson(
            new_op, preserving_proto_field_name=True
        )
        new_json_object = json.loads(new_json_str)
        result = diff(json_object, new_json_object)
        assert result == {}
