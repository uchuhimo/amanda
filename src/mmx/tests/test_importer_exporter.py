import json
import os
from pathlib import Path

import google.protobuf.json_format as json_format
from mmdnn.conversion.common.IR import graph_pb2
from mmdnn.conversion.common.IR.IR_graph import load_protobuf_from_file

from mmx.exporter import Exporter
from mmx.importer import Importer


# this function tests transformation between IR of MMdnn and IR of mmx
def test_importer_exporter():
    from mmdnn.conversion.examples.tensorflow.extractor import tensorflow_extractor
    from mmdnn.conversion.tensorflow.tensorflow_parser import TensorflowParser

    # download models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cachedir = Path(current_dir).parents[2] / "test_data" / "cache"
    tmpdir = Path(current_dir).parents[2] / "test_data" / "tmp"
    cachedir = str(cachedir) + "/"
    tmpdir = str(tmpdir) + "/"
    if not os.path.exists(cachedir):
        os.mkdirs(cachedir)
    if not os.path.exists(tmpdir):
        os.mkdirs(tmpdir)
    # for a complete list of architecture name supported, see
    # mmdnn/conversion/examples/tensorflow/extractor.py
    architecture_name = "vgg16"
    tensorflow_extractor.download(architecture_name, cachedir)
    # convert downloaded model to MMdnn IR
    IR_file = tmpdir + "tensorflow_" + architecture_name + "_converted"
    parser = TensorflowParser(
        cachedir + "imagenet_" + architecture_name + ".ckpt.meta",
        cachedir + "imagenet_" + architecture_name + ".ckpt",
        ["MMdnn_Output"],
    )
    parser.run(IR_file)
    del parser
    del TensorflowParser
    # check transformation between MMdnn IR and mmx IR
    path = IR_file + ".pb"
    model = graph_pb2.GraphDef()
    load_protobuf_from_file(model, str(path))
    json_str = json_format.MessageToJson(model, preserving_proto_field_name=True)
    json_object = json.loads(json_str)
    importer = Importer()
    importer.import_from_protobuf(model)
    exporter = Exporter(importer.graph, importer.op_list)
    new_model = exporter.export_to_protobuf()
    new_json_str = json_format.MessageToJson(
        new_model, preserving_proto_field_name=True
    )
    new_json_object = json.loads(new_json_str)
    # new_json_file = IR_file + "_new.json"
    # fp = open(new_json_file, "w")
    # fp.write(new_json_str)
    # fp.close()
    from jsondiff import diff

    result = diff(json_object, new_json_object)
    assert result == {}
    del importer
    del exporter
