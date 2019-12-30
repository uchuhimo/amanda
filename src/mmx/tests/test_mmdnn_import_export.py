import json

import google.protobuf.json_format as json_format
import pytest
import tensorflow as tf
from jsondiff import diff
from mmdnn.conversion.tensorflow.tensorflow_parser import TensorflowParser

from mmx.conversion.mmdnn import export_to_protobuf, import_from_protobuf
from mmx.tests.utils import root_dir


@pytest.fixture(
    params=[
        "vgg16",
        "vgg19",
        "inception_v1",
        "inception_v3",
        "resnet_v1_50",
        # "resnet_v1_152",
        "resnet_v2_50",
        "resnet_v2_101",
        # "resnet_v2_152",
        # "resnet_v2_200",
        "mobilenet_v1_1.0",
        "mobilenet_v2_1.0_224",
        "inception_resnet_v2",
        "nasnet-a_large",
        "facenet",
        # "rnn_lstm_gru_stacked",
    ]
)
def arch_name(request):
    return request.param


# this function tests transformation between IR of MMdnn and IR of mmx
def test_mmdnn_import_export(arch_name):
    checkpoint_dir = root_dir() / "tmp" / "model" / arch_name
    # convert downloaded model to MMdnn IR
    parser = TensorflowParser(
        tf.train.latest_checkpoint(checkpoint_dir) + ".meta",
        tf.train.latest_checkpoint(checkpoint_dir),
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
