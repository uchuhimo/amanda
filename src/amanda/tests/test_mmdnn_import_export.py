import pytest
import tensorflow as tf
from mmdnn.conversion.tensorflow.tensorflow_parser import TensorflowParser

from amanda.conversion.mmdnn import export_to_graph_def, import_from_protobuf
from amanda.conversion.tensorflow import diff_graph_def
from amanda.tests.utils import root_dir


@pytest.fixture(
    params=[
        "vgg16",
        # "vgg19",
        "inception_v1",
        # "inception_v3",
        # "resnet_v1_50",
        # # "resnet_v1_152",
        "resnet_v2_50",
        # "resnet_v2_101",
        # # "resnet_v2_152",
        # # "resnet_v2_200",
        # "mobilenet_v1_1.0",
        "mobilenet_v2_1.0_224",
        # "inception_resnet_v2",
        # "nasnet-a_large",
        "facenet",
        # # "rnn_lstm_gru_stacked",
    ]
)
def arch_name(request):
    return request.param


# this function tests transformation between IR of MMdnn and IR of amanda
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
    # check transformation between MMdnn IR and amanda IR
    graph = import_from_protobuf(model)
    new_model = export_to_graph_def(graph)
    assert diff_graph_def(model, new_model) == {}
