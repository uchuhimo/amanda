import os

import numpy as np
import pytest
import tensorflow as tf

import amanda
from amanda.io.file import root_dir
from examples.effective_path.tensorflow.effective_path_tool import EffectivePathTool


@pytest.fixture(
    params=[
        "vgg16",
        # "vgg19",
        # "inception_v1",
        pytest.param("inception_v1", marks=pytest.mark.slow),
        # "inception_v3",
        # "resnet_v1_50",
        # "resnet_v1_152",
        # "resnet_v2_50",
        pytest.param("resnet_v2_50", marks=pytest.mark.slow),
        # "resnet_v2_101",
        # "resnet_v2_152",
        # "mobilenet_v1_1.0",
        # "mobilenet_v2_1.0_224",
        pytest.param("mobilenet_v2_1.0_224", marks=pytest.mark.slow),
        # "inception_resnet_v2",
        # "nasnet-a_large",
        # "facenet",
        # "rnn_lstm_gru_stacked",
    ]
)
def arch_name(request):
    return request.param


input_shapes = {
    "vgg16": (224, 224, 3),
    "vgg19": (224, 224, 3),
    "inception_v1": (224, 224, 3),
    "inception_v3": (299, 299, 3),
    "resnet_v1_50": (224, 224, 3),
    "resnet_v1_152": (224, 224, 3),
    "resnet_v2_50": (299, 299, 3),
    "resnet_v2_101": (299, 299, 3),
    "resnet_v2_152": (299, 299, 3),
    "resnet_v2_200": (299, 299, 3),
    "mobilenet_v1_1.0": (224, 224, 3),
    "mobilenet_v2_1.0_224": (224, 224, 3),
    "inception_resnet_v2": (299, 299, 3),
    "nasnet-a_large": (33333),
    "facenet": (160, 160, 3),
    "rnn_lstm_gru_stacked": (150),
}


def test_effective_path(arch_name):
    batch = 4
    input = np.random.rand(batch, *input_shapes[arch_name])
    model_dir = "downloads/model"
    checkpoint_dir = root_dir() / model_dir / arch_name
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"{checkpoint_dir} is not existed")
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_file is None:
        raise FileNotFoundError(
            f"cannot find checkpoint in {checkpoint_dir}, "
            f"only find: {os.listdir(checkpoint_dir)}"
        )
    tool = EffectivePathTool()
    with amanda.tool.apply(tool):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(checkpoint_file + ".meta")
                saver.restore(sess, checkpoint_file)
                sess.run("Output:0", {"input:0": input})
    tool.extract_path(entry_points=["Output"], batch=batch)
    density = tool.calc_density_per_layer()
    print(density)
