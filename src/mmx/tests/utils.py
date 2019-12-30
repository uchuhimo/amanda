import os
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from mmdnn.conversion.examples.tensorflow.extractor import tensorflow_extractor


def root_dir() -> Path:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return Path(current_dir).parents[2]


def download_tf_model(arch_name, model_dir):
    full_model_dir = root_dir() / "tmp" / model_dir
    if not full_model_dir.exists():
        full_model_dir.mkdir(mode=0o755, parents=True)
    if not (full_model_dir / arch_name / "checkpoint").exists():
        tensorflow_extractor.download(arch_name, str(full_model_dir / arch_name) + "/")


def download_all_tf_models():
    with ProcessPoolExecutor() as executor:
        list(
            executor.map(
                partial(download_tf_model, model_dir="model"),
                # for a complete list of architecture name supported, see
                # mmdnn/conversion/examples/tensorflow/extractor.py
                [
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
                    "rnn_lstm_gru_stacked",
                ],
            )
        )
