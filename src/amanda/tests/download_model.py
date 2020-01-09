from concurrent.futures.process import ProcessPoolExecutor
from functools import partial

from mmdnn.conversion.common.utils import download_file
from mmdnn.conversion.examples.tensorflow.extractor import tensorflow_extractor

from amanda.tests.utils import root_dir


def download_tf_model(arch_name, model_dir):
    full_model_dir = root_dir() / "downloads" / model_dir
    if not full_model_dir.exists():
        full_model_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
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
                    # "rnn_lstm_gru_stacked",
                ],
            )
        )


def onnx_model_zoo(path: str) -> str:
    return f"https://s3.amazonaws.com/onnx-model-zoo/{path}"


onnx_arch_map = {
    "mobilenetv2-1.0": {
        "url": onnx_model_zoo("mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz")
    },
    "resnet18v2": {"url": onnx_model_zoo("resnet/resnet18v2/resnet18v2.tar.gz")},
}


def download_onnx_model(arch_name, model_dir):
    full_model_dir = root_dir() / "downloads" / model_dir
    if not full_model_dir.exists():
        full_model_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    download_file(
        onnx_arch_map[arch_name]["url"],
        directory=str(full_model_dir) + "/",
        auto_unzip=True,
    )


def download_all_onnx_models():
    with ProcessPoolExecutor() as executor:
        list(
            executor.map(
                partial(download_onnx_model, model_dir="onnx_model"),
                # for a complete list of architecture name supported, see
                # https://github.com/onnx/models
                ["mobilenetv2-1.0", "resnet18v2"],
            )
        )


def tflite_model_zoo(path: str) -> str:
    return f"https://storage.googleapis.com/download.tensorflow.org/models/{path}"


tflite_arch_map = {
    "mobilenet_v2_1.0_224_quant": {
        "url": tflite_model_zoo("tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz"),
    },
    "mobilenet_v2_1.0_224": {
        "url": tflite_model_zoo("tflite_11_05_08/mobilenet_v2_1.0_224.tgz"),
    },
    "nasnet_mobile": {
        "url": tflite_model_zoo(
            "tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz"
        ),
    },
}


def download_tflite_model(arch_name, model_dir):
    full_model_dir = root_dir() / "downloads" / model_dir
    if not full_model_dir.exists():
        full_model_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    download_file(
        tflite_arch_map[arch_name]["url"],
        directory=str(full_model_dir / arch_name) + "/",
        auto_unzip=True,
    )


def download_all_tflite_models():
    with ProcessPoolExecutor() as executor:
        list(
            executor.map(
                partial(download_tflite_model, model_dir="tflite_model"),
                # for a complete list of architecture name supported, see
                # https://www.tensorflow.org/lite/guide/hosted_models
                ["mobilenet_v2_1.0_224_quant", "mobilenet_v2_1.0_224", "nasnet_mobile"],
            )
        )


def download_all_models():
    download_all_tf_models()
    download_all_onnx_models()
    download_all_tflite_models()


if __name__ == "__main__":
    download_all_models()
