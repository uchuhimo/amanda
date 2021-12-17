import os
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from importlib.util import find_spec
from pathlib import Path

from amanda.io.file import root_dir
from loguru import logger


def tf_model_zoo(path: str) -> str:
    return f"http://download.tensorflow.org/models/{path}"


def download_tf_model(arch_name, model_dir, root=None, skip_download=False):
    if root is None:
        root = root_dir() / "downloads"
    full_model_dir = Path(root) / model_dir
    if not full_model_dir.exists():
        full_model_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    path = str(full_model_dir / arch_name) + "/"
    if not find_spec("tensorflow"):
        raise ImportError(
            "Please install TensorFlow before processing TensorFlow models."
        )
    from amanda.cli import tensorflow_extractor

    if not skip_download:
        download_file(
            tensorflow_extractor.architecture_map[arch_name]["url"],
            directory=path,
            auto_unzip=True,
        )
    if not (full_model_dir / arch_name / "checkpoint").exists():
        if "ckpt" in tensorflow_extractor.architecture_map[arch_name]["filename"]:
            tensorflow_extractor.handle_checkpoint(
                arch_name, path, skip_download=skip_download
            )


# for a complete list of architecture name supported, see
# https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models
# or
# https://github.com/microsoft/MMdnn/blob/master/mmdnn/conversion/examples/tensorflow/extractor.py
tf_arch_names = [
    "vgg16",
    "vgg19",
    "inception_v1",
    "inception_v3",
    "resnet_v1_50",
    "resnet_v1_152",
    "resnet_v2_50",
    "resnet_v2_101",
    "resnet_v2_152",
    "mobilenet_v1_1.0",
    "mobilenet_v2_1.0_224",
    "inception_resnet_v2",
    "nasnet-a_large",
    "facenet",
    "rnn_lstm_gru_stacked",
]


def download_all_tf_models(root=None, skip_download=False):
    # with ProcessPoolExecutor() as executor:
    #     list(
    #         executor.map(
    #             partial(
    #                 download_tf_model,
    #                 model_dir="model",
    #                 root=root,
    #                 skip_download=skip_download,
    #             ),
    #             list(tf_arch_names),
    #         )
    #     )
    for name in tf_arch_names:
        download_tf_model(name, model_dir="model", root=root, skip_download=False)


def onnx_model_zoo(path: str) -> str:
    return f"https://github.com/onnx/models/raw/master/{path}"


# for a complete list of architecture name supported, see
# https://github.com/onnx/models
onnx_arch_map = {
    "mobilenetv2-7": {
        "url": onnx_model_zoo(
            "vision/classification/mobilenet/model/mobilenetv2-7.tar.gz"
        )
    },
    "resnet18-v1-7": {
        "url": onnx_model_zoo("vision/classification/resnet/model/resnet18-v1-7.tar.gz")
    },
    "squeezenet1.1": {
        "url": onnx_model_zoo(
            "vision/classification/squeezenet/model/squeezenet1.1-7.tar.gz"
        ),
    },
}


def download_onnx_model(arch_name, model_dir, root=None):
    if root is None:
        root = root_dir() / "downloads"
    full_model_dir = Path(root) / model_dir
    if not full_model_dir.exists():
        full_model_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    download_file(
        onnx_arch_map[arch_name]["url"],
        directory=str(full_model_dir) + "/",
        auto_unzip=True,
    )
    if "raw_name" in onnx_arch_map[arch_name]:
        raw_path = full_model_dir / onnx_arch_map[arch_name]["raw_name"]
        raw_path.rename(full_model_dir / arch_name)


def download_all_onnx_models(root=None):
    with ProcessPoolExecutor() as executor:
        list(
            executor.map(
                partial(download_onnx_model, model_dir="onnx_model", root=root),
                list(onnx_arch_map.keys()),
            )
        )


def tflite_model_zoo(path: str) -> str:
    return f"https://storage.googleapis.com/download.tensorflow.org/models/{path}"


# for a complete list of architecture name supported, see
# https://www.tensorflow.org/lite/guide/hosted_models
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


def download_tflite_model(arch_name, model_dir, root=None):
    if root is None:
        root = root_dir() / "downloads"
    full_model_dir = Path(root) / model_dir
    if not full_model_dir.exists():
        full_model_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    download_file(
        tflite_arch_map[arch_name]["url"],
        directory=str(full_model_dir / arch_name) + "/",
        auto_unzip=True,
    )


def download_all_tflite_models(root=None):
    with ProcessPoolExecutor() as executor:
        list(
            executor.map(
                partial(download_tflite_model, model_dir="tflite_model", root=root),
                list(tflite_arch_map.keys()),
            )
        )


def download_all_models(root=None):
    with ProcessPoolExecutor() as executor:
        executor.submit(download_all_tf_models, root=root)
        executor.submit(download_all_onnx_models, root=root)
        executor.submit(download_all_tflite_models, root=root)


def _single_thread_download(url, file_name):
    from six.moves import urllib

    result, _ = urllib.request.urlretrieve(url, file_name)
    return result


def download_file(
    url, directory="./", local_fname=None, force_write=False, auto_unzip=False
):
    """Download the data from source url, unless it's already here.

    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.

    Returns:
        Path to resulting file.
    """

    if not os.path.isdir(directory):
        os.mkdir(directory)

    if not local_fname:
        k = url.rfind("/")
        local_fname = url[k + 1 :]

    local_fname = os.path.join(directory, local_fname)

    if os.path.exists(local_fname) and not force_write:
        logger.info(f"File [{local_fname}] existed!")
        return local_fname
    else:
        logger.info(f"Downloading file [{local_fname}] from [{url}]")
        try:
            import wget

            ret = wget.download(url, local_fname, bar=None)
        except Exception:
            ret = _single_thread_download(url, local_fname)
        logger.info(f"Successfully download file [{local_fname}] from [{url}]")

    if auto_unzip:
        if ret.endswith(".tar.gz") or ret.endswith(".tgz"):
            try:
                import tarfile

                tar = tarfile.open(ret)
                tar.extractall(directory)
                tar.close()
            except Exception:
                logger.exception(f"Unzip file [{ret}] failed.")

        elif ret.endswith(".zip"):
            try:
                import zipfile

                zip_ref = zipfile.ZipFile(ret, "r")
                zip_ref.extractall(directory)
                zip_ref.close()
            except Exception:
                logger.exception(f"Unzip file [{ret}] failed.")
    return ret


if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        executor.submit(download_all_tf_models, skip_download=False)
        # executor.submit(download_all_onnx_models)
