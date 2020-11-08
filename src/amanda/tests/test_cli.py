import pytest
from click.testing import CliRunner

from amanda.cli import download, main


def test_main():
    args = [
        "--import",
        "tensorflow_checkpoint",
        "--from",
        "downloads/model/vgg16/imagenet_vgg16.ckpt",
        "--export",
        "tensorflow_checkpoint",
        "--to",
        "tmp/modified_model/vgg16/imagenet_vgg16.ckpt",
        "--namespace",
        "amanda/tensorflow",
        "--tool",
        "amanda.tools.debugging.insert_debug_op_tensorflow.DebuggingTool",
    ]
    context = main.cli.make_context("", list(args))
    assert context.params == {
        "import_type": "tensorflow_checkpoint",
        "import_path": "downloads/model/vgg16/imagenet_vgg16.ckpt",
        "export_type": "tensorflow_checkpoint",
        "export_path": "tmp/modified_model/vgg16/imagenet_vgg16.ckpt",
        "namespace": "amanda/tensorflow",
        "tool_name": "amanda.tools.debugging.insert_debug_op_tensorflow.DebuggingTool",
        "tool_args": (),
    }
    runner = CliRunner()
    result = runner.invoke(main.cli, list(args))
    assert result.exit_code == 0


@pytest.mark.dependency()
def test_export_to_amanda():
    args = [
        "--import",
        "tensorflow_checkpoint",
        "--from",
        "downloads/model/vgg16/imagenet_vgg16.ckpt",
        "--export",
        "amanda_yaml",
        "--to",
        "tmp/amanda_graph/vgg16/imagenet_vgg16",
    ]
    runner = CliRunner()
    result = runner.invoke(main.cli, list(args))
    assert result.exit_code == 0


@pytest.mark.dependency(depends=["test_export_to_amanda"])
def test_import_from_amanda():
    args = [
        "--import",
        "amanda_yaml",
        "--from",
        "tmp/amanda_graph/vgg16/imagenet_vgg16",
        "--export",
        "tensorflow_checkpoint",
        "--to",
        "tmp/tf_model/vgg16/imagenet_vgg16.ckpt",
    ]
    runner = CliRunner()
    result = runner.invoke(main.cli, list(args))
    assert result.exit_code == 0


def test_download():
    args = ["tf", "--model", "vgg16", "--root-dir", "downloads"]
    runner = CliRunner()
    result = runner.invoke(download.cli, list(args))
    assert result.exit_code == 0
