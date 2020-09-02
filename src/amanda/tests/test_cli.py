from click.testing import CliRunner

from amanda.cli import cli


def test_cli():
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
    context = cli.make_context("", list(args))
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
    result = runner.invoke(cli, list(args))
    assert result.exit_code == 0
