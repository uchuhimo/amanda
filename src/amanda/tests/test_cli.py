from click.testing import CliRunner

from amanda.cli import cli


def test_cli():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
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
            "amanda.tools.debugging.insert_debug_op_tensorflow.modify_graph",
        ],
    )
    assert result.exit_code == 0
