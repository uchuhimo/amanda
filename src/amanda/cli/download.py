import os

import click

from amanda.cli.download_model import (
    download_all_models,
    download_all_onnx_models,
    download_all_tf_models,
    download_all_tflite_models,
    download_onnx_model,
    download_tf_model,
    download_tflite_model,
    onnx_arch_map,
    tf_arch_names,
    tflite_arch_map,
)


@click.group()
def cli():
    pass


@cli.command(name="tf")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Choice(["all"] + list(tf_arch_names)),
    help="Name of the model.",
)
@click.option(
    "--root-dir",
    "-r",
    required=True,
    type=click.Path(),
    help="Root directory of all the downloaded models.",
)
def download_tf(model, root_dir):
    root_dir = os.path.abspath(root_dir)
    if model == "all":
        download_all_tf_models(root=root_dir)
    else:
        download_tf_model(model, model_dir="model", root=root_dir)


@cli.command(name="onnx")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Choice(["all"] + list(onnx_arch_map.keys())),
    help="Name of the model.",
)
@click.option(
    "--root-dir",
    "-r",
    required=True,
    type=click.Path(),
    help="Root directory of all the downloaded models.",
)
def download_onnx(model, root_dir):
    root_dir = os.path.abspath(root_dir)
    if model == "all":
        download_all_onnx_models(root=root_dir)
    else:
        download_onnx_model(model, model_dir="onnx_model", root=root_dir)


@cli.command(name="all")
@click.option(
    "--root-dir",
    "-r",
    required=True,
    type=click.Path(),
    help="Root directory of all the downloaded models.",
)
def download_all(root_dir):
    root_dir = os.path.abspath(root_dir)
    download_all_models(root=root_dir)


@cli.command(name="tflite")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Choice(["all"] + list(tflite_arch_map.keys())),
    help="Name of the model.",
)
@click.option(
    "--root-dir",
    "-r",
    required=True,
    type=click.Path(),
    help="Root directory of all the downloaded models.",
)
def download_tflite(model, root_dir):
    root_dir = os.path.abspath(root_dir)
    if model == "all":
        download_all_tflite_models(root=root_dir)
    else:
        download_tflite_model(model, model_dir="tflite_model", root=root_dir)


if __name__ == "__main__":
    cli()
