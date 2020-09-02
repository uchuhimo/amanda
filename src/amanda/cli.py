import importlib
import os
from typing import List

import click

from amanda.conversion.mmdnn import export_types as mmdnn_export_types
from amanda.conversion.mmdnn import import_types as mmdnn_import_types
from amanda.conversion.onnx import export_types as onnx_export_types
from amanda.conversion.onnx import import_types as onnx_import_types
from amanda.conversion.pytorch import export_types as pytorch_export_types
from amanda.conversion.pytorch import import_types as pytorch_import_types
from amanda.conversion.tensorflow import export_types as tf_export_types
from amanda.conversion.tensorflow import import_types as tf_import_types
from amanda.tests.download_model import (
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

import_types = {
    **tf_import_types,
    **onnx_import_types,
    **pytorch_import_types,
    **mmdnn_import_types,
}
export_types = {
    **tf_export_types,
    **onnx_export_types,
    **pytorch_export_types,
    **mmdnn_export_types,
}


def ensure_dir(path: str) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except FileExistsError:
            pass
    return path


@click.command()
@click.option(
    "--import",
    "-i",
    "import_type",
    required=True,
    type=click.Choice(list(import_types.keys())),
    help="Type of the imported model.",
)
@click.option(
    "--from",
    "-f",
    "import_path",
    required=True,
    type=click.Path(),
    help="Path of the imported model.",
)
@click.option(
    "--export",
    "-e",
    "export_type",
    required=True,
    type=click.Choice(list(export_types.keys())),
    help="Type of the exported model.",
)
@click.option(
    "--to",
    "-t",
    "export_path",
    required=True,
    type=click.Path(),
    help="Path of the exported model.",
)
@click.option(
    "--namespace",
    "-ns",
    default="",
    type=str,
    help="Namespace of the graph instrumented by the tool.",
)
@click.option(
    "--tool",
    "-T",
    "tool_name",
    default="",
    type=str,
    help="Fully qualified name of the tool.",
)
@click.argument("tool_args", nargs=-1)
def cli(
    import_type: str,
    import_path: str,
    export_type: str,
    export_path: str,
    namespace: str,
    tool_name: str,
    tool_args: List[str],
):
    if len(tool_name) != 0:
        tool_module_name, tool_class_name = tool_name.rsplit(".", 1)
        tool_module = importlib.import_module(tool_module_name)
        tool_class = getattr(tool_module, tool_class_name)
        if len(tool_args) != 0:
            tool = tool_class(tool_args)
        else:
            tool = tool_class()
    else:
        tool = None
    try:
        import_func = import_types[import_type]
        import_path = os.path.abspath(import_path)
        graph = import_func(os.path.abspath(import_path))  # type: ignore
        graph = graph.to_namespace(namespace)
        if tool is not None:
            updated_graph = tool.instrument(graph)
        else:
            updated_graph = graph
        export_func = export_types[export_type]
        export_path = ensure_dir(os.path.abspath(export_path))
        export_func(updated_graph, export_path)  # type: ignore
    finally:
        if tool is not None:
            tool.finish()


@click.group()
def download_cli():
    pass


@download_cli.command(name="tf")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Choice(["all"] + list(tf_arch_names)),
    help="Name of the model.",
)
def download_tf(model):
    if model == "all":
        download_all_tf_models()
    else:
        download_tf_model(model, model_dir="model")


@download_cli.command(name="onnx")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Choice(["all"] + list(onnx_arch_map.keys())),
    help="Name of the model.",
)
def download_onnx(model):
    if model == "all":
        download_all_onnx_models()
    else:
        download_onnx_model(model, model_dir="onnx_model")


@download_cli.command(name="all")
def download_all():
    download_all_models()


@download_cli.command(name="tflite")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Choice(["all"] + list(tflite_arch_map.keys())),
    help="Name of the model.",
)
def download_tflite(model):
    if model == "all":
        download_all_tflite_models()
    else:
        download_tflite_model(model, model_dir="tflite_model")
