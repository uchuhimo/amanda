import os
from importlib.util import find_spec
from typing import Callable, Dict, List

import click

from amanda.cli.utils import import_from_name
from amanda.event import EventContext, on_graph_loaded, update_graph
from amanda.io.file import ensure_dir
from amanda.io.file import export_types as amanda_export_types
from amanda.io.file import import_types as amanda_import_types

import_types: Dict[str, Callable] = {}
export_types: Dict[str, Callable] = {}

import_types.update(amanda_import_types)
export_types.update(amanda_export_types)

if find_spec("tensorflow"):
    from amanda.conversion.tensorflow import export_types as tf_export_types
    from amanda.conversion.tensorflow import import_types as tf_import_types

    import_types.update(tf_import_types)
    export_types.update(tf_export_types)

if find_spec("torch"):
    from amanda.conversion.pytorch import export_types as pytorch_export_types
    from amanda.conversion.pytorch import import_types as pytorch_import_types

    import_types.update(pytorch_import_types)
    export_types.update(pytorch_export_types)

if find_spec("onnx"):
    from amanda.conversion.onnx import export_types as onnx_export_types
    from amanda.conversion.onnx import import_types as onnx_import_types

    import_types.update(onnx_import_types)
    export_types.update(onnx_export_types)

if find_spec("mmdnn"):
    from amanda.conversion.mmdnn import export_types as mmdnn_export_types
    from amanda.conversion.mmdnn import import_types as mmdnn_import_types

    import_types.update(mmdnn_import_types)
    export_types.update(mmdnn_export_types)


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
        tool_class = import_from_name(tool_name)
        if len(tool_args) != 0:
            tool = tool_class(tool_args)
        else:
            tool = tool_class()
    else:
        tool = None
    import_func = import_types[import_type]
    import_path = os.path.abspath(import_path)
    graph = import_func(import_path)
    if namespace != "":
        graph = graph.to_namespace(namespace)
    updated_graph = graph
    if tool is not None:
        context = EventContext()
        context["graph"] = graph
        if tool.is_registered(on_graph_loaded):

            def update_graph_fn(context):
                nonlocal updated_graph
                updated_graph = context["new_graph"]

            context.register_event(update_graph, update_graph_fn)
            tool.get_callback(on_graph_loaded)(context)
    export_func = export_types[export_type]
    export_path = ensure_dir(os.path.abspath(export_path))
    export_func(updated_graph, export_path)


if __name__ == "__main__":
    cli()
