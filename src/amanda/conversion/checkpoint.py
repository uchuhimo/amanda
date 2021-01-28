import os
from dataclasses import dataclass

from amanda.adapter import Adapter, get_adapter_registry
from amanda.cli.main import export_types, import_types
from amanda.event import EventContext, on_graph_loaded


@dataclass
class Checkpoint:
    type: str
    path: str


class CheckpointAdapter(Adapter):
    def __init__(self):
        super(CheckpointAdapter, self).__init__(namespace=None)

    def apply(self, target: Checkpoint, context: EventContext) -> None:
        import_func = import_types[target.type]
        path = os.path.abspath(target.path)
        graph = import_func(path)
        context.trigger(on_graph_loaded, graph=graph)
        updated_graph = context["graph"]
        export_func = export_types[target.type]
        export_func(updated_graph, path)


get_adapter_registry().register_adapter(Checkpoint, CheckpointAdapter())
