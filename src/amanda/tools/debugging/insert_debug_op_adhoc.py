from collections import OrderedDict
from pathlib import Path

import amanda
from amanda import Graph, Namespace, Op, get_global_registry
from amanda.conversion.pytorch import pytorch_namespace
from amanda.conversion.tf import tf_namespace
from amanda.io.file import root_dir
from amanda.rule import OpMapping, Rule, RuleMapper

debugging_namespace = Namespace("debugging")


class FromPyTorchRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        for output_port in op.output_ports:
            op.attrs[f"output_port/{output_port.name}/is_tensor"] = (
                output_port.type.raw.kind() == "TensorType"
            )
        return OpMapping(source_ops=[op], target_ops=[op])


class ToPyTorchRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        for output_port in op.output_ports:
            op.attrs.pop(f"output_port/{output_port.name}/is_tensor", "")
        return OpMapping(source_ops=[op], target_ops=[op])


get_global_registry().add_mapper(
    pytorch_namespace(), debugging_namespace, RuleMapper(rules=[FromPyTorchRule()])
)
get_global_registry().add_mapper(
    debugging_namespace, pytorch_namespace(), RuleMapper(rules=[ToPyTorchRule()])
)


class FromTensorFlowRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        for output_port in op.output_ports:
            op.attrs[
                f"output_port/{output_port.name}/is_tensor"
            ] = not output_port.type.raw._is_ref_dtype
        return OpMapping(source_ops=[op], target_ops=[op])


arch_name = "vgg16"
store_dir = root_dir() / "tmp" / "debug_info_adhoc" / arch_name

if not Path(store_dir).exists():
    Path(store_dir).mkdir(mode=0o755, parents=True, exist_ok=True)


class ToTensorFlowRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        if op.type == "amanda::store_tensor_to_file":
            output_port = op.output_port(0)
            op.type = "StoreTensorToFile"
            op.attrs["store_dir"] = str(store_dir)
            op.attrs["file_name"] = f"{op.name}_{output_port.name}".replace("/", "_")
            op.attrs["T"] = output_port.type.raw
        for output_port in op.output_ports:
            op.attrs.pop(f"output_port/{output_port.name}/is_tensor", "")
        return OpMapping(source_ops=[op], target_ops=[op])


get_global_registry().add_mapper(
    tf_namespace(), debugging_namespace, RuleMapper(rules=[FromTensorFlowRule()])
)
get_global_registry().add_mapper(
    debugging_namespace, tf_namespace(), RuleMapper(rules=[ToTensorFlowRule()])
)


def modify_graph(graph: amanda.Graph):
    namespace = graph.namespace
    graph = graph.to_namespace(debugging_namespace)
    for op in graph.ops:
        for output_port in op.output_ports:
            if op.attrs[f"output_port/{output_port.name}/is_tensor"]:
                debug_op = amanda.create_op(
                    name=f"debug/{op.name}/{output_port.name}",
                    type="amanda::store_tensor_to_file",
                    inputs=OrderedDict([("0", output_port.type)]),
                    outputs=OrderedDict([("0", output_port.type)]),
                )
                edges = output_port.out_edges
                graph.add_op(debug_op)
                graph.create_edge(output_port, debug_op.input_port(0))
                for edge in edges:
                    graph.create_edge(debug_op.output_port(0), edge.dst)
                    graph.remove_edge(edge)
    return graph.to_namespace(namespace)
