from torch._C import TensorType

import amanda
from amanda import Graph, Namespace, Op, get_global_registry
from amanda.conversion.pytorch import pytorch_namespace
from amanda.conversion.tensorflow import tf_namespace
from amanda.rule import OpMapping, RawRuleMapper, Rule

debugging_namespace = Namespace("debugging")


class FromPyTorchRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        for output_tensor in op.output_tensors:
            output_tensor.attrs["is_tensor"] = (
                output_tensor.attrs["type"].kind() == "TensorType"
            )
            output_tensor.attrs["is_ref"] = False
        return OpMapping(source_ops=[op], target_ops=[op])


get_global_registry().add_mapper(
    pytorch_namespace(), debugging_namespace, RawRuleMapper(rules=[FromPyTorchRule()])
)
get_global_registry().add_mapper(
    debugging_namespace, pytorch_namespace(), RawRuleMapper(rules=[])
)


class FromTensorFlowRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        for output_tensor in op.output_tensors:
            output_tensor.attrs["is_tensor"] = True
            output_tensor.attrs["is_ref"] = output_tensor.attrs["dtype"]._is_ref_dtype
            output_tensor.attrs["type"] = TensorType.get()
        return OpMapping(source_ops=[op], target_ops=[op])


class ToTensorFlowRule(Rule):
    def apply(self, graph: Graph, op: Op) -> OpMapping:
        if op.attrs["type"] == "amanda::store_tensor_to_file":
            op.attrs["type"] = "StoreTensorToFile"
            input_tensor = op.input_tensors[0]
            op.attrs[
                "name"
            ] = f"debug/{input_tensor.op.attrs['name']}/{input_tensor.output_index}"
            op.attrs["T"] = input_tensor.attrs["dtype"]
        return OpMapping(source_ops=[op], target_ops=[op])


get_global_registry().add_mapper(
    tf_namespace(), debugging_namespace, RawRuleMapper(rules=[FromTensorFlowRule()])
)
get_global_registry().add_mapper(
    debugging_namespace, tf_namespace(), RawRuleMapper(rules=[ToTensorFlowRule()])
)


def modify_graph(graph: amanda.Graph):
    namespace = graph.namespace
    graph = graph.to_namespace(debugging_namespace)
    for op in graph.ops:
        for tensor in op.output_tensors:
            if tensor.attrs["is_tensor"] and not tensor.attrs["is_ref"]:
                debug_op = amanda.create_op(
                    attrs={"type": "amanda::store_tensor_to_file"},
                    input_tensors=[tensor],
                    control_dependencies=[],
                    output_num=1,
                )
                debug_op.output_tensors[0].attrs["type"] = tensor.attrs["type"]

                for output_op in graph.ops:
                    for index, input_tensor in enumerate(output_op.input_tensors):
                        if tensor == input_tensor:
                            output_op.update_input_tensor(
                                index, debug_op.output_tensors[0]
                            )
                graph.add_op(debug_op)
    return graph.to_namespace(namespace)
