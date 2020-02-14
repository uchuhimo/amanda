from torch._C import TensorType

import amanda
import amanda.matcher
from amanda import Namespace

debugging_namespace = Namespace("debugging")


def map_pytorch_to_debugging(op_list):
    op = op_list[0]
    for tensor in op.output_tensors:
        tensor.attrs["is_tensor"] = tensor.attrs["type"].kind() == "TensorType"
        tensor.attrs["is_ref"] = False


def map_tf_to_debugging(op_list):
    op = op_list[0]
    for tensor in op.output_tensors:
        tensor.attrs["is_tensor"] = True
        tensor.attrs["is_ref"] = tensor.attrs["dtype"]._is_ref_dtype
        tensor.attrs["type"] = TensorType.get()


def map_debugging_to_tf(op_list):
    op = op_list[0]
    input_tensor = op.input_tensors[0]
    op.name = f"debug/{input_tensor.op.name}/{input_tensor.output_index}"
    op.attrs["T"] = input_tensor.attrs["dtype"]


def map_debugging_to_pytorch(op_list):
    op = op_list[0]
    op.output_tensors[0].attrs["type"] = op.input_tensors[0].attrs["type"]


amanda.get_mapper(amanda.pytorch.pytorch_namespace(), debugging_namespace).add_rule(
    src_op_list=["*"], dst_op_list=["*"], map_func=map_pytorch_to_debugging
)

amanda.get_mapper(amanda.tensorflow.tf_namespace(), debugging_namespace).add_rule(
    src_op_list=["*"], dst_op_list=["*"], map_func=map_tf_to_debugging
)

amanda.get_mapper(debugging_namespace, amanda.tensorflow.tf_namespace()).add_rule(
    src_op_list=["store_tensor_to_file"],
    dst_op_list=["StoreTensorToFile"],
    map_func=map_debugging_to_tf,
)

amanda.get_mapper(debugging_namespace, amanda.pytorch.pytorch_namespace()).add_rule(
    src_op_list=["store_tensor_to_file"],
    dst_op_list=["amanda::store_tensor_to_file"],
    map_func=map_debugging_to_pytorch,
)


def modify_graph(graph: amanda.Graph):
    namespace = graph.namespace
    graph = graph.to_namespace(debugging_namespace)
    for op in graph.ops:
        for tensor in op.output_tensors:
            if tensor.attrs["is_tensor"] and not tensor.attrs["is_ref"]:
                debug_op = amanda.create_op(type="store_tensor_to_file")
                debug_op.input_tensors[0] = tensor
                graph.add_op(debug_op)
                for output_op, index in tensor.outputs:
                    output_op.input_tensors[index] = debug_op.output_tensors[0]
    return graph.to_namespace(namespace)
