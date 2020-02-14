import amanda
import amanda.matcher
from amanda import Namespace

debugging_namespace = Namespace("debugging")


def map_pytorch_to_debugging(src_op_list, dst_op_list):
    op = dst_op_list[0]
    for edge in op.output_edges:
        edge.attrs["is_valid"] = (
            not edge.is_control_edge() and edge.attrs["type"].kind() == "TensorType"
        )


def map_tf_to_debugging(src_op_list, dst_op_list):
    op = dst_op_list[0]
    for edge in op.output_edges:
        edge.attrs["is_valid"] = (
            not edge.is_control_edge() and not edge.attrs["dtype"]._is_ref_dtype
        )


def map_debugging_to_tf(src_op_list, dst_op_list):
    op = dst_op_list[0]
    input_edge = op.input_edges[0]
    op.name = f"debug/{input_edge.src_op.name}/{input_edge.src_output_index}"
    op.attrs["T"] = input_edge.attrs["dtype"]


def map_debugging_to_pytorch(src_op_list, dst_op_list):
    op = dst_op_list[0]
    op.output_edges[0].attrs["type"] = op.input_edges[0].attrs["type"]


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
    # map the graph from the framework namespace to the application namespace
    graph = graph.to_namespace(debugging_namespace)
    for op in graph.ops:
        for edge in op.output_edges:
            # check whether the edge represents a valid tensor or not
            if "tensor" in edge.attrs and edge.attrs["tensor"]["is_valid"]:
                # create the debug op
                debug_op = amanda.create_op(type="store_tensor_to_file")
                # insert the debug op
                edge.insert_op(debug_op)
                # add the debug op into the graph
                graph.add_op(debug_op)
    # map the graph from the application namespace to the framework namespace
    return graph.to_namespace(namespace)
