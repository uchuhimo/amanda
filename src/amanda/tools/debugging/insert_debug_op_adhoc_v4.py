import amanda
import amanda.matcher
from amanda import Namespace

debugging_namespace = Namespace("debugging")


def map_pytorch_to_debugging(src_op_list, dst_op_list):
    op = dst_op_list[0]
    op.attrs["is_valid"] = [
        tensor_type.kind() == "TensorType" for tensor_type in op.attrs["output_types"]
    ]


def map_tf_to_debugging(src_op_list, dst_op_list):
    op = dst_op_list[0]
    op.attrs["is_valid"] = [
        not tensor_dtype._is_ref_dtype for tensor_dtype in op.attrs["output_dtypes"]
    ]


def map_debugging_to_tf(src_op_list, dst_op_list):
    op = dst_op_list[0]
    edge = op.input_edges[0]
    op.name = f"debug/{edge.src_op.name}/{edge.src_output_index}"
    op.attrs["T"] = edge.src_op.attrs["output_dtypes"][edge.src_output_index]


def map_debugging_to_pytorch(src_op_list, dst_op_list):
    op = dst_op_list[0]
    op.attrs["output_types"] = [op.input_edges[0].src_op.attrs["output_types"][0]]


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
            if op.attrs["is_valid"][edge.src_output_index]:
                # create the debug op
                debug_op = amanda.create_op(type="store_tensor_to_file")
                # insert the debug op
                edge.insert_op(debug_op)
                # add the debug op into the graph
                graph.add_op(debug_op)
    # map the graph from the application namespace to the framework namespace
    return graph.to_namespace(namespace)
