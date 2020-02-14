import amanda
import amanda.matcher
from amanda import Namespace

debugging_namespace = Namespace("debugging")

mapper = amanda.get_mapper(amanda.pytorch.pytorch_namespace(), debugging_namespace)
mapper.insert_rule(
    src_op="*",
    src_attr_name=[],
    src_attr_value=[],
    dst_op="*",
    dst_attr_name="is_valid",
    dst_value=amanda.exp(
        """
        [tensor_type.kind() == "TensorType" for tensor_type in op.attrs["output_types"]
        """
    ),
)

mapper = amanda.get_mapper(amanda.tensorflow.tf_namespace(), debugging_namespace)
mapper.insert_rule(
    src_op="*",
    src_attr_name=[],
    src_attr_value=[],
    dst_op="*",
    dst_attr_name="is_valid",
    dst_value=amanda.exp(
        """
        [not tensor_dtype._is_ref_dtype for tensor_dtype in op.attrs["output_dtypes"]]
        """
    ),
)

mapper = amanda.get_mapper(debugging_namespace, amanda.tensorflow.tf_namespace())
mapper.insert_rule(
    src_op="store_tensor_to_file",
    src_attr_name=[],
    src_attr_value=[],
    dst_op="StoreTensorToFile",
    dst_attr_name=["name", "T"],
    dst_value=[
        amanda.exp(
            """
            f"debug/{op.input_edges[0].src_op.name}/{op.input_edges[0].src_output_index}"
            """
        ),
        amanda.exp(
            """
            op.input_edges[0].src_op.attrs["output_dtypes"][op.input_edges[0].src_output_index]
            """
        ),
    ],
)

mapper = amanda.get_mapper(debugging_namespace, amanda.pytorch.pytorch_namespace())
mapper.insert_rule(
    src_op="store_tensor_to_file",
    src_attr_name=[],
    src_attr_value=[],
    dst_op="amanda::store_tensor_to_file",
    dst_attr_name="output_types",
    dst_value=amanda.exp(
        """
        [op.input_edges[0].src_op.attrs["output_types"][0]]
        """
    ),
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
