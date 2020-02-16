# type: ignore
import amanda


def modify_graph(graph: amanda.Graph):
    namespace = graph.namespace
    # convert the graph from the framework namespace to the application namespace
    graph = graph.to_namespace("debugging")
    for op in graph.ops:
        for edge in op.out_edges:
            # check whether the edge contains a valid tensor or not
            tensor = edge.tensor
            if tensor is not None and tensor.is_valid():
                # create the debug op
                debug_op = amanda.create_op(type="store_tensor_to_file")

                # change the connection from op->dst_op to op->debug_op->dst_op

                # connect op->debug_op
                new_edge1 = amanda.connect(src=op, dst=debug_op)
                # bind the tensor bound to op->dst_op to op->debug_op
                new_edge1.tensor = edge.tensor
                # connect debug_op->dst_op
                new_edge2 = amanda.connect(src=debug_op, dst=edge.dst)
                # bind the output tensor of the debug op to debug_op->dst_op
                new_edge2.tensor = debug_op.output_tensor[0]
                # disconnect op->dst_op
                amanda.disconnect(src=op, dst=edge.dst)

                # add the debug op into the graph
                graph.add_op(debug_op)
    # convert the graph from the application namespace to the framework namespace
    return graph.to_namespace(namespace)
