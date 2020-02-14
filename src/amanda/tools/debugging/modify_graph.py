import amanda


def modify_graph(graph: amanda.Graph):
    namespace = graph.namespace
    # convert the graph from the framework namespace to the application namespace
    graph = graph.to_namespace("debugging")
    for op in graph.ops:
        for edge in op.out_edges:
            # check whether the edge contains a valid tensor or not
            if "tensor" in edge.attrs:
                tensor = edge.attrs["tensor"]
                if tensor.is_valid():
                    # create the debug op
                    debug_op = amanda.create_op(type="store_tensor_to_file")

                    # change the connections from op->dst_op to op->debug_op->dst_op

                    # connect op->debug_op
                    amanda.connect(src=op, dst=debug_op)

                    # replace op->dst_op with debug_op->dst_op
                    edge.replace_src(debug_op)

                    # add the debug op into the graph
                    graph.add_op(debug_op)
    # convert the graph from the application namespace to the framework namespace
    return graph.to_namespace(namespace)
