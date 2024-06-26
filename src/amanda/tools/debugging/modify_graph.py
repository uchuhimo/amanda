# type: ignore
import amanda.tools.debugging.graph as amanda

# load all mapping tables
amanda.load_mapping_tables("tables.yaml")


def modify_graph(graph: amanda.Graph):
    namespace = graph.namespace
    # convert the graph from the framework namespace to the application namespace
    graph = graph.to_namespace("debugging")
    for op in graph.ops:
        for output_port in op.output_ports:
            for edge in output_port.out_edges:
                # create the debug op
                debug_op = amanda.create_op(type="store_tensor_to_file")
                # create the edge from output_port to debug_op's first input port
                graph.create_edge(src=output_port, dst=debug_op.input_port("0"))
                # create the edge from debug_op's first output port
                # to edge's dst input port
                graph.create_edge(src=debug_op.output_port("0"), dst=edge.dst)
                # remove the original edge
                graph.remove_edge(edge)
                # add the debug op into the graph
                graph.add_op(debug_op)
    # convert the graph from the application namespace to the framework namespace
    return graph.to_namespace(namespace)
