# type: ignore
import amanda.tools.debugging.graph as amanda

# load all mapping tables
amanda.load_mapping_tables("tables.yaml")


def modify_graph(graph: amanda.Graph):
    namespace = graph.namespace
    # convert the graph from the framework namespace to the application namespace
    graph = graph.to_namespace("effective_path")

    for op in reversed(graph.sorted_ops):
        extract_op = amanda.create_op(type="Extract" + op.type)
        extract_op.name = "extract_" + op.name
        path_op = amanda.create_op(type="Path")
        path_op.name = op.name + "_path"

        graph.create_edge(
            src=path_op.output_port("ref"),
            dst=extract_op.input_port("read_path"),
        )
        graph.create_edge(
            src=extract_op.output_port("path"), dst=path_op.input_port("value")
        )

        for input_port in op.input_ports:
            graph.create_edge(
                src=input_port.in_edges[0].src,
                dst=extract_op.input_port(input_port.name),
            )
        for output_port in op.output_ports:
            graph.create_edge(
                src=output_port, dst=extract_op.input_port(output_port.name)
            )
            if len(output_port.out_edges) == 1:
                downstream_input_port = output_port.out_edges[0].dst
                downstream_extract_op = graph.get_op(
                    "extract_" + downstream_input_port.op.name
                )
                graph.create_edge(
                    src=downstream_extract_op.output_port(downstream_input_port.name),
                    dst=extract_op.input_port("extract_" + output_port.name),
                )

    # convert the graph from the application namespace to the framework namespace
    return graph.to_namespace(namespace)
