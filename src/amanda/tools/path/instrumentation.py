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
            src=path_op.output_ports["ref"], dst=extract_op.input_ports["read_path"],
        )
        graph.create_edge(
            src=extract_op.output_ports["path"], dst=path_op.input_ports["value"]
        )

        for name, input_port in op.input_ports.items():
            graph.create_edge(
                src=input_port.in_edges[0].src, dst=extract_op.input_ports[name]
            )
        for name, output_port in op.output_ports.items():
            graph.create_edge(src=output_port, dst=extract_op.input_ports[name])
            if len(output_port.out_edges) == 1:
                downstream_input_port = output_port.out_edges[0].dst
                downstream_extract_op = graph.get_op(
                    "extract_" + downstream_input_port.op.name
                )
                graph.create_edge(
                    src=downstream_extract_op.output_ports[downstream_input_port.name],
                    dst=extract_op.input_ports["extract_" + name],
                )

    # convert the graph from the application namespace to the framework namespace
    return graph.to_namespace(namespace)
