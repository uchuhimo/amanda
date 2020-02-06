import amanda
import amanda.marker


@amanda.marker.dispatch
def is_valid(tensor: amanda.Tensor) -> bool:
    return True


@amanda.marker.dispatch
def set_attrs(debug_op: amanda.Op, tensor: amanda.Tensor):
    pass


@amanda.marker.instrumentation
def modify_graph(graph: amanda.Graph):
    for op in graph.ops:
        for tensor in op.output_tensors:
            if is_valid(tensor):
                debug_op = amanda.create_op(
                    input_tensors=[tensor], control_dependencies=[], output_num=1,
                )
                set_attrs(debug_op, tensor)

                for output_op in graph.ops:
                    for index, input_tensor in enumerate(output_op.input_tensors):
                        if tensor == input_tensor:
                            output_op.update_input_tensor(
                                index, debug_op.output_tensors[0]
                            )
                graph.add_op(debug_op)
