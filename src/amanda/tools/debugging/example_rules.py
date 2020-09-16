# type: ignore
import amanda.tools.debugging.graph as amanda


def downgrade_batch_norm_op(graph: amanda.Graph):
    for op in graph.ops:
        if op.type == "FusedBatchNormV2":
            op.type = "FusedBatchNorm"


def assign_tensor_name(graph: amanda.Graph):
    for op in graph.ops:
        for output_port in op.output_ports:
            tensor = output_port.tensor
            tensor.attrs["name"] = op.name + ":" + output_port.name


def constant_folding(graph: amanda.Graph):
    for op in graph.ops:
        if (
            op.type == "Add"
            and op.input_ops[0].type == "Const"
            and op.input_ops[1].type == "Const"
        ):
            const1 = op.input_ops[0]
            const2 = op.input_ops[1]
            new_const = amanda.create_op(type="Const")
            new_value = const1.attrs["value"] + const2.attrs["value"]
            new_const.attrs["value"] = new_value
            graph.add_op(new_const)
            graph.remove_op(const1)
            graph.remove_op(const2)
            graph.remove_op(op)
            amanda.remove_edge(
                amanda.get_edge(src=const1.output_port(0), dst=op.input_port(0))
            )
            amanda.remove_edge(
                amanda.get_edge(src=const2.output_port(0), dst=op.input_port(1))
            )
            for output_port in op.output_ports:
                for out_edge in output_port.out_edges:
                    amanda.create_edge(
                        src=new_const.output_port(output_port.name), dst=out_edge.dst
                    )
                    amanda.remove_edge(out_edge)


def upgrade_op_version(graph: amanda.Graph):
    for op in graph.ops:
        if op.attrs["version"] == "v7":
            op.attrs["version"] = "v10"


def mark_fused_op(graph: amanda.Graph):
    for op in graph.ops:
        op.attrs["is_fused"] = op.type.startswith("Fused")


def unmark_fused_op(graph: amanda.Graph):
    for op in graph.ops:
        if "is_fused" in op.attrs:
            del op.attrs["is_fused"]
