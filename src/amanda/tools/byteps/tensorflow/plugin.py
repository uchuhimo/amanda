from dataclasses import dataclass, field
from typing import Dict, List

from amanda import Graph, InputPort, Op, Tensor
from amanda.attributes import Attributes


@dataclass
class VariableInfo:
    grad_op: Op
    apply_op: Op
    variable_name: str


priorities: Dict[str, int] = {}


def rewrite_graph(graph: Graph):
    variables: Dict[str, VariableInfo] = {}

    # find gradients
    for op in graph.ops:
        if op.type == "ApplyGradientDescent":
            var = op.input_op(0)
            grad = op.input_op(2)
            info = VariableInfo(variable_name=var.name, apply_op=op, grad_op=grad)
            variables[var.name] = info

    # assign priority to gradients, smaller sequence number has higher priority
    for op in reversed(graph.sorted_ops):
        if op.type in ["Variable", "VariableV2"] and op.name in variables:
            grad_op_name = variables[op.name].grad_op.name
            priorities[grad_op_name] = len(priorities)


@dataclass
class WorkerRewriteTask:
    send_op: Op
    grad_op: Op
    grad_tensor: Tensor = None
    recv_ports: List[InputPort] = field(default_factory=list)


def rewrite_worker(partition_graphs: Dict[str, Graph]):
    for partition_name, graph in partition_graphs.items():
        if "worker" in partition_name:
            tasks: List[WorkerRewriteTask] = []

            # find send ops
            for op in graph.ops:
                if op.type in ["_Send", "_HostSend"]:
                    send = op
                    grad = send.input_op(0)
                    if grad.name in priorities:
                        task = WorkerRewriteTask(send_op=send, grad_op=grad)
                        tasks.append(task)

            # find downstream ops of send ops
            for edge in graph.edges:
                for task in tasks:
                    if edge.src_op == task.send_op:
                        task.recv_ports.append(edge.dst_port)
                    elif edge.src_op == task.grad_op and edge.dst_op == task.send_op:
                        task.grad_tensor = edge.src_tensor

            # replace send ops
            for task in tasks:
                send_op = task.send_op
                op_attrs = send_op.attrs
                op = Op(
                    input_tensors=[task.grad_tensor],
                    attrs=Attributes(
                        name=send_op.name,
                        type="SendGradient",
                        T=op_attrs["T"],
                        gradient_name=task.grad_op.name,
                        tensor_name=op_attrs["tensor_name"],
                        send_device=op_attrs["send_device"],
                        recv_device=op_attrs["recv_device"],
                        send_device_incarnation=op_attrs["send_device_incarnation"],
                    ),
                )
                graph.add_op(op)
                for recv_port in task.recv_ports:
                    recv_op = recv_port.op
                    if recv_port.is_control():
                        recv_op.remove_control_dependency(send_op)
                        recv_op.add_control_dependency(op)
                    else:
                        recv_op.input_tensors[
                            recv_port.input_index
                        ] = op.output_tensor()
                graph.remove_op(send_op)

            # find recv ops
            recv_ops: Dict[Op, List[InputPort]] = {}
            for op in graph.ops:
                if op.type in ["_Recv", "_HostRecv"]:
                    recv_ops[op] = []

            # find downstream ops of recv ops
            for edge in graph.edges:
                if edge.src_op.type in ["_Recv", "_HostRecv"]:
                    recv = edge.src_op
                    if recv in recv_ops:
                        recv_ops[recv].append(edge.dst_port)

            # replace recv ops
            for recv_op, dst_ports in recv_ops.items():
                op_attrs = recv_op.attrs
                op = Op(
                    attrs=Attributes(
                        name=recv_op.name,
                        type="RecvParameter",
                        tensor_type=op_attrs["tensor_type"],
                        tensor_name=op_attrs["tensor_name"],
                        send_device=op_attrs["send_device"],
                        recv_device=op_attrs["recv_device"],
                        send_device_incarnation=op_attrs["send_device_incarnation"],
                    )
                )
                graph.add_op(op)
                for dst_port in dst_ports:
                    if dst_port.is_control():
                        dst_port.op.remove_control_dependency(recv_op)
                        dst_port.op.add_control_dependency(op)
                    else:
                        dst_port.op.input_tensors[
                            dst_port.input_index
                        ] = op.output_tensor()
                graph.remove_op(recv_op)


@dataclass
class PSRewriteTask:
    variable_name: str
    update_op: Op
    recv_op: Op
    var_op: Op = None
    send_op: Op = None
    update_dst_ports: List[InputPort] = field(default_factory=list)
    recv_ports: List[InputPort] = field(default_factory=list)


def rewrite_ps(partition_graphs: Dict[str, Graph]):
    task_map: Dict[str, PSRewriteTask] = {}
    for partition_name, graph in partition_graphs.items():
        if "ps" in partition_name:

            # find update ops and recv ops
            for op in graph.ops:
                if op.type == "ApplyGradientDescent":
                    var_op = op.input_op(0)
                    assert var_op.type in ["Variable", "VariableV2"]
                    grad = op.input_op(2)
                    assert grad.type in ["_Recv", "_HostRecv"]
                    task = PSRewriteTask(
                        variable_name=var_op.name, update_op=op, recv_op=grad,
                    )
                    task_map[var_op.name] = task

            # find downstream ops of update/recv ops
            for edge in graph.edges:
                for task in task_map.values():
                    if edge.src_op == task.update_op:
                        task.update_dst_ports.append(edge.dst_port)
                    elif edge.src_op == task.send_op:
                        task.recv_ports.append(edge.dst_port)

            # find send ops
            for op in graph.ops:
                if op.type in ["_Send", "_HostSend"]:
                    var_op = op.input_op(0)
                    if var_op.name in task_map:
                        task = task_map[var_op.name]
                        task.var_op = var_op
                        task.send_op = op

            # replace send/recv/update ops
            for task in task_map.values():
                # replace recv/update ops with fused recv+update ops
                update_op = task.update_op
                var = update_op.input_tensors[0]
                lr = update_op.input_tensors[1]
                op_attrs = update_op.attrs
                fused_op = Op(
                    input_tensors=[var, lr],
                    attrs=Attributes(
                        name=update_op.name,
                        type="RecvApplyGradientDescent",
                        T=op_attrs["T"],
                        variable_name=task.variable_name,
                        tensor_name=op_attrs["tensor_name"],
                        send_device=op_attrs["send_device"],
                        recv_device=op_attrs["recv_device"],
                        send_device_incarnation=op_attrs["send_device_incarnation"],
                    ),
                )
                graph.add_op(fused_op)
                for update_dst_port in task.update_dst_ports:
                    if update_dst_port.is_control():
                        update_dst_port.op.remove_control_dependency(update_op)
                        update_dst_port.op.add_control_dependency(fused_op)
                    else:
                        update_dst_port.op.input_tensors[
                            update_dst_port.input_index
                        ] = fused_op.output_tensor()
                graph.remove_op(update_op)
                graph.remove_op(task.recv_op)

                # replace send ops
                attrs = task.send_op.attrs
                op = Op(
                    input_tensors=[task.var_op.output_tensor()],
                    attrs=Attributes(
                        name=task.send_op.name,
                        type="SendParameter",
                        T=attrs["T"],
                        variable_name=task.variable_name,
                        tensor_name=attrs["tensor_name"],
                        send_device=attrs["send_device"],
                        recv_device=attrs["recv_device"],
                        send_device_incarnation=attrs["send_device_incarnation"],
                    ),
                )
                graph.add_op(op)
                for recv_port in task.recv_ports:
                    if recv_port.is_control():
                        recv_port.op.remove_control_dependency(task.send_op)
                        recv_port.op.add_control_dependency(op)
                    else:
                        recv_port.op.input_tensors[
                            recv_port.input_index
                        ] = op.output_tensor()
                graph.remove_op(task.send_op)
