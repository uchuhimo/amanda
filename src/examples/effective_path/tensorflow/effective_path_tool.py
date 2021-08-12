from typing import List

import amanda
import numpy as np
import pandas as pd
import tensorflow as tf

from examples.effective_path.graph import Graph, Op, Tensor
from examples.effective_path.tensorflow.path_extraction import (
    TraceKey,
    calc_density_per_layer,
    compact_path,
    get_ops_in_path,
    get_path,
    merge_path,
)
from examples.effective_path.utils import arg_approx


class EffectivePathTool(amanda.Tool):
    def __init__(self):
        super().__init__(namespace="amanda/tensorflow")
        self.add_inst_for_op(self.forward_instrumentation, require_outputs=True)
        self.graph = Graph()
        self.name_to_tensor = {}

    def forward_instrumentation(self, context: amanda.OpContext):
        raw_op = context.get_op()
        input_index = []
        input_names = []
        for index, input in enumerate(raw_op.inputs):
            if not input.dtype._is_ref_dtype:
                input_index.append(index)
                input_names.append(input.name)
        output_index = []
        output_names = []
        for index, output in enumerate(raw_op.outputs):
            if not output.dtype._is_ref_dtype:
                output_index.append(index)
                output_names.append(output.name)
        op = Op(raw_op=raw_op, id=raw_op.name)
        self.graph.ops[op.id] = op
        context.insert_before_op(
            self.extract_inputs,
            inputs=input_index,
            op=op,
            tensor_names=input_names,
        )
        context.insert_after_op(
            self.extract_outputs,
            outputs=output_index,
            op=op,
            tensor_names=output_names,
        )

    def extract_inputs(self, *inputs, op, tensor_names):
        def extract_fn(*np_inputs):
            op.inputs = []
            for name in tensor_names:
                tensor = self.name_to_tensor[name]
                op.inputs.append(tensor)
                tensor.outputs.append(op)
            return np_inputs

        if len(inputs) == 0:
            return
        new_inputs = tf.numpy_function(
            extract_fn,
            inputs,
            [tensor.dtype for tensor in inputs],
        )
        return new_inputs

    def extract_outputs(self, *outputs, op, tensor_names):
        def extract_fn(*np_outputs):
            op.outputs = []
            for name, value in zip(tensor_names, np_outputs):
                tensor = Tensor(value, op)
                self.graph.tensors.append(tensor)
                op.outputs.append(tensor)
                self.name_to_tensor[name] = tensor
            return np_outputs

        if len(outputs) == 0:
            return
        new_outputs = tf.numpy_function(
            extract_fn,
            outputs,
            [tensor.dtype for tensor in outputs],
        )
        return new_outputs

    def collect_parameters(self):
        for op in self.graph.ops.values():
            op_type = op.raw_op.type
            node_def = op.raw_op.node_def
            if op_type in ["Conv2D", "DepthwiseConv2dNative"]:
                op.attrs["kernel"] = op.inputs[1]
                op.attrs["kernel_size"] = np.array(op.inputs[1].value.shape)[:2]
                op.attrs["kernel"].value = np.transpose(
                    op.attrs["kernel"].value, (3, 2, 0, 1)
                )
                op.attrs["data_format"] = node_def.attr["data_format"].s.decode()
                op.attrs["strides"] = list(node_def.attr["strides"].list.i)
                op.attrs["padding"] = node_def.attr["padding"].s.decode()
                op.attrs["dilations"] = list(node_def.attr["dilations"].list.i)
            elif op_type == "MatMul":
                op.attrs["weight"] = op.inputs[1]
            elif op_type == "Transpose":
                op.attrs["perm"] = op.inputs[1]
            elif op_type in ["Concat", "ConcatV2"]:
                op.attrs["axis"] = op.inputs[-1]
            elif op_type == "Pad":
                op.attrs["paddings"] = op.inputs[1]
            elif op_type in ["MaxPool", "AvgPool"]:
                op.attrs["kernel_size"] = list(node_def.attr["ksize"].list.i)
                op.attrs["data_format"] = node_def.attr["data_format"].s.decode()
                op.attrs["strides"] = list(node_def.attr["strides"].list.i)
                op.attrs["padding"] = node_def.attr["padding"].s.decode()
            elif op_type == "Squeeze":
                op.attrs["squeeze_dims"] = list(node_def.attr["squeeze_dims"].list.i)
            elif op_type == "Mean":
                op.attrs["reduction_indices"] = op.inputs[1]
                op.attrs["keep_dims"] = node_def.attr["keep_dims"].b

            if op_type in ["MaxPool", "AvgPool", "Conv2D", "DepthwiseConv2dNative"]:
                if op.attrs["data_format"] == "NHWC":
                    op.attrs["strides"] = op.attrs["strides"][1:3]
                    if len(op.attrs["kernel_size"]) == 4:
                        op.attrs["kernel_size"] = op.attrs["kernel_size"][1:3]
                elif op.attrs["data_format"] == "NCHW":
                    op.attrs["strides"] = op.attrs["strides"][2:]
                    if len(op.attrs["kernel_size"]) == 4:
                        op.attrs["kernel_size"] = op.attrs["kernel_size"][2:]
                else:
                    raise RuntimeError(
                        op.attrs["data_format"]
                        + " is not supported value of data format"
                    )

    def extract_path(
        self,
        entry_points: List[int] = None,
        batch: int = 1,
    ) -> None:
        self.collect_parameters()
        ops_in_path = get_ops_in_path(
            self.graph,
            entry_points=entry_points,
            is_critical_op=lambda op: op.raw_op.type
            in ["Conv2D", "MatMul", "DepthwiseConv2dNative"],
        )
        paths = []
        for batch_index in range(batch):
            path = get_path(
                self.graph,
                batch_index=batch_index,
                ops_in_path=ops_in_path,
                entry_points=entry_points,
                select_fn=lambda input: arg_approx(input, 0.5),
            )
            path = compact_path(path)
            paths.append(path)
        self.path = merge_path(*paths)

    def calc_density_per_layer(self) -> pd.DataFrame:
        layers = [op.id for op in self.graph.ops.values() if TraceKey.EDGE in op.attrs]
        return calc_density_per_layer(self.graph, layers)
