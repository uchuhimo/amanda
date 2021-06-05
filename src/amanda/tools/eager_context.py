from typing import Union

import tensorflow as tf

from amanda.event import (
    EventContext,
    after_backward_op_added,
    after_backward_op_executed,
    after_graph_constructed,
    after_op_added,
    after_op_executed,
    before_backward_op_added,
    before_backward_op_executed,
    before_op_added,
    before_op_executed,
)
from amanda.tool import Tool


def before_op_executed_hook(context: EventContext):
    def hook_fn(*inputs):
        context.trigger(
            before_op_executed,
            inputs=list(inputs),
        )
        if len(inputs) == 0:
            return None
        elif len(inputs) == 1:
            return context["inputs"][0]
        else:
            return context["inputs"]

    return hook_fn


def after_op_executed_hook(context: EventContext):
    def hook_fn(*outputs):
        context.trigger(
            after_op_executed,
            outputs=list(outputs),
        )
        if len(outputs) == 0:
            return None
        elif len(outputs) == 1:
            return context["outputs"][0]
        else:
            return context["outputs"]

    return hook_fn


def before_backward_op_executed_hook(context: EventContext):
    def hook_fn(*grad_outputs):
        context.trigger(
            before_backward_op_executed,
            grad_outputs=list(grad_outputs),
        )
        if len(grad_outputs) == 0:
            return None
        elif len(grad_outputs) == 1:
            return context["grad_outputs"][0]
        else:
            return context["grad_outputs"]

    return hook_fn


def after_backward_op_executed_hook(context: EventContext):
    def hook_fn(*grad_inputs):
        context.trigger(
            after_backward_op_executed,
            grad_inputs=list(grad_inputs),
        )
        if len(grad_inputs) == 0:
            return None
        elif len(grad_inputs) == 1:
            return context["grad_inputs"][0]
        else:
            return context["grad_inputs"]

    return hook_fn


class EagerContextTool(Tool):
    def __init__(self):
        super().__init__()
        self.register_event(before_op_added, self.before_op_added)
        self.register_event(after_op_added, self.after_op_added)
        self.register_event(before_backward_op_added, self.before_backward_op_added)
        self.register_event(after_backward_op_added, self.after_backward_op_added)
        self.register_event(
            after_graph_constructed,
            self.after_graph_constructed,
        )
        self.ignored_ops = {
            "VariableV2",
            "Merge",
            "Switch",
            "AudioSummary",
            "AudioSummaryV2",
            "HistogramSummary",
            "ImageSummary",
            "MergeSummary",
            "ScalarSummary",
            "TensorSummary",
            "TensorSummaryV2",
        }

    def get_id(self) -> str:
        return "EagerContext"

    def before_op_added(self, context: EventContext):
        if "hook_before_op_added" in context and not context["hook_before_op_added"]:
            return
        op = context["op"]
        if op.type in self.ignored_ops:
            return
        input_with_index = [
            (index, input)
            for index, input in enumerate(context["inputs"])
            if not input.dtype._is_ref_dtype
        ]
        inputs = [input for _, input in input_with_index]
        input_indices = [index for index, _ in input_with_index]
        input_types = op._input_types
        input_types = [input_types[index] for index in input_indices]
        with tf.control_dependencies(op.control_inputs):
            new_inputs = tf.py_function(
                before_op_executed_hook(context),
                inputs,
                input_types,
                name=f"{op.name}_before_op_executed",
            )
        if len(inputs) == 0:
            new_op = new_inputs
            op._add_control_input(new_op)
            new_inputs = []
        for index, new_input in zip(input_indices, new_inputs):
            context["inputs"][index] = new_input

    def after_op_added(self, context: EventContext):
        if "hook_after_op_added" in context and not context["hook_after_op_added"]:
            return
        op = context["op"]
        if op.type in self.ignored_ops:
            return
        output_with_index = [
            (index, output)
            for index, output in enumerate(context["outputs"])
            if not output.dtype._is_ref_dtype
        ]
        outputs = [output for _, output in output_with_index]
        output_indices = [index for index, _ in output_with_index]
        output_types = op._output_types
        output_types = [output_types[index] for index in output_indices]
        new_outputs = tf.py_function(
            after_op_executed_hook(context),
            outputs,
            output_types,
            name=f"{op.name}_after_op_executed",
        )
        if len(outputs) == 0:
            new_op = new_outputs
            new_outputs = []
        else:
            new_op = new_outputs[0].op
        for control_output in op._control_outputs:
            control_output._add_control_input(new_op)
        if len(outputs) == 0:
            new_op._add_control_input(op)
        for index, new_output in zip(output_indices, new_outputs):
            context["outputs"][index] = new_output

    def before_backward_op_added(self, context: EventContext):
        if (
            "hook_before_backward_op_added" in context
            and not context["hook_before_backward_op_added"]
        ):
            return
        op = context["op"]
        backward_op = context["backward_op"]
        if op.type in self.ignored_ops:
            return
        input_with_index = [
            (index, input)
            for index, input in enumerate(context["grad_outputs"])
            if not input.dtype._is_ref_dtype
        ]
        inputs = [input for _, input in input_with_index]
        input_indices = [index for index, _ in input_with_index]
        input_types = backward_op._input_types
        input_types = [input_types[index] for index in input_indices]
        with tf.control_dependencies(backward_op.control_inputs):
            new_inputs = tf.py_function(
                before_backward_op_executed_hook(context),
                inputs,
                input_types,
                name=f"{backward_op.name}_before_op_executed",
            )
        if len(inputs) == 0:
            new_op = new_inputs
            backward_op._add_control_input(new_op)
            new_inputs = []
        for index, new_input in zip(input_indices, new_inputs):
            context["grad_outputs"][index] = new_input

    def after_backward_op_added(self, context: EventContext):
        if (
            "hook_after_backward_op_added" in context
            and not context["hook_after_backward_op_added"]
        ):
            return
        op = context["op"]
        backward_op = context["backward_op"]
        if op.type in self.ignored_ops:
            return
        output_with_index = [
            (index, output)
            for index, output in enumerate(context["grad_inputs"])
            if not output.dtype._is_ref_dtype
        ]
        outputs = [output for _, output in output_with_index]
        output_indices = [index for index, _ in output_with_index]
        output_types = backward_op._output_types
        output_types = [output_types[index] for index in output_indices]
        new_outputs = tf.py_function(
            after_backward_op_executed_hook(context),
            outputs,
            output_types,
            name=f"{backward_op.name}_after_op_executed",
        )
        if len(outputs) == 0:
            new_op = new_outputs
            new_outputs = []
        else:
            new_op = new_outputs[0].op
        for control_output in backward_op._control_outputs:
            control_output._add_control_input(new_op)
        if len(outputs) == 0:
            new_op._add_control_input(backward_op)
        for index, new_output in zip(output_indices, new_outputs):
            context["grad_inputs"][index] = new_output

    def after_graph_constructed(self, context: EventContext):
        def get_after_hook_op_name(op_name: str):
            hook_op_name = f"{op_name}_after_op_executed"
            if hook_op_name in graph._nodes_by_name:
                return hook_op_name
            else:
                return op_name

        def get_after_hook_tensor_name(tensor_name: str):
            op_name, index = tensor_name.split(":")
            hook_op_name = f"{op_name}_after_op_executed"
            if hook_op_name in graph._nodes_by_name:
                return f"{hook_op_name}:{index}"
            else:
                return tensor_name

        def get_after_hook(node: Union[tf.Operation, tf.Tensor]):
            name = node.name
            if ":" in name:
                return graph.get_tensor_by_name(get_after_hook_tensor_name(name))
            else:
                return graph.get_operation_by_name(get_after_hook_op_name(name))

        graph = context["graph"]
        spec = context["estimator_spec"]
        if isinstance(spec.predictions, dict):
            for name, tensor in spec.predictions.items():
                spec.predictions[name] = get_after_hook(tensor)
        elif spec.predictions is not None:
            spec = spec._replace(predictions=get_after_hook(spec.predictions))
        if spec.loss is not None:
            spec = spec._replace(loss=get_after_hook(spec.loss))
        if spec.train_op is not None:
            spec = spec._replace(train_op=get_after_hook(spec.train_op))
        for name in spec.eval_metric_ops.keys():
            spec.eval_metric_ops[name] = (
                get_after_hook(spec.eval_metric_ops[name][0]),
                get_after_hook(spec.eval_metric_ops[name][1]),
            )
        context["estimator_spec"] = spec
