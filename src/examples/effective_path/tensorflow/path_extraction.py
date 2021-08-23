import gc
from functools import partial, reduce
from typing import Any, Callable, Dict, List, Optional, Set

import jax

# import jax.numpy as np
import jax.numpy as jnp
import numpy as np
import numpy as nnp
import pandas as pd

from examples.effective_path.graph import Graph, Op, Tensor
from examples.effective_path.utils import (
    argtopk,
    filter_not_null,
    index_update,
    repeat,
    to_np_array,
)

_trace_func_by_op: Dict[str, Callable[..., None]] = {}

EPS = np.finfo(np.float16).eps


class TraceKey:
    OP_TYPE = "trace.op_type"

    PATH = "trace.path"
    POINT = "trace.point"
    EDGE = "trace.edge"
    WEIGHT = "trace.weight"
    TRIVIAL = "trace.trivial"
    WEIGHTED_INPUT = "trace.weighted_input"
    FLIP_SIGN = "trace.flip_sign"
    FLIP_SIGN_CONFLICT = "trace.flip_sign_conflict"

    POINT_MASK = "trace.point_mask"
    EDGE_MASK = "trace.edge_mask"
    WEIGHT_MASK = "trace.weight_mask"
    FLIP_SIGN_MASK = "trace.flip_sign_mask"

    POINT_SHAPE = "trace.point_shape"
    EDGE_SHAPE = "trace.edge_shape"
    WEIGHT_SHAPE = "trace.weight_shape"

    POINT_NUM = "trace.point_num"
    EDGE_NUM = "trace.edge_num"
    WEIGHT_NUM = "trace.weight_num"

    MAX_POINT_NUM = "trace.max_point_num"
    MAX_EDGE_NUM = "trace.max_edge_num"
    MAX_WEIGHT_NUM = "trace.max_weight_num"

    MIN_POINT_NUM = "trace.min_point_num"
    MIN_EDGE_NUM = "trace.min_edge_num"
    MIN_WEIGHT_NUM = "trace.min_weight_num"

    COUNT = "trace.count"
    DATA_FORMAT = "trace.data_format"
    ENTRY_POINTS = "trace.entry_points"

    OUTPUT_DENSITY = "trace.output_density"
    OUTPUT_THRESHOLD = "trace.output_threshold"
    RECEPTIVE_FIELD_DENSITY = "trace.receptive_field_density"
    RECEPTIVE_FIELD_THRESHOLD = "trace.receptive_field_threshold"

    META = {OP_TYPE, TRIVIAL, POINT_SHAPE, EDGE_SHAPE, WEIGHT_SHAPE, DATA_FORMAT}
    STATISTICS = {
        POINT_NUM,
        EDGE_NUM,
        WEIGHT_NUM,
        MAX_POINT_NUM,
        MAX_EDGE_NUM,
        MAX_WEIGHT_NUM,
        MIN_POINT_NUM,
        MIN_EDGE_NUM,
        MIN_WEIGHT_NUM,
        COUNT,
    }
    METRICS = {
        OUTPUT_DENSITY,
        OUTPUT_THRESHOLD,
        RECEPTIVE_FIELD_DENSITY,
        RECEPTIVE_FIELD_THRESHOLD,
    }

    _base_keys = [POINT, EDGE, WEIGHT]

    @staticmethod
    def num_of(key: str) -> str:
        assert key in TraceKey._base_keys
        return key + "_num"

    @staticmethod
    def shape_of(key: str) -> str:
        assert key in TraceKey._base_keys
        return key + "_shape"

    @staticmethod
    def min_of(key: str) -> str:
        return key.replace("trace.", "trace.min_")

    @staticmethod
    def max_of(key: str) -> str:
        return key.replace("trace.", "trace.max_")

    @staticmethod
    def to_array(x, compact: bool = False):
        if isinstance(x, pd.DataFrame):
            return x.index.values
        elif isinstance(x, np.ndarray):
            return TraceKey.from_bitmap(x) if compact else x
        else:
            raise RuntimeError(f"expect array or data frame, get {type(x)}")

    @staticmethod
    def to_frame(x, compact: bool = False):
        if isinstance(x, pd.DataFrame):
            return x
        elif isinstance(x, np.ndarray):
            x = TraceKey.from_bitmap(x) if compact else x
            return pd.DataFrame(dict(count=np.ones(x.size, dtype=np.int)), index=x)
        else:
            raise RuntimeError(f"expect array or data frame, get {type(x)}")

    @staticmethod
    def to_bitmap(x, shape, compact: bool = False):
        if compact:
            return x
        else:
            mask = np.zeros(np.prod(shape), dtype=np.int8)
            mask[TraceKey.to_array(x)] = 1
            return np.packbits(mask)

    @staticmethod
    def to_mask(x, shape, compact: bool = False):
        if compact:
            mask = np.unpackbits(x)
        else:
            mask = np.zeros(np.prod(shape), dtype=np.int8)
            mask[TraceKey.to_array(x)] = 1
        return mask.reshape(shape)

    @staticmethod
    def from_bitmap(x):
        return np.nonzero(np.unpackbits(x))[0]

    @staticmethod
    def is_trivial(op: Op):
        return TraceKey.TRIVIAL in op.attrs and op.attrs[TraceKey.TRIVIAL]


# def to_mask(x, shape):
#     mask = np.zeros(nnp.prod(shape), dtype=np.bool_)
#     return np.packbits(index_update(mask, x, True, inplace=True))


@partial(jax.jit, static_argnums=1, backend="gpu")
def to_mask(x, shape):
    mask = jnp.zeros(nnp.prod(shape), dtype=np.bool_)
    return jnp.packbits(jax.ops.index_update(mask, x, True))


def from_mask(x):
    return np.nonzero(np.unpackbits(x))[0]


def get_point(tensor):
    if TraceKey.POINT not in tensor.attrs:
        tensor.attrs[TraceKey.POINT] = from_mask(tensor.attrs[TraceKey.POINT_MASK])
    return tensor.attrs[TraceKey.POINT]


def get_flip_sign(tensor):
    if TraceKey.FLIP_SIGN not in tensor.attrs:
        if TraceKey.FLIP_SIGN_MASK not in tensor.attrs:
            tensor.attrs[TraceKey.FLIP_SIGN] = None
        else:
            point = get_point(tensor)
            flip_sign_mask = tensor.attrs[TraceKey.FLIP_SIGN_MASK]
            flip_sign = index_update(
                np.ones((len(point),), np.int32),
                flip_sign_mask[point],
                -1,
                inplace=True,
            )
            tensor.attrs[TraceKey.FLIP_SIGN] = flip_sign
    return tensor.attrs[TraceKey.FLIP_SIGN]


def compact_path(graph: Graph, key: str = None) -> Graph:
    if graph is None:
        return graph
    if key is not None:
        attr_names = [key + "_mask", key + "_shape", TraceKey.OP_TYPE]
    else:
        attr_names = [
            TraceKey.POINT_MASK,
            TraceKey.WEIGHT_MASK,
            TraceKey.EDGE_MASK,
            TraceKey.POINT_SHAPE,
            TraceKey.WEIGHT_SHAPE,
            TraceKey.EDGE_SHAPE,
            TraceKey.OP_TYPE,
        ]
    for op in graph.ops.values():
        attrs = op.attrs
        for attr_name in list(attrs.keys()):
            if attr_name not in attr_names:
                del attrs[attr_name]
    for tensor in graph.tensors:
        tensor.value = None
    gc.collect()
    return graph


def merge_path(*paths: Graph, format: str = "bitmap") -> Optional[Graph]:
    def merge_attr(attr_name: str, attrs: List[Any]) -> Any:
        if attr_name in [TraceKey.POINT_MASK, TraceKey.EDGE_MASK, TraceKey.WEIGHT_MASK]:

            def merge_with_count(attr1, attr2):
                merged_attr = pd.concat(
                    [TraceKey.to_frame(attr1), TraceKey.to_frame(attr2)]
                )
                return merged_attr.groupby(merged_attr.index).sum()

            if format == "sparse":
                return reduce(merge_with_count, attrs)
            elif format == "mask":
                return reduce(jnp.logical_or, attrs)
            elif format == "bitmap":
                return reduce(jnp.bitwise_or, attrs)
            elif format == "bitmap_intersect":
                return reduce(np.bitwise_and, attrs)
            elif format == "bitmap_diff":
                return np.bitwise_xor(attrs[0], np.bitwise_and(attrs[0], attrs[1]))
            elif format == "bitmap_xor":
                return np.bitwise_xor(attrs[0], attrs[1])
            elif format == "array":
                return reduce(np.union1d, map(TraceKey.to_array, attrs))
            elif format == "array_intersect":
                return reduce(np.intersect1d, map(TraceKey.to_array, attrs))
            elif format == "array_xor":
                return np.setxor1d(
                    TraceKey.to_array(attrs[0]), TraceKey.to_array(attrs[1])
                )
            else:
                raise RuntimeError(f"unsupported format {format}")
        elif attr_name in [
            TraceKey.POINT_SHAPE,
            TraceKey.EDGE_SHAPE,
            TraceKey.WEIGHT_SHAPE,
            TraceKey.OP_TYPE,
            "kernel",
            "kernel_size",
            "data_format",
            "strides",
            "padding",
            "dilations",
            "weight",
            "perm",
            "axis",
            "squeeze_dims",
            "reduction_indices",
            "keep_dims",
        ]:
            return attrs[0]
        elif attr_name in [
            TraceKey.POINT_NUM,
            TraceKey.EDGE_NUM,
            TraceKey.WEIGHT_NUM,
            TraceKey.COUNT,
            TraceKey.FLIP_SIGN_CONFLICT,
        ]:
            return sum(attrs)
        elif attr_name in [
            TraceKey.MAX_EDGE_NUM,
            TraceKey.MAX_POINT_NUM,
            TraceKey.MAX_WEIGHT_NUM,
        ]:
            return max(attrs)
        elif attr_name in [
            TraceKey.MIN_EDGE_NUM,
            TraceKey.MIN_POINT_NUM,
            TraceKey.MIN_WEIGHT_NUM,
        ]:
            return min(attrs)
        elif attr_name == TraceKey.TRIVIAL:
            return True
        else:
            return None

    paths = list(filter_not_null(paths))  # type: ignore
    if len(paths) == 0:
        return None
    if len(paths) == 1:
        return paths[0]
    first_path = paths[0]
    merged_path = first_path.clone()
    merged_path.attrs = {
        attr_name: merge_attr(
            attr_name,
            [path.attrs[attr_name] for path in paths],
        )
        for attr_name in merged_path.attrs
        if nnp.all([attr_name in path.attrs for path in paths])
    }
    for op_id, op in merged_path.ops.items():
        op.attrs = {
            attr_name: merge_attr(
                attr_name,
                [path.ops[op_id].attrs[attr_name] for path in paths],
            )
            for attr_name in op.attrs
            if nnp.all([attr_name in path.ops[op_id].attrs for path in paths])
        }
    for index, tensor in enumerate(merged_path.tensors):
        tensor.attrs = {
            attr_name: merge_attr(
                attr_name,
                [path.tensors[index].attrs[attr_name] for path in paths],
            )
            for attr_name in first_path.tensors[index].attrs
            if nnp.all([attr_name in path.tensors[index].attrs for path in paths])
        }
    return merged_path


def calc_density_per_layer(
    graph: Graph, layers: List[str] = None, key: str = TraceKey.EDGE
) -> pd.DataFrame:
    layers = layers or list(graph.ops.keys())
    result_layers = []
    densities = []
    for layer_name in layers:
        op = graph.ops[layer_name]
        if key + "_mask" in op.attrs:
            result_layers.append(layer_name)
            mask = np.unpackbits(np.array(op.attrs[key + "_mask"]))
            densities.append(np.count_nonzero(mask) / nnp.prod(mask.shape))
    return pd.DataFrame(dict(density=densities), index=result_layers).rename_axis(
        "layer"
    )


def register_op(op_type: str, trace_func: Callable[..., None]):
    _trace_func_by_op[op_type] = trace_func


def get_ops_in_path(
    graph: Graph,
    entry_points: List[int],
    is_critical_op: Callable[[Op], bool] = None,
):
    def search_downstream_ops(op: Op, ops):
        ops.add(op)
        for output in op.outputs:
            for output_op in output.outputs:
                if output_op not in ops:
                    search_downstream_ops(output_op, ops)

    def search_uptream_ops(op: Op, ops_in_path):
        ops_in_path.add(op.id)
        for input in op.inputs:
            input_op = input.op
            if input_op.id not in ops_in_path and input_op in ops:
                search_uptream_ops(input_op, ops_in_path)

    ops: Set[Op]
    if is_critical_op is not None:
        ops = set()
        for op in graph.ops.values():
            if is_critical_op(op) and op not in ops:
                search_downstream_ops(op, ops)
    else:
        ops = set(graph.ops.values())

    ops_in_path: Set[str] = set()
    for entry_point in entry_points:
        op = graph.ops[entry_point]
        search_uptream_ops(op, ops_in_path)
    return ops_in_path


def get_unsupported_ops(graph, ops):
    return {
        graph.ops[op_id].attrs[TraceKey.OP_TYPE]
        for op_id in ops
        if graph.ops[op_id].attrs[TraceKey.OP_TYPE] not in _trace_func_by_op
    }


def get_path(
    graph: Graph,
    batch_index: int,
    ops_in_path: Set[str],
    entry_points: List[int],
    select_fn: Callable[[np.ndarray], np.ndarray],
    select_seed_fn: Callable[[np.ndarray], np.ndarray] = None,
    stop_hook: Callable[[Op], bool] = None,
) -> Graph:
    unsupported_ops = get_unsupported_ops(graph, ops_in_path)
    if len(unsupported_ops) != 0:
        raise RuntimeError(f"The following ops are unsupported: {unsupported_ops}")
    select_seed_fn = select_seed_fn or (lambda output: argtopk(output, 1))
    stop_hook = stop_hook or (lambda op: False)
    tensor_to_wait_count = {
        id(tensor): len(
            [output for output in tensor.outputs if output.id in ops_in_path]
        )
        for tensor in graph.tensors
    }
    op_to_wait_count = {
        op_id: len(
            [
                output
                for output in graph.ops[op_id].outputs
                if tensor_to_wait_count[id(output)] != 0
            ]
        )
        for op_id in ops_in_path
    }
    for entry_point in entry_points:
        output_op = graph.ops[entry_point]
        output_tensor = output_op.outputs[0]
        output = output_tensor.value[batch_index]
        output_tensor.attrs[TraceKey.POINT_MASK] = to_mask(
            select_seed_fn(output), output.shape
        )
        output_tensor.attrs[TraceKey.POINT_SHAPE] = output.shape
        output_tensor.attrs[TraceKey.FLIP_SIGN] = None
    ready_ops = list(entry_points)
    while len(ready_ops) != 0:
        ready_op_id = ready_ops.pop()
        ready_op = graph.ops[ready_op_id]
        if ready_op.id not in ops_in_path:
            continue
        _trace_func_by_op[ready_op.attrs[TraceKey.OP_TYPE]](
            ready_op,
            select_fn=select_fn,
            batch_index=batch_index,
        )
        if stop_hook(ready_op):
            break
        for input_tensor in ready_op.inputs:
            tensor_to_wait_count[id(input_tensor)] = (
                tensor_to_wait_count[id(input_tensor)] - 1
            )
            if tensor_to_wait_count[id(input_tensor)] == 0:
                tensor_to_wait_count.pop(id(input_tensor))
                if input_tensor.op is not None:
                    input_op = input_tensor.op
                    if input_op.id not in ops_in_path:
                        continue
                    op_to_wait_count[input_op.id] = op_to_wait_count[input_op.id] - 1
                    if op_to_wait_count[input_op.id] == 0:
                        op_to_wait_count.pop(input_op.id)
                        ready_ops.append(input_op.id)
    return graph


def early_stop_hook(layer_num: int):
    current_op_count = 0

    def stop_hook(op: Op):
        nonlocal current_op_count
        if op.attrs[TraceKey.OP_TYPE] in ["MatMul", "Conv2D"]:
            current_op_count += 1
        if current_op_count >= layer_num:
            current_op_count = 0
            return True
        else:
            return False

    return stop_hook


def merge_traced_points(
    tensor: Tensor,
    op: Op,
    traced_points: np.ndarray,
    flip_sign: Optional[np.ndarray],
):
    assert flip_sign is None
    op_index = tensor.outputs.index(op)
    tensor.attrs[TraceKey.POINT_SHAPE] = tensor.value.shape[1:]
    traced_points_mask = to_mask(traced_points, tensor.value.shape)
    tensor.attrs[TraceKey.POINT_MASK + "." + str(op_index)] = traced_points_mask
    if TraceKey.POINT_MASK in tensor.attrs:
        tensor.attrs[TraceKey.POINT_MASK] = np.bitwise_or(
            traced_points_mask, tensor.attrs[TraceKey.POINT_MASK]
        )
    else:
        tensor.attrs[TraceKey.POINT_MASK] = traced_points_mask
    if flip_sign is not None:
        flip_sign_mask = to_mask(traced_points[flip_sign < 0], tensor.value.shape)
        tensor.attrs[TraceKey.FLIP_SIGN_MASK + "." + str(op_index)] = flip_sign_mask
        if TraceKey.FLIP_SIGN_MASK in tensor.attrs:
            tensor.attrs[TraceKey.FLIP_SIGN_MASK] = np.bitwise_or(
                flip_sign_mask, tensor.attrs[TraceKey.FLIP_SIGN_MASK]
            )
        else:
            tensor.attrs[TraceKey.FLIP_SIGN_MASK] = flip_sign_mask


def calc_padding(
    input_shape: np.ndarray,
    output_shape: np.ndarray,
    stride: np.ndarray,
    kernel_size: np.ndarray,
) -> np.ndarray:
    margin = ((output_shape - 1) * stride) + np.array(kernel_size) - input_shape
    margin = np.maximum(margin, 0)
    padding = [[0, 0]]
    for i in [0, 1]:
        if margin[i] % 2 == 0:
            padding.append(np.array([margin[i] // 2, margin[i] // 2]))
        else:
            padding.append(np.array([(margin[i] - 1) // 2, (margin[i] + 1) // 2]))
    return np.array(padding)


def calc_flip_sign(
    flip_sign,
    index,
    output_value,
    weighted_input,
    select_fn,
    flip_sign_inputs: List[Any],
    return_threshold: bool = False,
):
    if flip_sign is not None:
        flipped_output_value = output_value * flip_sign[index]
    else:
        flipped_output_value = output_value
    if flipped_output_value < 0:
        flipped_weighed_input = -weighted_input
    else:
        flipped_weighed_input = weighted_input
    input_points = select_fn(flipped_weighed_input)
    if flip_sign is not None:
        if flip_sign[index] == -1 and flipped_weighed_input.max() < 0:
            new_flip_sign = -1
        else:
            new_flip_sign = 1
        flip_sign_inputs.append(repeat(new_flip_sign, input_points.size))
    if return_threshold:
        return (
            input_points,
            float(np.min(flipped_weighed_input.flatten()[input_points])),
        )
    else:
        return input_points


def calc_flip_sign_batch(
    flip_sign,
    output_value,
    weighted_input,
    select_fn,
):
    if flip_sign is not None:
        flipped_output_value = output_value * flip_sign
    else:
        flipped_output_value = output_value
    flipped = flipped_output_value < 0
    flipped_weighed_input = index_update(
        weighted_input,
        flipped,
        -weighted_input[flipped],
        inplace=True,
    )
    input_points = select_fn(flipped_weighed_input)
    if flip_sign is not None and len(input_points[0]) != 0:
        flip_sign_inputs = np.ones((len(input_points[0]),), np.int32)
        flip_sign_inputs = index_update(
            flip_sign_inputs,
            np.logical_and(flip_sign == -1, flipped_weighed_input.max(axis=-1) < 0)[
                input_points[0]
            ],
            -1,
            inplace=True,
        )
    else:
        flip_sign_inputs = None
    return input_points, flip_sign_inputs


def linear_layer_trace(
    op: Op,
    select_fn: Callable[[np.ndarray], np.ndarray],
    batch_index: int,
):
    weight = op.attrs["weight"].value
    weight = np.transpose(weight, (1, 0))
    input_tensor: Tensor = op.inputs[0]
    input = input_tensor.value[batch_index]
    output_tensor: Tensor = op.outputs[0]
    output: np.ndarray = output_tensor.value[batch_index]
    output_points = get_point(output_tensor)
    flip_sign = get_flip_sign(output_tensor)
    weighted_input = weight[output_points] * input
    output_value = output[output_points]
    input_points, flip_sign_inputs = calc_flip_sign_batch(
        flip_sign=flip_sign,
        output_value=output_value,
        weighted_input=weighted_input,
        select_fn=select_fn,
    )
    output_trace_points = output_points[input_points[0]]
    input_trace_points = input_points[1]
    edge_shape = (input.size, output.size)
    weight_shape = weight.shape
    edge_path = np.ravel_multi_index(
        (input_trace_points, output_trace_points), edge_shape
    )
    weight_path = np.ravel_multi_index(
        (output_trace_points, input_trace_points), weight_shape
    )
    op.attrs[TraceKey.EDGE_SHAPE] = edge_shape
    op.attrs[TraceKey.WEIGHT_SHAPE] = weight_shape
    op.attrs[TraceKey.EDGE_MASK] = to_mask(edge_path, edge_shape)
    op.attrs[TraceKey.WEIGHT_MASK] = to_mask(weight_path, weight_shape)
    merge_traced_points(
        input_tensor,
        op,
        input_trace_points,
        flip_sign=flip_sign_inputs,
    )


register_op("MatMul", linear_layer_trace)


def max_layer_trace(
    op: Op,
    select_fn: Callable[[np.ndarray], np.ndarray],
    batch_index: int,
):
    kernel_size = op.attrs["kernel_size"]
    stride = np.array(op.attrs["strides"])
    input_tensor: Tensor = op.inputs[0]
    input = input_tensor.value[batch_index]
    output_tensor: Tensor = op.outputs[0]
    output = output_tensor.value[batch_index]
    output_points = get_point(output_tensor)
    flip_sign = get_flip_sign(output_tensor)
    if op.attrs["data_format"] == "NHWC":
        input = np.rollaxis(input, 2)
        output = np.rollaxis(output, 2)

    padding = calc_padding(
        np.array(input.shape)[1:], np.array(output.shape)[1:], stride, kernel_size
    )
    padded_input = np.pad(input, padding, mode="constant")

    output_point_index_tuple = np.unravel_index(output_points, output.shape)
    output_point_index = np.array(output_point_index_tuple)  # [out_ch + h + w, out]
    index = output_point_index[1:]  # [h + w, out]
    start_index = index * stride[:, np.newaxis]  # [h + w, out]
    receptive_field = padded_input[
        output_point_index[0][:, np.newaxis, np.newaxis],
        start_index[0][:, np.newaxis, np.newaxis]
        + np.arange(kernel_size[0])[np.newaxis, :, np.newaxis],
        start_index[1][:, np.newaxis, np.newaxis]
        + np.arange(kernel_size[1])[np.newaxis, np.newaxis, :],
    ]  # (out, kh, kw)
    if receptive_field.size != 0:
        unaligned_max_input = np.nonzero(
            np.logical_and(
                receptive_field
                == np.nanmax(receptive_field, axis=(1, 2))[:, np.newaxis, np.newaxis],
                receptive_field > 0,
            )
        )  # [out + kh + kw, rf]
    else:
        unaligned_max_input = np.nonzero(receptive_field > 0)
    max_input = (
        output_point_index[0][unaligned_max_input[0]],
        unaligned_max_input[1] + start_index[0][unaligned_max_input[0]] - padding[1][0],
        unaligned_max_input[2] + start_index[1][unaligned_max_input[0]] - padding[2][0],
    )  # [in_ch + h + w, in]
    output_trace_points = output_points[unaligned_max_input[0]]
    input_trace_points = np.ravel_multi_index(max_input, input.shape)
    unaligned_input_trace_points = np.ravel_multi_index(
        unaligned_max_input[1:], kernel_size
    )
    if flip_sign is not None:
        flip_sign_inputs = flip_sign[unaligned_max_input[0]]
    else:
        flip_sign_inputs = None
    edge_shape = kernel_size + output.shape
    edge_path = np.ravel_multi_index(
        (unaligned_input_trace_points, output_trace_points),
        (kernel_size[0] * kernel_size[1], output.size),
    )
    op.attrs[TraceKey.EDGE_SHAPE] = edge_shape
    op.attrs[TraceKey.EDGE_MASK] = to_mask(edge_path, edge_shape)
    merge_traced_points(
        input_tensor, op, input_trace_points, flip_sign=flip_sign_inputs
    )


register_op("MaxPool", max_layer_trace)


def avg_layer_trace(
    op: Op,
    select_fn: Callable[[np.ndarray], np.ndarray],
    batch_index: int,
):
    kernel_size = op.attrs["kernel_size"]
    stride = np.array(op.attrs["strides"])
    input_tensor: Tensor = op.inputs[0]
    input = input_tensor.value[batch_index]
    output_tensor: Tensor = op.outputs[0]
    output = output_tensor.value[batch_index]
    output_points = get_point(output_tensor)
    flip_sign = get_flip_sign(output_tensor)
    if op.attrs["data_format"] == "NHWC":
        input = np.rollaxis(input, 2)
        output = np.rollaxis(output, 2)

    in_channels = input.shape[0]
    receptive_field_shape = (in_channels,) + kernel_size
    receptive_field_size = nnp.prod(receptive_field_shape)
    padding = calc_padding(
        np.array(input.shape)[1:], np.array(output.shape)[1:], stride, kernel_size
    )
    padded_input = np.pad(input, padding, mode="constant")

    output_point_index_tuple = np.unravel_index(output_points, output.shape)
    output_point_index = np.array(output_point_index_tuple)  # [out_ch + h + w, out]
    output_point_size = output_point_index.shape[1]
    output_value = output[output_point_index_tuple]  # (out)
    index = output_point_index[1:]  # [h + w, out]
    start_index = index * stride[:, np.newaxis]  # [h + w, out]
    receptive_field = padded_input[
        np.arange(in_channels)[np.newaxis, :, np.newaxis, np.newaxis],
        start_index[0][:, np.newaxis, np.newaxis, np.newaxis]
        + np.arange(kernel_size[0])[np.newaxis, np.newaxis, :, np.newaxis],
        start_index[1][:, np.newaxis, np.newaxis, np.newaxis]
        + np.arange(kernel_size[1])[np.newaxis, np.newaxis, np.newaxis, :],
    ]  # (out, in_ch, kh, kw)
    weighted_input = receptive_field  # (out, in_ch, kh, kw)
    unaligned_input_points, flip_sign_inputs = calc_flip_sign_batch(
        flip_sign=flip_sign,
        output_value=output_value,
        weighted_input=weighted_input.reshape(
            (output_point_size, receptive_field_size)
        ),
        select_fn=select_fn,
    )  # [out + in_ch*kh*kw, rf]
    unaligned_input_index = np.unravel_index(
        unaligned_input_points[1], receptive_field_shape
    )  # [in_ch + kh + kw, rf]
    input_index = (
        unaligned_input_index[0],
        unaligned_input_index[1]
        + start_index[0][unaligned_input_points[0]]
        - padding[1][0],
        unaligned_input_index[2]
        + start_index[1][unaligned_input_points[0]]
        - padding[2][0],
    )  # [in_ch + h + w, in]
    output_trace_points = output_points[unaligned_input_points[0]]
    input_trace_points = np.ravel_multi_index(input_index, input.shape)
    edge_shape = receptive_field_shape + output.shape
    op.attrs[TraceKey.EDGE_SHAPE] = edge_shape
    edge_path = np.ravel_multi_index(
        (unaligned_input_points[1], output_trace_points),
        (receptive_field_size, output.size),
    )
    op.attrs[TraceKey.EDGE_MASK] = to_mask(edge_path, edge_shape)
    merge_traced_points(
        input_tensor,
        op,
        input_trace_points,
        flip_sign=flip_sign_inputs,
    )


register_op("AvgPool", avg_layer_trace)


def mean_layer_trace(
    op: Op,
    select_fn: Callable[[np.ndarray], np.ndarray],
    batch_index: int,
):
    reduction_indices = tuple(np.array(op.attrs["reduction_indices"].value) - 1)
    input_tensor: Tensor = op.inputs[0]
    input = input_tensor.value[batch_index]
    output_tensor: Tensor = op.outputs[0]
    output = output_tensor.value[batch_index]
    output_points = get_point(output_tensor)
    flip_sign = get_flip_sign(output_tensor)
    output_point_index_tuple = np.unravel_index(output_points, output.shape)
    output_point_size = output_points.size
    output_value = output[output_point_index_tuple]  # (out)
    receptive_field_index_list = list(output_point_index_tuple)
    for reduction_index in reduction_indices:
        if op.attrs["keep_dims"]:
            receptive_field_index_list[reduction_index] = slice(None)
        else:
            receptive_field_index_list.insert(reduction_index, slice(None))
    receptive_field_index = tuple(receptive_field_index_list)
    receptive_field = input[receptive_field_index]
    reserved_indices = tuple(
        [index for index in range(len(input.shape)) if index not in reduction_indices]
    )
    weighted_input = np.transpose(receptive_field, reserved_indices + reduction_indices)
    unaligned_input_points, flip_sign_inputs = calc_flip_sign_batch(
        flip_sign=flip_sign,
        output_value=output_value,
        weighted_input=weighted_input.reshape(
            (output_point_size, weighted_input.size // output_point_size)
        ),
        select_fn=select_fn,
    )
    unaligned_input_index = np.unravel_index(
        unaligned_input_points[1], weighted_input.shape
    )
    input_index_list = [
        index[unaligned_input_points[0]] for index in output_point_index_tuple
    ]
    for reduction_index, index in zip(reduction_indices, unaligned_input_index[1:]):
        if op.attrs["keep_dims"]:
            input_index_list[reduction_index] = index
        else:
            input_index_list.insert(reduction_index, index)
    input_index = tuple(input_index_list)
    input_points = np.ravel_multi_index(input_index, input.shape)
    input_trace_points = input_points
    edge_shape = input.shape
    edge_path = input_trace_points
    op.attrs[TraceKey.EDGE_SHAPE] = edge_shape
    op.attrs[TraceKey.EDGE_MASK] = to_mask(edge_path, edge_shape)
    merge_traced_points(
        input_tensor, op, input_trace_points, flip_sign=flip_sign_inputs
    )


register_op("Mean", mean_layer_trace)


@jax.jit
# @partial(jax.jit, backend="gpu")
def get_weighted_input(padded_input, weight, receptive_field_index, weight_index):
    receptive_field = padded_input[receptive_field_index]  # (out, in_ch, kh, kw)
    weighted_input = (
        receptive_field * jnp.array(weight)[weight_index]
    )  # (out, in_ch, kh, kw)
    return weighted_input


def conv2d_layer_trace(
    op: Op,
    select_fn: Callable[[np.ndarray], np.ndarray],
    batch_index: int,
):
    weight: np.ndarray = op.attrs["kernel"].value
    kernel_size = op.attrs["kernel_size"]
    stride = np.array(op.attrs["strides"])
    input_tensor: Tensor = op.inputs[0]
    input = input_tensor.value[batch_index]
    output_tensor: Tensor = op.outputs[0]
    output = output_tensor.value[batch_index]
    output_points = get_point(output_tensor)
    flip_sign = get_flip_sign(output_tensor)
    if op.attrs["data_format"] == "NHWC":
        input = np.rollaxis(input, 2)
        output = np.rollaxis(output, 2)

    in_channels = input.shape[0]
    out_channels = output.shape[0]
    receptive_field_shape = weight.shape[1:]
    receptive_field_size = nnp.prod(receptive_field_shape)
    padding = calc_padding(
        np.array(input.shape)[1:], np.array(output.shape)[1:], stride, kernel_size
    )
    padded_input = np.pad(input, padding, mode="constant")

    output_point_index_tuple = np.unravel_index(output_points, output.shape)
    output_point_index = np.array(output_point_index_tuple)  # [out_ch + h + w, out]
    output_point_size = output_point_index.shape[1]
    output_value = output[output_point_index_tuple]  # (out)
    index = output_point_index[1:]  # [h + w, out]
    start_index = index * stride[:, np.newaxis]  # [h + w, out]
    receptive_field_index = (
        np.arange(in_channels)[np.newaxis, :, np.newaxis, np.newaxis],
        start_index[0][:, np.newaxis, np.newaxis, np.newaxis]
        + np.arange(kernel_size[0])[np.newaxis, np.newaxis, :, np.newaxis],
        start_index[1][:, np.newaxis, np.newaxis, np.newaxis]
        + np.arange(kernel_size[1])[np.newaxis, np.newaxis, np.newaxis, :],
    )  # [out * in_ch * kh * kw, rf]
    weight_index = (
        output_point_index[0][:, np.newaxis, np.newaxis, np.newaxis],
        np.arange(weight.shape[1])[np.newaxis, :, np.newaxis, np.newaxis],
        np.arange(weight.shape[2])[np.newaxis, np.newaxis, :, np.newaxis],
        np.arange(weight.shape[3])[np.newaxis, np.newaxis, np.newaxis, :],
    )  # [out * in_ch * kh * kw, rf]
    weighted_input = to_np_array(
        get_weighted_input(padded_input, weight, receptive_field_index, weight_index)
    )  # (out, in_ch, kh, kw)
    unaligned_input_points, flip_sign_inputs = calc_flip_sign_batch(
        flip_sign=flip_sign,
        output_value=output_value,
        weighted_input=weighted_input.reshape(
            (output_point_size, receptive_field_size)
        ),
        select_fn=select_fn,
    )  # [out + in_ch*kh*kw, rf]
    unaligned_input_index = np.unravel_index(
        unaligned_input_points[1], receptive_field_shape
    )  # [in_ch + kh + kw, rf]
    input_index = (
        unaligned_input_index[0],
        unaligned_input_index[1]
        + start_index[0][unaligned_input_points[0]]
        - padding[1][0],
        unaligned_input_index[2]
        + start_index[1][unaligned_input_points[0]]
        - padding[2][0],
    )  # [in_ch + h + w, in]
    output_trace_points = output_points[unaligned_input_points[0]]
    input_trace_points = np.ravel_multi_index(input_index, input.shape)
    weight_indices = np.ravel_multi_index(
        (output_point_index[0][unaligned_input_points[0]], unaligned_input_points[1]),
        (out_channels, receptive_field_size),
    )
    edge_shape = receptive_field_shape + output.shape
    weight_shape = weight.shape
    op.attrs[TraceKey.EDGE_SHAPE] = edge_shape
    op.attrs[TraceKey.WEIGHT_SHAPE] = weight_shape
    edge_path = np.ravel_multi_index(
        (unaligned_input_points[1], output_trace_points),
        (receptive_field_size, output.size),
    )
    op.attrs[TraceKey.EDGE_MASK] = to_mask(edge_path, edge_shape)
    op.attrs[TraceKey.WEIGHT_MASK] = to_mask(weight_indices, weight_shape)
    merge_traced_points(
        input_tensor,
        op,
        input_trace_points,
        flip_sign=flip_sign_inputs,
    )


register_op("Conv2D", conv2d_layer_trace)


def dw_conv2d_layer_trace(
    op: Op,
    select_fn: Callable[[np.ndarray], np.ndarray],
    batch_index: int,
):
    weight: np.ndarray = op.attrs["kernel"].value
    kernel_size = op.attrs["kernel_size"]
    stride = np.array(op.attrs["strides"])
    input_tensor: Tensor = op.inputs[0]
    input = input_tensor.value[batch_index]
    output_tensor: Tensor = op.outputs[0]
    output = output_tensor.value[batch_index]
    output_points = get_point(output_tensor)
    flip_sign = get_flip_sign(output_tensor)
    if op.attrs["data_format"] == "NHWC":
        input = np.rollaxis(input, 2)
        output = np.rollaxis(output, 2)

    in_channels = input.shape[0]
    out_channels = output.shape[0]
    channel_multiplier = out_channels // in_channels
    receptive_field_size = kernel_size[0] * kernel_size[1]
    padding = calc_padding(
        np.array(input.shape)[1:], np.array(output.shape)[1:], stride, kernel_size
    )
    padded_input = np.pad(input, padding, mode="constant")

    output_point_index_tuple = np.unravel_index(output_points, output.shape)
    output_point_index = np.array(output_point_index_tuple)  # [out_ch + h + w, out]
    output_point_size = output_point_index.shape[1]
    output_value = output[output_point_index_tuple]  # (out)
    index = output_point_index[1:]  # [h + w, out]
    start_index = index * stride[:, np.newaxis]  # [h + w, out]
    in_channel_index = output_point_index_tuple[0] // channel_multiplier
    multiplier_index = output_point_index_tuple[0] % channel_multiplier
    receptive_field_index = (
        in_channel_index[:, np.newaxis, np.newaxis],
        start_index[0][:, np.newaxis, np.newaxis]
        + np.arange(kernel_size[0])[np.newaxis, :, np.newaxis],
        start_index[1][:, np.newaxis, np.newaxis]
        + np.arange(kernel_size[1])[np.newaxis, np.newaxis, :],
    )  # [out + kh + kw, rf]
    weight_index = (
        multiplier_index[:, np.newaxis, np.newaxis],
        in_channel_index[:, np.newaxis, np.newaxis],
        np.arange(weight.shape[2])[np.newaxis, :, np.newaxis],
        np.arange(weight.shape[3])[np.newaxis, np.newaxis, :],
    )  # [out + kh + kw, rf]
    weighted_input = to_np_array(
        get_weighted_input(padded_input, weight, receptive_field_index, weight_index)
    )  # (out, kh, kw)
    unaligned_input_points, flip_sign_inputs = calc_flip_sign_batch(
        flip_sign=flip_sign,
        output_value=output_value,
        weighted_input=weighted_input.reshape(
            (output_point_size, receptive_field_size)
        ),
        select_fn=select_fn,
    )  # [out + kh*kw, rf]
    unaligned_input_index = np.unravel_index(
        unaligned_input_points[1], kernel_size
    )  # [kh + kw, rf]
    input_index = (
        in_channel_index[unaligned_input_points[0]],
        unaligned_input_index[0]
        + start_index[0][unaligned_input_points[0]]
        - padding[1][0],
        unaligned_input_index[1]
        + start_index[1][unaligned_input_points[0]]
        - padding[2][0],
    )  # [in_ch + h + w, in]
    output_trace_points = output_points[unaligned_input_points[0]]
    input_trace_points = np.ravel_multi_index(input_index, input.shape)
    weight_indices = np.ravel_multi_index(
        (
            multiplier_index[unaligned_input_points[0]],
            in_channel_index[unaligned_input_points[0]],
            unaligned_input_points[1],
        ),
        (channel_multiplier, in_channels, receptive_field_size),
    )
    edge_shape = kernel_size + output.shape
    weight_shape = weight.shape
    op.attrs[TraceKey.EDGE_SHAPE] = edge_shape
    op.attrs[TraceKey.WEIGHT_SHAPE] = weight_shape
    edge_path = np.ravel_multi_index(
        (unaligned_input_points[1], output_trace_points),
        (receptive_field_size, output.size),
    )
    op.attrs[TraceKey.EDGE_MASK] = to_mask(edge_path, edge_shape)
    op.attrs[TraceKey.WEIGHT_MASK] = to_mask(weight_indices, weight_shape)
    merge_traced_points(
        input_tensor,
        op,
        input_trace_points,
        flip_sign=flip_sign_inputs,
    )


register_op("DepthwiseConv2dNative", dw_conv2d_layer_trace)


def add_layer_trace(
    op: Op,
    select_fn: Callable[[np.ndarray], np.ndarray],
    batch_index: int,
):
    left_input_tensor: Tensor = op.inputs[0]
    left_input: np.ndarray = left_input_tensor.value[batch_index]
    right_input_tensor: Tensor = op.inputs[1]
    right_input: np.ndarray = right_input_tensor.value[batch_index]
    both_input = np.transpose(np.array([left_input.flatten(), right_input.flatten()]))
    output_tensor: Tensor = op.outputs[0]
    output: np.ndarray = output_tensor.value[batch_index]
    output_points = get_point(output_tensor)
    flip_sign = get_flip_sign(output_tensor)
    output_size = output.size
    output_point_index_tuple = np.unravel_index(output_points, output.shape)
    output_value = output[output_point_index_tuple]
    input_points, flip_sign_inputs = calc_flip_sign_batch(
        flip_sign=flip_sign,
        output_value=output_value,
        weighted_input=both_input[output_points],
        select_fn=select_fn,
    )
    # if np.all(input_points[1] == 0):
    #     logger.warning("op {} get all effective paths from left side", op.id)
    # if np.all(input_points[1] == 1):
    #     logger.warning("op {} get all effective paths from right side", op.id)
    left_input_trace_points = input_points[0][input_points[1] == 0]
    right_input_trace_points = input_points[0][input_points[1] == 1]
    edge_shape = (2, output_size)
    edge_path = np.concatenate(
        [left_input_trace_points, right_input_trace_points + output_size]
    )
    op.attrs[TraceKey.EDGE_SHAPE] = edge_shape
    op.attrs[TraceKey.EDGE_MASK] = to_mask(edge_path, edge_shape)
    if flip_sign is not None:
        left_flip_sign_inputs = flip_sign_inputs[input_points[1] == 0]
        right_flip_sign_inputs = flip_sign_inputs[input_points[1] == 1]
    else:
        left_flip_sign_inputs = None
        right_flip_sign_inputs = None
    merge_traced_points(
        left_input_tensor,
        op,
        left_input_trace_points,
        flip_sign=left_flip_sign_inputs,
    )
    merge_traced_points(
        right_input_tensor,
        op,
        right_input_trace_points,
        flip_sign=right_flip_sign_inputs,
    )


register_op("Add", add_layer_trace)
register_op("AddV2", add_layer_trace)


def transpose_layer_trace(
    op: Op,
    select_fn: Callable[[np.ndarray], np.ndarray],
    batch_index: int,
):
    perm = np.array(op.attrs["perm"].value[1:]) - 1
    input_tensor: Tensor = op.inputs[0]
    input_shape = input_tensor.value.shape[1:]
    output_tensor: Tensor = op.outputs[0]
    output_points = get_point(output_tensor)
    flip_sign = get_flip_sign(output_tensor)
    output_shape = output_tensor.value.shape[1:]
    output_point_index = np.unravel_index(output_points, output_shape)
    input_to_output_perm = {
        input_axis: output_axis for output_axis, input_axis in enumerate(perm)
    }
    inverse_perm = [
        input_to_output_perm[input_axis] for input_axis in range(len(input_shape))
    ]
    input_point_index = tuple([output_point_index[axis] for axis in inverse_perm])
    merge_traced_points(
        input_tensor,
        op,
        np.ravel_multi_index(input_point_index, input_shape),
        flip_sign=flip_sign,
    )
    op.attrs[TraceKey.TRIVIAL] = True


register_op("Transpose", transpose_layer_trace)


def pad_layer_trace(
    op: Op,
    select_fn: Callable[[np.ndarray], np.ndarray],
    batch_index: int,
):
    paddings = op.attrs["paddings"].value[1:]
    input_tensor: Tensor = op.inputs[0]
    input_shape = input_tensor.value.shape[1:]
    output_tensor: Tensor = op.outputs[0]
    output_points = get_point(output_tensor)
    flip_sign = get_flip_sign(output_tensor)
    output = output_tensor.value[batch_index]
    output_point_index = np.unravel_index(output_points, output.shape)
    input_point_index = tuple(
        [
            output_point_index[axis] - paddings[axis][0]
            for axis in range(len(output_point_index))
        ]
    )
    input_filter = reduce(
        np.logical_and,
        [
            np.logical_and(
                input_point_index[axis] >= 0,
                input_point_index[axis] < input_shape[axis],
            )
            for axis in range(len(input_point_index))
        ],
    )
    filtered_input_point_index = tuple(
        [
            input_point_index[axis][input_filter]
            for axis in range(len(input_point_index))
        ]
    )
    filtered_flip_sign = flip_sign[input_filter] if flip_sign is not None else None
    op.attrs[TraceKey.TRIVIAL] = True
    merge_traced_points(
        input_tensor,
        op,
        np.ravel_multi_index(filtered_input_point_index, input_shape),
        flip_sign=filtered_flip_sign,
    )


register_op("Pad", pad_layer_trace)


def concat_layer_trace(
    op: Op,
    select_fn: Callable[[np.ndarray], np.ndarray],
    batch_index: int,
):
    axis = op.attrs["axis"].value - 1
    input_tensors: List[Tensor] = op.inputs[:-1]
    input_shapes = list(map(lambda tensor: tensor.value.shape[1:], input_tensors))
    output_tensor: Tensor = op.outputs[0]
    output_points = get_point(output_tensor)
    flip_sign = get_flip_sign(output_tensor)
    output = output_tensor.value[batch_index]
    output_point_index = np.unravel_index(output_points, output.shape)
    start_index = 0
    for input_tensor, input_shape in zip(input_tensors, input_shapes):
        end_index = start_index + input_shape[axis]
        input_filter = np.logical_and(
            output_point_index[axis] >= start_index,
            output_point_index[axis] < end_index,
        )
        input_point_index = list(map(lambda x: x[input_filter], output_point_index))
        input_point_index[axis] = input_point_index[axis] - start_index
        merge_traced_points(
            input_tensor,
            op,
            np.ravel_multi_index(input_point_index, input_shape),
            flip_sign=flip_sign[input_filter],
        )
        start_index = end_index
    op.attrs[TraceKey.TRIVIAL] = True


register_op("Concat", concat_layer_trace)
register_op("ConcatV2", concat_layer_trace)


def batch_norm_layer_trace(
    op: Op,
    select_fn: Callable[[np.ndarray], np.ndarray],
    batch_index: int,
):
    input_tensor: Tensor = op.inputs[0]
    input = input_tensor.value[batch_index]
    output_tensor: Tensor = op.outputs[0]
    output = output_tensor.value[batch_index]
    output_points = get_point(output_tensor)
    index = np.unravel_index(output_points, output.shape)
    # flip_sign_filter = np.logical_xor(
    #     input[index] >= 0, output[index] >= 0
    # )
    # flip_sign = index_update(
    #     np.ones_like(flip_sign_filter, np.int32),
    #     flip_sign_filter, -1, inplace=True,
    # )
    # merge_traced_points(input_tensor, op, output_points, flip_sign=flip_sign)
    flip_sign_filter = np.logical_xor(input[index] >= 0, output[index] >= 0)
    filtered_output_points = output_points[~flip_sign_filter]
    merge_traced_points(input_tensor, op, filtered_output_points, flip_sign=None)
    op.attrs[TraceKey.TRIVIAL] = True


register_op("FusedBatchNorm", batch_norm_layer_trace)
register_op("FusedBatchNormV2", batch_norm_layer_trace)
register_op("FusedBatchNormV3", batch_norm_layer_trace)


def trivial_layer_trace(
    op: Op,
    select_fn: Callable[[np.ndarray], np.ndarray],
    batch_index: int,
):
    input_tensor: Tensor = op.inputs[0]
    output_tensor: Tensor = op.outputs[0]
    output_points = get_point(output_tensor)
    flip_sign = get_flip_sign(output_tensor)
    merge_traced_points(
        input_tensor,
        op,
        output_points,
        flip_sign=flip_sign,
    )
    op.attrs[TraceKey.TRIVIAL] = True


register_op("Relu", trivial_layer_trace)
register_op("Relu6", trivial_layer_trace)
register_op("Identity", trivial_layer_trace)
register_op("Reshape", trivial_layer_trace)
register_op("Squeeze", trivial_layer_trace)
register_op("BiasAdd", trivial_layer_trace)
