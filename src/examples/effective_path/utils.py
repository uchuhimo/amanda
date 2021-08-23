from functools import partial
from typing import Iterable, Tuple, TypeVar, Union

import jax

# import jax.numpy as np
import jax.numpy as jnp
import numpy as np
import numpy as nnp

__all__ = ["argtopk", "arg_approx", "arg_approx_signed", "repeat", "concatenate"]

jax.config.update("jax_platform_name", "cpu")


def get_int_k(array: np.ndarray, k: Union[int, float]) -> int:
    if type(k) is float:
        if 0.0 < k < 1.0:
            int_k = round(array.size * k)
            if int_k == array.size:
                return array.size - 1
            elif int_k == 0:
                return 1
            return int_k
        else:
            raise ValueError()
    else:
        return int(k)


def argtopk(array: np.ndarray, k: Union[int, float]) -> np.ndarray:
    k = get_int_k(array, k)
    if k == 1:
        return np.array([np.argmax(array)])
    else:
        if isinstance(array, jax.numpy.DeviceArray):
            array = nnp.array(array)
        return nnp.argpartition(array, -k, axis=None)[-k:]


def arg_sorted_topk(array: np.ndarray, k: Union[int, float]) -> np.ndarray:
    k = get_int_k(array, k)
    return np.argsort(array)[::-1][:k]


def arg_approx(array: np.ndarray, precision: float) -> np.ndarray:
    if (1 / array.size) >= precision:
        return np.array([np.argmax(array)])
    input_sum = array.sum()
    if input_sum <= 0:
        return np.array([np.argmax(array)])
    input = array.flatten()
    threshold = input_sum * precision
    sorted_input = np.sort(input[::-1])
    topk = sorted_input.cumsum().searchsorted(threshold)
    if topk == len(input):
        return np.where(input > 0)[0]
    else:
        return argtopk(input, topk + 1)


@partial(jax.jit, backend="gpu")
def argmax_batch(array: np.ndarray):
    return (jnp.arange(array.shape[0]), jnp.argmax(array, axis=-1))


vsearchsorted = jax.vmap(jnp.searchsorted, (0, 0), 0)


@partial(jax.jit, static_argnums=1, backend="gpu")
# @jax.jit
def arg_approx_batch_mask(
    input,
    precision: float,
) -> Tuple[np.ndarray, np.ndarray]:
    input_sum = input.sum(axis=-1)
    threshold = input_sum * precision
    sorted_input = jnp.sort(input[:, ::-1], axis=-1)
    input_sum = sorted_input.cumsum(axis=-1)
    topk = jnp.minimum(vsearchsorted(input_sum, threshold), input.shape[-1] - 1)
    input_threshold = sorted_input[jnp.arange(input.shape[0]), topk]
    return input > input_threshold[:, np.newaxis]


def to_np_array(array):
    if np == jnp:
        return array
    if isinstance(array, jax.numpy.DeviceArray):
        return np.array(array)
    elif isinstance(array, tuple):
        return tuple([to_np_array(element) for element in array])
    else:
        return array


def arg_approx_batch(
    array: np.ndarray,
    precision: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if (1 / array.shape[-1]) >= precision:
        return to_np_array(argmax_batch(array))
    else:
        return np.nonzero(to_np_array(arg_approx_batch_mask(array, precision)))


def index_update(x, idx, y, inplace=False):
    if isinstance(x, jax.numpy.DeviceArray):
        return jax.ops.index_update(x, idx, y)
    else:
        if not inplace:
            x = np.copy(x)
        x[idx] = y
        return x


def arg_approx_signed(array: np.ndarray, precision: float) -> np.ndarray:
    result = []
    for input in [array.copy(), -array]:
        input[input < 0] = 0
        result.append(arg_approx(input, precision))
    return np.concatenate(result)


def repeat(a: int, repeats: int) -> np.ndarray:
    if repeats > 1:
        return np.repeat(a, repeats)
    elif repeats == 1:
        return np.array([a])
    else:
        return np.array([])


def concatenate(a_tuple, axis=0, dtype=np.int64) -> np.ndarray:
    if len(a_tuple) == 0:
        return np.array([], dtype=dtype)
    else:
        return np.concatenate(a_tuple, axis)


T = TypeVar("T")


def filter_not_null(iterable: Iterable[T]) -> Iterable[T]:
    return (element for element in iterable if element is not None)
