from typing import Iterable, Tuple, TypeVar, Union

import numpy as np

__all__ = ["argtopk", "arg_approx", "arg_approx_signed", "repeat", "concatenate"]


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
        return np.argpartition(array, -k, axis=None)[-k:]


def arg_sorted_topk(array: np.ndarray, k: Union[int, float]) -> np.ndarray:
    # topk_index = argtopk(array, k)
    # sorted_index = np.array(list(reversed(np.argsort(array[topk_index]))))
    # return topk_index[sorted_index]
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
    sorted_input = input.copy()
    sorted_input[::-1].sort()
    # topk = np.argmax(sorted_input.cumsum() >= threshold)
    topk = sorted_input.cumsum().searchsorted(threshold)
    if topk == len(input):
        return np.where(input > 0)[0]
    else:
        return argtopk(input, topk + 1)


def argmax_batch(array: np.ndarray):
    return (np.arange(array.shape[0]), np.argmax(array, axis=-1))


def arg_approx_batch(
    array: np.ndarray,
    precision: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if (1 / array.shape[-1]) >= precision:
        return argmax_batch(array)
    input_sum = array.sum(axis=-1)
    input = array
    threshold = input_sum * precision
    sorted_input = np.copy(input)
    sorted_input[:, ::-1].sort(axis=-1)
    topk = np.zeros_like(threshold, dtype=np.int32)
    input_sum = sorted_input.cumsum(axis=-1)
    for i in range(topk.size):
        topk[i] = np.searchsorted(input_sum[i], threshold[i])
    topk = np.minimum(topk, input.shape[-1] - 1)
    input_threshold = sorted_input[np.arange(input.shape[0]), topk]
    return np.nonzero(input > input_threshold[:, np.newaxis])


# def arg_approx(array: np.ndarray, precision: float) -> np.ndarray:
#     input_sum = array.sum()
#     if input_sum == 0:
#         return np.array([], dtype=np.int64)
#     input = array.flatten()
#     threshold = input_sum * precision
#     sorted_input = input.copy()
#     sorted_input[::-1].sort()
#     # topk = np.argmax(sorted_input.cumsum() >= threshold)
#     topk = sorted_input.cumsum().searchsorted(threshold)
#     return argtopk(input, topk + 1)


def arg_approx_signed(array: np.ndarray, precision: float) -> np.ndarray:
    result = []
    for input in [array.copy(), -array]:
        input[input < 0] = 0
        result.append(arg_approx(input, precision))
    return np.concatenate(result)


def repeat(a: int, repeats: int) -> np.ndarray:
    # if repeats > 1:
    #     return np.repeat(a, repeats)
    # elif repeats == 1:
    #     return np.array([a])
    # else:
    #     return np.array([])
    return np.repeat(a, repeats)


def concatenate(a_tuple, axis=0, out=None, dtype=np.int64) -> np.ndarray:
    if len(a_tuple) == 0:
        return np.array([], dtype=dtype)
    else:
        return np.concatenate(a_tuple, axis, out)


T = TypeVar("T")


def filter_not_null(iterable: Iterable[T]) -> Iterable[T]:
    return (element for element in iterable if element is not None)
