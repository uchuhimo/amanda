import operator
from functools import reduce


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def addmm(node):
    # [n, p] = addmm([n, p], [n, m], [m, p], *, *)
    n, m = node.inputs[1].shape
    m, p = node.inputs[2].shape
    return n * m * p


def addmv(node):
    # [n] = addmv([n], [n, m], [m], *, *)
    n, m = node.inputs[1].shape
    return n * m


def bmm(node):
    # [b, n, p] = bmm([b, n, m], [b, m, p])
    b, n, m = node.inputs[0].shape
    b, m, p = node.inputs[1].shape
    return b * n * m * p


def matmul(node):
    if node.inputs[0].ndim == 1 and node.inputs[1].ndim == 1:
        # [] = matmul([n], [n])
        n = node.inputs[0].shape[0]
        return n
    elif node.inputs[0].ndim == 1 and node.inputs[1].ndim == 2:
        # [m] = matmul([n], [n, m])
        n, m = node.inputs[1].shape
        return n * m
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 1:
        # [n] = matmul([n, m], [m])
        n, m = node.inputs[0].shape
        return n * m
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 2:
        # [n, p] = matmul([n, m], [m, p])
        n, m = node.inputs[0].shape
        m, p = node.inputs[1].shape
        return n * m * p
    elif node.inputs[0].ndim == 1:
        # [..., m] = matmul([n], [..., n, m])
        *b, n, m = node.inputs[1].shape
        return prod(b) * n * m
    elif node.inputs[1].ndim == 1:
        # [..., n] = matmul([..., n, m], [m])
        *b, n, m = node.inputs[0].shape
        return prod(b) * n * m
    else:
        # [..., n, p] = matmul([..., n, m], [..., m, p])
        *b, n, p = node.outputs[0].shape
        *_, n, m = node.inputs[0].shape
        *_, m, p = node.inputs[1].shape
        return prod(b) * n * m * p


def mul(node):
    os = node.outputs[0].shape
    return prod(os)


def convolution(node):
    if node.outputs[0].shape[1] == node.inputs[1].shape[0]:
        oc, ic, *ks = node.inputs[1].shape
    else:
        ic, oc, *ks = node.inputs[1].shape
    os = node.outputs[0].shape
    return prod(os) * ic * prod(ks)


def norm(node):
    if node.name in ["batch_norm", "instance_norm"]:
        affine = node.inputs[1].shape is not None
    elif node.name in ["layer_norm", "group_norm"]:
        affine = node.inputs[2].shape is not None
    else:
        raise ValueError(node.name)

    os = node.outputs[0].shape
    return prod(os) if affine else 0


def avg_pool_or_mean(node):
    os = node.outputs[0].shape
    return prod(os)


def leaky_relu(node):
    os = node.outputs[0].shape
    return prod(os)


def upsample_bilinear2d(node):
    os = node.outputs[0].shape
    return prod(os) * 4
