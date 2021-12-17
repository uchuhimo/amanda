"""
adapted from nvidia apex/asp repo
"""
import collections
import sys
from itertools import permutations

import numpy as np
import torch

""" compute density (helper fn to compute % NNZs in a tensor) """


def fill(x):
    return float(x.nonzero().size(0)) / torch.numel(x)


""" reshape matrix into m-dimensional vectors: (h,w) -> (hw/m, m) """


def reshape_1d(matrix, m):
    # If not a nice multiple of m, fill with zeroes.
    if matrix.shape[1] % m > 0:
        mat = torch.FloatTensor(
            matrix.shape[0], matrix.shape[1] + (m - matrix.shape[1] % m)
        ).fill_(0)
        mat[:, : matrix.shape[1]] = matrix
        shape = mat.shape
        return mat.view(-1, m), shape
    else:
        return matrix.view(-1, m), matrix.shape


""" return all possible m:n patterns in a 1d vector """
valid_m4n2_1d_patterns = None


def compute_valid_1d_patterns(m, n):
    # Early exit if patterns was already created.
    global valid_m4n2_1d_patterns

    if m == 4 and n == 2 and valid_m4n2_1d_patterns is not None:
        return valid_m4n2_1d_patterns
    patterns = torch.zeros(m)
    patterns[:n] = 1
    valid_patterns = torch.Tensor(list(set(permutations(patterns.tolist()))))
    if m == 4 and n == 2:
        valid_m4n2_1d_patterns = valid_patterns
    return valid_patterns


""" m:n 1d structured best """


def mn_1d_best(matrix, m, n):
    # Find all possible patterns.
    patterns = compute_valid_1d_patterns(m, n)

    # Find the best m:n pattern (sum of non-masked weights).
    mask = torch.IntTensor(matrix.shape).fill_(1).view(-1, m)
    mat, shape = reshape_1d(matrix, m)
    pmax = torch.argmax(torch.matmul(mat.abs(), patterns.t()), dim=1)
    mask[:] = patterns[pmax[:]]
    mask = mask.view(matrix.shape)
    return mask


def m4n2_1d(mat, density):
    return mn_1d_best(mat, 4, 2)


"""
  Below 2d-masking related code is targeted more for training (from scratch).
  2d-pruning of a weight tensor is done to accelerate DGRAD step during backprop
  phase of training algorithm. Acceleration comes from using SpMMA instructions in
  Tensor Cores of NVIDIA Ampere GPU Architecture
  (note: this code does not do the acceleration, GPU kernels are required for this).
  1d pruning of weight tensor helps speed up FPROP step by pruning in 2:4 pattern
  along the horizontal (logical) direction.
  During DGRAD step, weight tensor is transposed. 2d pruning functions below, mask
  weight tensor such that their transposed versions are also 2:4 sparse along the
  horizontal (logical) direction. Thus, with 2d pruning, weight tensors are
  2:4 sparse along row and column directions.
 """

""" m:n 2d structured pruning: greedy method to select mask """


def mn_2d_greedy(matrix, m, n):
    # Convert to numpy
    mat = matrix.cpu().detach().numpy()
    mask = np.ones(mat.shape, dtype=int)

    rowCount = int(mat.shape[0] / m) * m
    colCount = int(mat.shape[1] / m) * m
    for rowStartIdx in range(0, rowCount, m):
        rowEndIdx = rowStartIdx + m
        for colStartIdx in range(0, colCount, m):
            colEndIdx = colStartIdx + m
            matrixSub = np.absolute(
                np.squeeze(mat[rowStartIdx:rowEndIdx, colStartIdx:colEndIdx])
            )
            maskSub = np.squeeze(mask[rowStartIdx:rowEndIdx, colStartIdx:colEndIdx])
            maskSub.fill(0.0)
            matrixVecView = matrixSub.reshape(-1)
            linearIdx = np.argsort(matrixVecView)
            matrixIdx = [(int(x / m), x % m) for x in linearIdx]
            rowCounter = collections.Counter()
            colCounter = collections.Counter()
            for currIdx in range(len(linearIdx) - 1, -1, -1):
                currMatrixEntry = matrixIdx[currIdx]
                if (rowCounter[currMatrixEntry[0]] == n) or (
                    colCounter[currMatrixEntry[1]] == n
                ):
                    continue
                # end if
                maskSub[currMatrixEntry[0], currMatrixEntry[1]] = 1.0
                rowCounter[currMatrixEntry[0]] += 1
                colCounter[currMatrixEntry[1]] += 1

    return torch.tensor(mask)


def m4n2_2d_greedy(mat, density):
    return mn_2d_greedy(mat, 4, 2)


""" return all possible m:n patterns in a mxn block. """
valid_m4n2_2d_patterns = None


def compute_valid_2d_patterns(m, n):
    # Early exit if patterns was already created.
    global valid_m4n2_2d_patterns
    if valid_m4n2_2d_patterns is not None:
        return valid_m4n2_2d_patterns

    patterns = torch.zeros(m)
    patterns[:n] = 1
    patterns = list(set(permutations(patterns.tolist())))
    patterns = patterns + patterns
    patterns = torch.Tensor(list(set(permutations(patterns, m))))

    valid = ((patterns.sum(dim=1) <= n).sum(dim=1) == m).nonzero().view(-1)
    valid_patterns = torch.Tensor(valid.shape[0], m, m)
    valid_patterns[:] = patterns[valid[:]]

    if m == 4 and n == 2:
        valid_m4n2_2d_patterns = valid_patterns
    return valid_patterns


""" returns a sparse mask """


def create_mask(tensor, pattern="m4n2_1d", density=0.5):
    # Reshape tensor and mask.
    shape = tensor.shape
    ttype = tensor.type()
    mask = None
    t = tensor.float().contiguous()
    is_cuda = t.is_cuda
    if is_cuda:
        t = t.cpu()

    # 1d-tensor
    if len(shape) == 1:
        t = t.view(1, shape[0])
        func = getattr(sys.modules[__name__], pattern, None)
        mask = func(t, density)
    # 2d-tensor (in, out)
    elif len(shape) == 2:
        t = t.view(shape[0], shape[1])
        func = getattr(sys.modules[__name__], pattern, None)
        mask = func(t, density)
    # 3d-tensor (batch, in, out)
    elif len(shape) == 3:
        t = t.view(shape[0] * shape[1], shape[2])
        func = getattr(sys.modules[__name__], pattern, None)
        mask = func(t, density)
    # 4d-tensor (in, out, h, w)
    elif len(shape) == 4:
        """
        # transformers (bmm)
        t = t.view(shape[0]*shape[1]*shape[2], shape[3])
        func = getattr(sys.modules[__name__], pattern, None)
        mask = func(t, density)
        return mask.view(shape).type(ttype)
        """
        # convs
        t = (
            t.permute(2, 3, 0, 1)
            .contiguous()
            .view(shape[2] * shape[3] * shape[0], shape[1])
        )
        func = getattr(sys.modules[__name__], pattern, None)
        mask = func(t, density)
        mask = (
            mask.view(shape[2], shape[3], shape[0], shape[1])
            .permute(2, 3, 0, 1)
            .contiguous()
        )

    mask = mask.view(shape).type(ttype)
    if is_cuda:
        mask = mask.cuda()
    return mask
