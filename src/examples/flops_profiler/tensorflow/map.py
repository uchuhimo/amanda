from examples.flops_profiler.utils.kernel import *

# pytorch_map = (
#     ('addmm', addmm),
#     ('addmv', addmv),
#     ('bmm', bmm),
#     (('linear', 'matmul'), matmul),
#     (('mul', 'mul_'), mul),
#     (('_convolution','conv2d'), convolution),
#     (('batch_norm', 'instance_norm', 'layer_norm',
#       'group_norm'), norm),
#     (('adaptive_avg_pool1d', 'adaptive_avg_pool2d',
#       'adaptive_avg_pool3d', 'avg_pool1d', 'avg_pool2d',
#       'avg_pool3d', 'mean'), avg_pool_or_mean),
#     ('leaky_relu', leaky_relu),
#     ('upsample_bilinear2d', upsample_bilinear2d),
#     (('adaptive_max_pool1d', 'adaptive_max_pool2d',
#       'adaptive_max_pool3d', 'add', 'add_',
#       'alpha_dropout', 'cat', 'chunk', 'clamp',
#       'clone', 'constant_pad_nd', 'contiguous',
#       'detach', 'div', 'div_', 'dropout',
#       'dropout_', 'embedding', 'eq', 'feature_dropout',
#       'flatten', 'floor', 'floor_divide', 'gt',
#       'hardtanh_', 'hardtanh', 'index', 'int',  'log_softmax',
#       'lt', 'max_pool1d', 'max_pool1d_with_indices',
#       'max_pool2d', 'max_pool2d_with_indices', 'max_pool3d',
#       'max_pool3d_with_indices', 'max_unpool1d',
#       'max_unpool2d', 'max_unpool3d', 'ne',
#       'reflection_pad1d', 'reflection_pad2d',
#       'reflection_pad3d', 'relu', 'relu_',
#       'replication_pad1d', 'replication_pad2d',
#       'replication_pad3d', 'rsub', 'select', 'sigmoid',
#       'size', 'slice', 'softmax', 'softshrink',
#       'squeeze', 'stack', 'sub', 'sum', 't',
#       'tanh', 'threshold', 'to', 'transpose',
#       'upsample_nearest2d', 'view', 'zeros',
#       'constant', 'listconstruct', 'listunpack',
#       'numtotensor', 'tupleconstruct'), None),
# )

tf_map = (
    ("addmm", addmm),
    ("addmv", addmv),
    ("bmm", bmm),
    (("MatMul",), matmul),
    (("Mul",), mul),
    (("_convolution", "conv2d"), convolution),
    (("batch_norm", "instance_norm", "layer_norm", "group_norm"), norm),
    (
        (
            "adaptive_avg_pool1d",
            "adaptive_avg_pool2d",
            "adaptive_avg_pool3d",
            "avg_pool1d",
            "avg_pool2d",
            "avg_pool3d",
            "mean",
        ),
        avg_pool_or_mean,
    ),
    ("leaky_relu", leaky_relu),
    ("upsample_bilinear2d", upsample_bilinear2d),
    (
        (
            "adaptive_max_pool1d",
            "adaptive_max_pool2d",
            "adaptive_max_pool3d",
            "add",
            "add_",
            "alpha_dropout",
            "cat",
            "chunk",
            "clamp",
            "clone",
            "constant_pad_nd",
            "contiguous",
            "detach",
            "div",
            "div_",
            "dropout",
            "dropout_",
            "embedding",
            "eq",
            "feature_dropout",
            "flatten",
            "floor",
            "floor_divide",
            "gt",
            "hardtanh_",
            "hardtanh",
            "index",
            "int",
            "log_softmax",
            "lt",
            "max_pool1d",
            "max_pool1d_with_indices",
            "max_pool2d",
            "max_pool2d_with_indices",
            "max_pool3d",
            "max_pool3d_with_indices",
            "max_unpool1d",
            "max_unpool2d",
            "max_unpool3d",
            "ne",
            "reflection_pad1d",
            "reflection_pad2d",
            "reflection_pad3d",
            "relu",
            "relu_",
            "replication_pad1d",
            "replication_pad2d",
            "replication_pad3d",
            "rsub",
            "select",
            "sigmoid",
            "size",
            "slice",
            "softmax",
            "softshrink",
            "squeeze",
            "stack",
            "sub",
            "sum",
            "t",
            "tanh",
            "threshold",
            "to",
            "transpose",
            "upsample_nearest2d",
            "view",
            "zeros",
            "constant",
            "listconstruct",
            "listunpack",
            "numtotensor",
            "tupleconstruct",
        ),
        None,
    ),
)
