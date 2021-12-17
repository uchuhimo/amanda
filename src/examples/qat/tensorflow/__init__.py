from .resnet_18_cifar100_train import cifar100_model_fn  # noqa: F401
from .resnet_50_imagenet_train import (  # noqa: F401
    resnet_50_model_fn,
    validate_batch_size_for_multi_gpu,
)
