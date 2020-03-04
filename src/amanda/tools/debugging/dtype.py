from dataclasses import dataclass
from typing import Any

import tensorflow as tf
from torch._C import Type as TorchType


@dataclass
class WrappedDType:
    value: Any

    def is_valid(self) -> bool:
        if isinstance(self.value, TorchType):
            return self.value.kind() == "TensorType"
        elif isinstance(self.value, tf.DType):
            return self.value._is_ref_dtype
        else:
            return False
