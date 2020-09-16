import os
import warnings
from importlib.util import find_spec

from .graph import Edge  # noqa: F401
from .graph import Graph  # noqa: F401
from .graph import InputPort  # noqa: F401
from .graph import Op  # noqa: F401
from .graph import OutputPort  # noqa: F401
from .graph import create_control_edge  # noqa: F401
from .graph import create_control_input_port  # noqa: F401
from .graph import create_control_output_port  # noqa: F401
from .graph import create_edge  # noqa: F401
from .graph import create_op  # noqa: F401
from .namespace import (  # noqa: F401
    Namespace,
    Registry,
    exp,
    get_global_registry,
    get_mapper,
    get_mapping_table,
)
from .type import DataType, unknown_type  # noqa: F401

if find_spec("tensorflow"):
    os.environ["KMP_WARNINGS"] = "FALSE"
    warnings.filterwarnings("ignore", category=FutureWarning)
    from .conversion import tensorflow  # noqa: F401
if find_spec("torch"):
    from .conversion import pytorch  # noqa: F401
if find_spec("onnx"):
    from .conversion import onnx  # noqa: F401
if find_spec("mmdnn"):
    from .conversion import mmdnn  # noqa: F401

__version__ = "0.1.0"
