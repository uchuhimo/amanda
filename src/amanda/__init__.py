from importlib.util import find_spec

from . import event, import_hook, intercepts, io, tools  # noqa: F401
from .adapter import Adapter, get_adapter_registry  # noqa: F401
from .attributes import Attributes  # noqa: F401
from .cache import cache_disabled, cache_enabled, is_cache_enabled  # noqa: F401
from .event import Event, EventContext, OpContext  # noqa: F401
from .graph import (  # noqa: F401
    Edge,
    Graph,
    InputPort,
    Op,
    OutputPort,
    create_control_edge,
    create_control_input_port,
    create_control_output_port,
    create_edge,
    create_op,
)
from .import_hook import disabled, enabled, is_enabled  # noqa: F401
from .mapping import Mapping  # noqa: F401
from .namespace import (  # noqa: F401
    Namespace,
    Registry,
    exp,
    get_global_registry,
    get_mapper,
    get_mapping_table,
)
from .tool import Tool, apply  # noqa: F401
from .type import DataType, unknown_type  # noqa: F401

if find_spec("tensorflow"):
    from .conversion import tf as tensorflow  # noqa: F401
if find_spec("torch"):
    from .conversion import pytorch  # noqa: F401
if find_spec("onnx"):
    from .conversion import onnx  # noqa: F401
if find_spec("mmdnn"):
    from .conversion import mmdnn  # noqa: F401

from .conversion.checkpoint import Checkpoint  # noqa: F401

intercepts.init()

__version__ = "0.1.0"
