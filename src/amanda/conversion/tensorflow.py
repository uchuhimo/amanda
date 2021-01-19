import gc
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Dict, List, Tuple, Union

import google.protobuf.text_format
import tensorflow as tf
from tensorflow.core.framework import tensor_pb2, variable_pb2
from tensorflow.core.framework.op_def_pb2 import OpDef
from tensorflow.core.framework.versions_pb2 import VersionDef
from tensorflow.core.protobuf.meta_graph_pb2 import AssetFileDef, SignatureDef
from tensorflow.core.protobuf.saved_object_graph_pb2 import SavedObjectGraph
from tensorflow.core.protobuf.saver_pb2 import SaverDef
from tensorflow.python import tensor_shape, types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.meta_graph import stripped_op_list_for_graph
from tensorflow.python.framework.op_def_library import (
    _IsListValue,
    _MakeBool,
    _MakeFloat,
    _MakeInt,
    _MakeShape,
    _MakeStr,
    _MakeTensor,
    _MakeType,
)
from tensorflow.python.util import compat

from amanda.adapter import Adapter, get_adapter_registry
from amanda.conversion.utils import to_proto, without_internal_attrs
from amanda.event import EventContext, on_graph_loaded
from amanda.exception import MismatchNamespaceError
from amanda.graph import Graph, Op, OutputPort, create_graph, create_op
from amanda.io.serde import (
    ProtoToDictSerde,
    Serde,
    SerdeContext,
    SerdeDispatcher,
    TypeSerde,
    get_serde_registry,
    serialize_type,
)
from amanda.lang import replace_all_refs
from amanda.namespace import Namespace, default_namespace
from amanda.type import DataType

_namespace = default_namespace() / Namespace("tensorflow")
_internal_namespace = _namespace / Namespace("internal")
_type_namespace = Namespace("tf")


def tf_namespace() -> Namespace:
    return _namespace


def tf_internal_namespace() -> Namespace:
    return _internal_namespace


def tf_type_namespace() -> Namespace:
    return _type_namespace


def tf_dtype(name: str):
    return DataType(tf_type_namespace(), name)


_name_to_dtype = {
    dtype.name: tf_dtype(dtype.name) for _, dtype in dtypes._INTERN_TABLE.items()
}

_name_to_tf_dtype = {
    tf_dtype.name: tf_dtype for _, tf_dtype in dtypes._INTERN_TABLE.items()
}


class TFSerde(TypeSerde):
    def serialize_type(self, type: Any) -> DataType:
        return _name_to_dtype[type.name]

    def deserialize_type(self, dtype: DataType) -> Any:
        return _name_to_tf_dtype[dtype.name]


_tf_serde = TFSerde()


class TFTensorShapeSerde(Serde):
    def serialize_type(self, type: Any) -> DataType:
        return tf_dtype("TensorShape")

    def deserialize_type(self, dtype: DataType) -> Any:
        return tf.TensorShape

    def serialize(self, value: Any) -> Any:
        return value.dims if value.dims is None else [dim.value for dim in value.dims]

    def deserialize(self, value: Any, context: SerdeContext) -> Any:
        return tf.TensorShape(value)


class TFDTypeSerde(Serde):
    def serialize_type(self, type: Any) -> DataType:
        return tf_dtype("DType")

    def deserialize_type(self, dtype: DataType) -> Any:
        return tf.DType

    def serialize(self, value: Any) -> Any:
        return value.name

    def deserialize(self, value: Any, context: SerdeContext) -> Any:
        return _name_to_tf_dtype[value]


@dataclass
class TFSerdeDispatcher(SerdeDispatcher):
    def __post_init__(self):
        for tf_dtype in _name_to_tf_dtype.values():
            self.register_type(tf_dtype, _tf_serde)
        for dtype in _name_to_dtype.values():
            self.register_dtype_name(dtype.name, _tf_serde)
        for proto_type in [
            SaverDef,
            tf.MetaGraphDef.MetaInfoDef,
            AssetFileDef,
            SavedObjectGraph,
            SignatureDef,
            VersionDef,
            tensor_pb2.TensorProto,
            tf.NameAttrList,
        ]:
            serde = ProtoToDictSerde(
                proto_type,
                name=proto_type.__name__,
                namespace=tf_type_namespace(),
            )
            self.register_type(proto_type, serde)
            self.register_dtype_name(proto_type.__name__, serde)
        self.register_type(tf.TensorShape, TFTensorShapeSerde())
        self.register_dtype_name("TensorShape", TFTensorShapeSerde())
        self.register_type(tf.DType, TFDTypeSerde())
        self.register_dtype_name("DType", TFDTypeSerde())


get_serde_registry().register_namespace(tf_type_namespace(), TFSerdeDispatcher())


def import_from_graph(
    tf_graph: tf.Graph,
    saver_def: SaverDef = None,
    session: tf.Session = None,
) -> Graph:
    graph_def = tf_graph.as_graph_def()
    graph = import_from_graph_def(graph_def)
    for variable in tf_graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        variable_def = variable.to_proto()
        op = graph.get_op(variable.name[: variable.name.rfind(":")])
        op.attrs["initial_value_name"] = variable_def.initial_value_name
        op.attrs["initializer_name"] = variable_def.initializer_name
        op.attrs["snapshot_name"] = variable_def.snapshot_name
        op.attrs["is_resource"] = variable_def.is_resource
        op.attrs["trainable"] = variable_def.trainable
    for name in tf_graph.collections:
        collection = tf_graph.get_collection(name)
        if name in tf.GraphKeys._VARIABLE_COLLECTIONS:
            graph.attrs[f"collection/{name}"] = [
                graph.get_op(variable.name[: variable.name.rfind(":")])
                for variable in collection
            ]
        elif isinstance(collection[0], tf.Operation):
            graph.attrs[f"collection/{name}"] = [
                graph.get_op(tf_op.name) for tf_op in collection
            ]
        elif isinstance(collection[0], tf.Tensor):
            graph.attrs[f"collection/{name}"] = [
                graph.get_op(tf_tensor.op.name).output_port(tf_tensor.value_index)
                for tf_tensor in collection
            ]
        else:
            graph.attrs[f"collection/{name}"] = collection
    if saver_def is not None:
        graph.attrs["saver_def"] = saver_def
    if session is not None:
        update_initialized_variables(graph, tf_graph, session)
    return graph


def init_op_attrs(op: Op, node: tf.NodeDef):
    op.namespace = tf_namespace()
    if node.HasField("experimental_debug_info"):
        op.attrs["experimental_debug_info"] = node.experimental_debug_info


def update_initialized_variables(graph: Graph, tf_graph: tf.Graph, session: tf.Session):
    with tf_graph.as_default():
        all_variables = tf.global_variables()
        uninitialized_variables = session.run(
            tf.report_uninitialized_variables(tf.global_variables())
        )
        if len(all_variables) == len(uninitialized_variables):
            return
        initialized_variables = set(
            map(lambda variable: variable.op.name, all_variables)
        ) - set(map(lambda x: x.decode(), uninitialized_variables))
        initialized_variable_to_value = session.run(
            {
                variable.name: variable.read_value()
                for variable in all_variables
                if variable.op.name in initialized_variables
            }
        )
        for name, value in initialized_variable_to_value.items():
            op_name = name.split(":")[0]
            graph.get_op(op_name).attrs["value"] = value


@dataclass(frozen=True)
class TFTensor:
    op: str
    output_index: int


@lru_cache(maxsize=100000)
def lower_name(name: str):
    return name.lower()


def import_from_graph_def(graph_def: Union[tf.GraphDef, str, bytes, Path]) -> Graph:
    graph_def = to_proto(graph_def, tf.GraphDef)
    graph = create_graph(
        namespace=tf_namespace(),
    )
    tf_graph = tf.Graph()
    name_to_node = {node.name: node for node in graph_def.node}

    def add_op(node: tf.NodeDef):
        if graph.get_op(node.name) is not None:
            return
        input_tensors: List[TFTensor] = []
        control_input_nodes: List[str] = []
        input: str
        for input in node.input:
            if input.startswith("^"):
                control_input_nodes.append(input[1:])
            else:
                names = input.split(":")
                assert len(names) == 1 or len(names) == 2
                if len(names) == 1:
                    input_tensors.append(TFTensor(names[0], 0))
                else:
                    input_tensors.append(TFTensor(names[0], int(names[1])))
        for input_tensor in input_tensors:
            add_op(name_to_node[input_tensor.op])
        for control_input_node in control_input_nodes:
            add_op(name_to_node[control_input_node])

        attrs = {
            attr_name: from_attr_proto(node.attr[attr_name]) for attr_name in node.attr
        }

        op = create_op(
            type=node.op,
            name=node.name,
            attrs=attrs,
            inputs=OrderedDict(
                (name, serialize_type(dtype))
                for name, dtype in get_input_args(tf_graph, node)
            ),
            outputs=OrderedDict(
                (name, serialize_type(dtype))
                for name, dtype in get_output_args(tf_graph, node)
            ),
        )
        init_op_attrs(op, node)
        op.attrs["device"] = node.device
        graph.add_op(op)
        for index, input_tensor in enumerate(input_tensors):
            graph.create_edge(
                graph.get_op(input_tensor.op).output_port(input_tensor.output_index),
                op.input_port(index),
            )
        for control_input_name in control_input_nodes:
            graph.create_control_edge(graph.get_op(control_input_name), op)

    for node in graph_def.node:
        add_op(node)
    for key in ["versions", "library"]:
        if graph_def.HasField(key):
            graph.attrs[key] = getattr(graph_def, key)
    return graph


def import_from_pbtxt(file: Union[str, Path]) -> Graph:
    file = Path(file)
    graph_def = tf.GraphDef()
    google.protobuf.text_format.Parse(file.read_text(), graph_def)
    return import_from_graph_def(graph_def)


def extract_meta_graph_fields(graph, meta_graph):
    new_meta_info_def = tf.MetaGraphDef.MetaInfoDef()
    new_meta_info_def.CopyFrom(meta_graph.meta_info_def)
    new_meta_info_def.stripped_op_list.Clear()
    graph.attrs["meta_info_def"] = new_meta_info_def
    graph.attrs["signature_def"] = dict(meta_graph.signature_def)
    graph.attrs["asset_file_def"] = list(meta_graph.asset_file_def)
    if meta_graph.HasField("object_graph_def"):
        graph.attrs["object_graph_def"] = meta_graph.object_graph_def


def construct_meta_graph_fields(graph, meta_graph, graph_def):
    if "meta_info_def" in graph.attrs:
        meta_graph.meta_info_def.CopyFrom(graph.attrs["meta_info_def"])
    meta_graph.meta_info_def.stripped_op_list.CopyFrom(
        stripped_op_list_for_graph(graph_def)
    )
    if "signature_def" in graph.attrs:
        for key, proto in graph.attrs["signature_def"]:
            meta_graph.signature_def[key].CopyFrom(proto)
    if "asset_file_def" in graph.attrs:
        for proto in graph.attrs["asset_file_def"]:
            new_proto = meta_graph.asset_file_def.add()
            new_proto.CopyFrom(proto)
    if "object_graph_def" in graph.attrs:
        meta_graph.object_graph_def.CopyFrom(graph.attrs["object_graph_def"])


def import_from_meta_graph(
    meta_graph: Union[tf.MetaGraphDef, str, bytes, Path],
    checkpoint: Union[str, Path] = None,
    session: tf.Session = None,
) -> Graph:
    meta_graph = to_proto(meta_graph, tf.MetaGraphDef)
    with tf.Graph().as_default() as tf_graph, ExitStack() as exit_stack:
        saver = tf.train.import_meta_graph(meta_graph)
        if checkpoint is not None:
            if session is None:
                session = tf.Session()
                exit_stack.enter_context(session)
            saver.restore(session, str(checkpoint))
        graph = import_from_graph(tf_graph, saver.as_saver_def(), session)
        extract_meta_graph_fields(graph, meta_graph)
        return graph


def import_from_checkpoint(path: Union[str, Path]) -> Graph:
    path = str(path)
    return import_from_meta_graph(path + ".meta", path)


def import_from_saved_model(path: Union[str, Path], tags: List[str]) -> Graph:
    path = str(path)
    with tf.Graph().as_default() as tf_graph:
        with tf.Session() as session:
            meta_graph = tf.saved_model.load(session, tags, path)
            graph = import_from_graph(tf_graph, meta_graph.saver_def, session)
            extract_meta_graph_fields(graph, meta_graph)
            return graph


def get_dtype_proto(node_def, op_def, arg):
    def with_number_attr(dtype):
        if len(arg.number_attr) != 0:
            for attr in op_def.attr:
                if attr.name == arg.number_attr:
                    return [dtype] * node_def.attr[attr.name].i
            raise AssertionError()
        else:
            return dtype

    def with_ref(dtype):
        if arg.is_ref:
            return dtype._as_ref
        else:
            return dtype

    if len(arg.type_attr) != 0:
        for attr in op_def.attr:
            if attr.name == arg.type_attr:
                return with_number_attr(
                    with_ref(tf.as_dtype(node_def.attr[attr.name].type))
                )
        raise AssertionError()
    elif len(arg.type_list_attr) != 0:
        for attr in op_def.attr:
            if attr.name == arg.type_list_attr:
                return [
                    with_ref(tf.as_dtype(dtype))
                    for dtype in node_def.attr[attr.name].list.type
                ]
        raise AssertionError()
    else:
        assert arg.type != types_pb2.DT_INVALID
        return with_number_attr(with_ref(tf.as_dtype(arg.type)))


def get_op_def(tf_graph, node_def):
    return tf_graph._get_op_def(node_def.op)


def flatten_args(args):
    flat_args = []
    for name, dtype in args:
        if isinstance(dtype, list):
            for index, child_dtype in enumerate(dtype):
                flat_args.append((f"{name}/{index}", child_dtype))
        else:
            flat_args.append((name, dtype))
    return flat_args


def get_input_args(tf_graph, node_def):
    op_def = get_op_def(tf_graph, node_def)
    return flatten_args(
        (input_arg.name, get_dtype_proto(node_def, op_def, input_arg))
        for input_arg in op_def.input_arg
    )


def get_output_args(tf_graph, node_def):
    op_def = get_op_def(tf_graph, node_def)
    return flatten_args(
        (output_arg.name, get_dtype_proto(node_def, op_def, output_arg))
        for output_arg in op_def.output_arg
    )


def get_dtypes(tf_graph, node_def):
    op_def = get_op_def(tf_graph, node_def)
    return [
        get_dtype_proto(node_def, op_def, output_arg)
        for output_arg in op_def.output_arg
    ]


def from_attr_proto(attr_value: tf.AttrValue) -> Any:
    field_name = attr_value.WhichOneof("value")
    if field_name == "s":
        return attr_value.s
    elif field_name == "b":
        return attr_value.b
    elif field_name == "i":
        return attr_value.i
    elif field_name == "f":
        return attr_value.f
    elif field_name == "type":
        return tf.as_dtype(attr_value.type)
    elif field_name == "shape":
        return tensor_shape.as_shape(attr_value.shape)
    elif field_name == "tensor":
        return attr_value.tensor
    elif field_name == "func":
        return attr_value.func
    elif field_name == "placeholder":
        return attr_value.placeholder
    elif field_name == "list":
        list_value = attr_value.list
        if len(list_value.s) != 0:
            return [value for value in list_value.s]
        elif len(list_value.b) != 0:
            return [value for value in list_value.b]
        elif len(list_value.i) != 0:
            return [value for value in list_value.i]
        elif len(list_value.f) != 0:
            return [value for value in list_value.f]
        elif len(list_value.type) != 0:
            return [tf.as_dtype(value) for value in list_value.type]
        elif len(list_value.shape) != 0:
            return [tensor_shape.as_shape(value) for value in list_value.shape]
        elif len(list_value.tensor) != 0:
            return [value for value in list_value.tensor]
        elif len(list_value.func) != 0:
            return [value for value in list_value.func]
        else:
            return []


def get_tensor_name_by_port(port: OutputPort, compact: bool = False) -> str:
    port_index = list(port.op.name_to_output_port).index(port.name)
    if compact and port_index == 0:
        return port.op.name
    else:
        return f"{port.op.name}:{port_index}"


def export_to_graph(
    graph: Graph, session: tf.Session = None
) -> Tuple[tf.Graph, tf.train.Saver, tf.Session]:
    def export_fn(tf_graph, session):
        if "saver_def" in graph.attrs:
            meta_graph = tf.MetaGraphDef()
            graph_def = export_to_graph_def(graph)
            meta_graph.graph_def.CopyFrom(graph_def)
            meta_graph.saver_def.CopyFrom(graph.attrs["saver_def"])
            construct_meta_graph_fields(graph, meta_graph, graph_def)
            saver = tf.train.import_meta_graph(meta_graph)
        else:
            tf.graph_util.import_graph_def(export_to_graph_def(graph), name="")
            saver = None
        variables = {}
        for op in graph.ops:
            if op.type == "VariableV2":
                variable_def = variable_pb2.VariableDef()
                variable_def.variable_name = f"{op.name}:0"
                variable_def.initial_value_name = op.attrs["initial_value_name"]
                variable_def.initializer_name = op.attrs["initializer_name"]
                variable_def.snapshot_name = op.attrs["snapshot_name"]
                variable_def.is_resource = op.attrs["is_resource"]
                variable_def.trainable = op.attrs["trainable"]
                variable = tf.Variable.from_proto(variable_def)
                variables[variable.name] = variable
        for name in graph.attrs:
            if name.startswith("collection/"):
                collection_name = name[len("collection/") :]
                collection = graph.attrs[name]
                tf_collection = tf_graph.get_collection_ref(collection_name)
                if collection_name in tf.GraphKeys._VARIABLE_COLLECTIONS:
                    tf_collection.extend(
                        [variables[f"{op.name}:0"] for op in collection]
                    )
                elif isinstance(collection[0], Op):
                    tf_collection.extend(
                        [tf_graph.get_operation_by_name(op.name) for op in collection]
                    )
                elif isinstance(collection[0], OutputPort):
                    tf_collection.extend(
                        [
                            tf_graph.get_tensor_by_name(get_tensor_name_by_port(port))
                            for port in collection
                        ]
                    )
                else:
                    tf_collection.extend(collection)
        initialized_variables = {
            f"{op.name}:0": op.attrs["value"]
            for op in graph.ops
            if op.type == "VariableV2" and "value" in op.attrs
        }
        initializers = [
            variables[name].initializer for name in initialized_variables.keys()
        ]
        feed_dict = {
            variables[name].initializer.inputs[1]: value
            for name, value in initialized_variables.items()
        }
        if len(initializers) != 0:
            session.run(initializers, feed_dict)
        return tf_graph, saver, session

    if not graph.namespace.belong_to(tf_namespace()):
        raise MismatchNamespaceError(expect=tf_namespace(), actual=graph.namespace)
    if session is not None:
        tf_graph = session.graph
        with tf_graph.as_default():
            return export_fn(tf_graph, session)
    else:
        with tf.Graph().as_default() as tf_graph:
            session = tf.Session()
            return export_fn(tf_graph, session)


def export_to_graph_def(graph: Graph) -> tf.GraphDef:
    if not graph.namespace.belong_to(tf_namespace()):
        raise MismatchNamespaceError(expect=tf_namespace(), actual=graph.namespace)
    tf_graph = tf.Graph()
    graph_def = tf_graph.as_graph_def()
    for key in ["versions", "library"]:
        if key in graph.attrs:
            getattr(graph_def, key).CopyFrom(graph.attrs[key])
    for op in graph.sorted_ops:
        attrs = without_internal_attrs(op.attrs, ["device", "experimental_debug_info"])
        node = graph_def.node.add()
        node.name = op.name
        node.op = op.type
        if "device" in op.attrs:
            node.device = op.attrs["device"]
        if "experimental_debug_info" in op.attrs:
            node.experimental_debug_info.CopyFrom(op.attrs["experimental_debug_info"])
        if op.type == "VariableV2":
            for key in [
                "value",
                "initial_value_name",
                "initializer_name",
                "snapshot_name",
                "is_resource",
                "trainable",
            ]:
                if key in attrs:
                    del attrs[key]
        for name, attr_value in to_attrs_proto(
            tf_graph._get_op_def(op.type), op.type, attrs
        ).items():
            node.attr[name].CopyFrom(attr_value)
        for port in op.input_ports:
            src_port = port.in_edges[0].src
            node.input.append(get_tensor_name_by_port(src_port, compact=True))
        for op in op.control_dependencies:
            node.input.append(f"^{op.name}")
    return graph_def


def export_to_pbtxt(graph: Graph, file: Union[str, Path] = None) -> str:
    graph_def = export_to_graph_def(graph)
    graph_pbtxt = google.protobuf.text_format.MessageToString(graph_def)
    if file is not None:
        file = Path(file)
        file.write_text(graph_pbtxt)
    return graph_pbtxt


def export_to_checkpoint(graph: Graph, path: Union[str, Path]) -> None:
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    tf_graph, saver, session = export_to_graph(graph)
    with session, tf_graph.as_default():
        if saver is None:
            saver = tf.train.Saver()
        saver.save(session, str(path))


def export_to_saved_model(
    graph: Graph, path: Union[str, Path], tags: List[str]
) -> None:
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    tf_graph, saver, session = export_to_graph(graph)
    path = str(path)
    builder = tf.saved_model.builder.SavedModelBuilder(path)
    with tf_graph.as_default(), session:
        builder.add_meta_graph_and_variables(
            session,
            tags,
            strip_default_attrs=True,
            saver=saver,
        )
    builder.save()


def to_attrs_proto(
    op_def: OpDef,
    op_type_name: str,
    attrs: Dict[str, Any],
) -> Dict[str, Any]:
    # Convert attr values to AttrValue protos.
    attr_protos = {}
    attr_defs = {attr_def.name: attr_def for attr_def in op_def.attr}
    for key, value in attrs.items():
        attr_value = tf.AttrValue()
        if key in attr_defs:
            attr_def = attr_defs[key]
        elif value is None:
            attr_protos[key] = attr_value
            continue
        else:
            attr_def = OpDef.AttrDef()
            if isinstance(value, (str, bytes)):
                attr_def.type = "string"
            elif isinstance(value, float):
                attr_def.type = "float"
            elif isinstance(value, bool):
                attr_def.type = "bool"
            # bool is a subclass of int, so we should check bool before checking int
            elif isinstance(value, int):
                attr_def.type = "int"
            elif isinstance(value, tf.DType):
                attr_def.type = "type"
            elif isinstance(value, tf.TensorShape):
                attr_def.type = "shape"
            elif isinstance(value, tensor_pb2.TensorProto):
                attr_def.type = "tensor"
            elif isinstance(value, tf.NameAttrList):
                attr_def.type = "func"
            elif isinstance(value, list) and len(value) == 0:
                attr_value.list.SetInParent()
                attr_protos[key] = attr_value
                continue
            elif isinstance(value, list) and isinstance(value[0], (str, bytes)):
                attr_def.type = "list(string)"
            elif isinstance(value, list) and isinstance(value[0], bool):
                attr_def.type = "list(bool)"
            # bool is a subclass of int, so we should check bool before checking int
            elif isinstance(value, list) and isinstance(value[0], int):
                attr_def.type = "list(int)"
            elif isinstance(value, list) and isinstance(value[0], float):
                attr_def.type = "list(float)"
            elif isinstance(value, list) and isinstance(value[0], tf.DType):
                attr_def.type = "list(type)"
            elif isinstance(value, list) and isinstance(value[0], tf.TensorShape):
                attr_def.type = "list(shape)"
            elif isinstance(value, list) and isinstance(
                value[0], tensor_pb2.TensorProto
            ):
                attr_def.type = "list(tensor)"
            else:
                raise AssertionError(f"{value} has unsupported type")
        if attr_def.HasField("default_value") and value is None:
            attr_value.CopyFrom(attr_def.default_value)
            attr_protos[key] = attr_value
            continue
        if attr_def.type.startswith("list("):
            if not _IsListValue(value):
                raise TypeError("Expected list for attr " + key)
            if attr_def.has_minimum:
                if len(value) < attr_def.minimum:
                    raise ValueError(
                        "Attr '%s' of '%s' Op passed list of length %d "
                        "less than minimum %d."
                        % (key, op_type_name, len(value), attr_def.minimum)
                    )
            attr_value.list.SetInParent()
        if attr_def.type == "string":
            attr_value.s = _MakeStr(value, key)
            if attr_def.HasField("allowed_values"):
                if attr_value.s not in attr_def.allowed_values.list.s:
                    raise ValueError(
                        "Attr '%s' of '%s' Op passed string '%s' not in: \"%s\"."
                        % (
                            key,
                            op_type_name,
                            compat.as_text(attr_value.s),
                            '", "'.join(
                                map(compat.as_text, attr_def.allowed_values.list.s)
                            ),
                        )
                    )
        elif attr_def.type == "list(string)":
            attr_value.list.s.extend([_MakeStr(x, key) for x in value])
            if attr_def.HasField("allowed_values"):
                for x in attr_value.list.s:
                    if x not in attr_def.allowed_values.list.s:
                        raise ValueError(
                            "Attr '%s' of '%s' Op passed string '%s' not in: \"%s\"."
                            % (
                                key,
                                op_type_name,
                                compat.as_text(x),
                                '", "'.join(
                                    map(compat.as_text, attr_def.allowed_values.list.s)
                                ),
                            )
                        )
        elif attr_def.type == "int":
            attr_value.i = _MakeInt(value, key)
            if attr_def.has_minimum:
                if attr_value.i < attr_def.minimum:
                    raise ValueError(
                        "Attr '%s' of '%s' Op passed %d less than minimum %d."
                        % (key, op_type_name, attr_value.i, attr_def.minimum)
                    )
        elif attr_def.type == "list(int)":
            attr_value.list.i.extend([_MakeInt(x, key) for x in value])
        elif attr_def.type == "float":
            attr_value.f = _MakeFloat(value, key)
        elif attr_def.type == "list(float)":
            attr_value.list.f.extend([_MakeFloat(x, key) for x in value])
        elif attr_def.type == "bool":
            attr_value.b = _MakeBool(value, key)
        elif attr_def.type == "list(bool)":
            attr_value.list.b.extend([_MakeBool(x, key) for x in value])
        elif attr_def.type == "type":
            attr_value.type = _MakeType(value, attr_def)
        elif attr_def.type == "list(type)":
            attr_value.list.type.extend([_MakeType(x, attr_def) for x in value])
        elif attr_def.type == "shape":
            attr_value.shape.CopyFrom(_MakeShape(value, key))
        elif attr_def.type == "list(shape)":
            attr_value.list.shape.extend([_MakeShape(x, key) for x in value])
        elif attr_def.type == "tensor":
            attr_value.tensor.CopyFrom(_MakeTensor(value, key))
        elif attr_def.type == "list(tensor)":
            attr_value.list.tensor.extend([_MakeTensor(x, key) for x in value])
        elif attr_def.type == "func":
            if isinstance(value, tf.NameAttrList):
                attr_value.func.CopyFrom(value)
            elif isinstance(value, compat.bytes_or_text_types):
                attr_value.func.name = value
            else:
                value.add_to_graph(tf.get_default_graph())
                attr_value.func.name = value.name
        else:
            raise TypeError("Unrecognized Attr type " + attr_def.type)

        attr_protos[key] = attr_value
    return attr_protos


class AmandaHook(tf.train.SessionRunHook):
    def __init__(self, context: EventContext):
        self.context = context

    def begin(self):
        if self.context.is_registered(on_graph_loaded):
            tf_graph = tf.get_default_graph()
            graph = import_from_graph(tf_graph)
            self.context.trigger(on_graph_loaded, graph=graph)
            new_graph = self.context["graph"]
            new_tf_graph, _, session = export_to_graph(new_graph)
            session.close()
            if new_tf_graph != tf_graph:
                new_tf_graph._device_function_stack = tf_graph._device_function_stack
                gc.collect()
                replace_all_refs(tf_graph, new_tf_graph)


class EstimatorAdapter(Adapter):
    def __init__(self):
        super(EstimatorAdapter, self).__init__(namespace="tensorflow")

    def apply(self, target: tf.estimator.Estimator, context: EventContext) -> None:
        def train_with_tools(
            target_self,
            input_fn,
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None,
        ):
            return train(
                input_fn,
                [amanda_hook] if hooks is None else [amanda_hook, *hooks],
                steps,
                max_steps,
                saving_listeners,
            )

        def predict_with_tools(
            target_self,
            input_fn,
            predict_keys=None,
            hooks=None,
            checkpoint_path=None,
            yield_single_examples=True,
        ):
            return predict(
                input_fn,
                predict_keys,
                [amanda_hook] if hooks is None else [amanda_hook, *hooks],
                checkpoint_path,
                yield_single_examples,
            )

        def evaluate_with_tools(
            target_self,
            input_fn,
            steps=None,
            hooks=None,
            checkpoint_path=None,
            name=None,
        ):
            return evaluate(
                input_fn,
                steps,
                [amanda_hook] if hooks is None else [amanda_hook, *hooks],
                checkpoint_path,
                name,
            )

        amanda_hook = AmandaHook(context)
        train = target.train
        target.train = MethodType(train_with_tools, target)
        predict = target.predict
        target.predict = MethodType(predict_with_tools, target)
        evaluate = target.evaluate
        target.evaluate = MethodType(evaluate_with_tools, target)


get_adapter_registry().register_adapter(tf.estimator.Estimator, EstimatorAdapter())

_py_funcs: List[Any] = []


def import_from_tf_func(tf_func):
    def amanda_func(graph: Graph):
        def func(*args, **kwargs):
            if (
                hasattr(tf_func, "_tf_api_names_v1")
                and len(tf_func._tf_api_names_v1) == 1
                and tf_func._tf_api_names_v1[0] == "py_func"
            ):
                # The Global registry for py functions
                # `tensorflow.python.ops.script_ops._py_funcs`
                # is a `WeakValueDictionary`, we keep strong reference to
                # `func` in `tf.py_func` to ensure its lifetime is longer enough when
                # we need to export this graph.
                _py_funcs.append(args[0])
            args_dict = dict(enumerate(args))
            tf_args = list(args)
            tf_kwargs = dict(kwargs)
            src_ports: Dict[str, OutputPort] = {}
            tf_placeholders: List[tf.Tensor] = []

            def to_tf_tensor(output_port: OutputPort) -> tf.Tensor:
                placeholder = tf.placeholder(output_port.type.raw)
                tf_placeholders.append(placeholder)
                src_ports[placeholder.op.name] = output_port
                return placeholder

            def get_args(name: Union[int, str]) -> Dict[Any, Any]:
                return args_dict if isinstance(name, int) else kwargs

            def get_tf_args(name: Union[int, str]) -> Dict[Any, Any]:
                return tf_args if isinstance(name, int) else tf_kwargs

            with tf.Graph().as_default() as tf_graph:
                # mimic tensorflow.python.framework.ops.Graph.unique_name
                tf_graph._names_in_use = {lower_name(op.name): 1 for op in graph.ops}

                all_args = {**args_dict, **kwargs}
                for name in all_args:
                    arg = get_args(name)[name]
                    if isinstance(arg, OutputPort):
                        get_tf_args(name)[name] = to_tf_tensor(arg)
                    if isinstance(arg, list):
                        arg_list = arg
                        for i, arg in enumerate(arg_list):
                            if isinstance(arg, OutputPort):
                                get_tf_args(name)[name][i] = to_tf_tensor(arg)
                output_tensor: tf.Tensor = tf_func(*tf_args, **tf_kwargs)
                new_graph = import_from_graph(tf_graph)
                for name in new_graph.attrs:
                    if name.startswith("collection/"):
                        new_collection = new_graph.attrs[name]
                        if name in graph.attrs:
                            graph.attrs[name].extend(new_collection)
                        else:
                            graph.attrs[name] = new_collection
                placeholders = [
                    new_graph.get_op(placeholder.op.name)
                    for placeholder in tf_placeholders
                ]
                for op in new_graph.ops:
                    if op not in placeholders:
                        graph.add_op(op)
                for edge in new_graph.edges:
                    if edge.src.op in placeholders:
                        graph.create_edge(src_ports[edge.src.op.name], edge.dst)
                    else:
                        graph.create_edge(edge.src, edge.dst)
                output_op = graph.get_op(output_tensor.op.name)
                return output_op.output_port(output_tensor.value_index)

        return func

    return amanda_func


import_types: Dict[str, Callable] = {
    "tensorflow_pbtxt": import_from_pbtxt,
    "tensorflow_checkpoint": import_from_checkpoint,
    "tensorflow_saved_model": import_from_saved_model,
}

export_types: Dict[str, Callable] = {
    "tensorflow_pbtxt": export_to_pbtxt,
    "tensorflow_checkpoint": export_to_checkpoint,
    "tensorflow_saved_model": export_to_saved_model,
}
