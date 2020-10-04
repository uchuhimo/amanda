import base64
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Union

import h5py
import numpy as np
from ruamel.yaml import YAML, Constructor, Node, Representer, ScalarNode

from amanda.graph import Graph
from amanda.io.graph_pb2 import NodeDef
from amanda.io.proto import from_proto, to_proto
from amanda.io.text import (
    DTypeWrapper,
    RefWrapper,
    from_proto_to_text,
    from_text_to_proto,
)
from amanda.io.value_pb2 import DataTypeValue, ListValue, SerializedValue, Struct, Value


def extract_from_value(proto: Value, bins: Dict[str, bytes]) -> None:
    kind = proto.WhichOneof("kind")
    if kind == "bytes_value":
        if len(proto.bytes_value) >= 128:
            m = hashlib.md5()
            m.update(proto.bytes_value)
            digest = base64.urlsafe_b64encode(m.digest()).decode("utf-8")
            bins[digest] = proto.bytes_value
            proto.ref_value.SetInParent()
            proto.ref_value.schema = "h5"
            proto.ref_value.value = digest
    elif kind == "struct_value":
        extract_from_dict(proto.struct_value, bins)
    elif kind == "list_value":
        extract_from_list(proto.list_value, bins)
    elif kind == "serialized_value":
        extract_from_serialized_value(proto.serialized_value, bins)
    elif kind == "type_value":
        extract_from_dtype(proto.type_value, bins)


def extract_from_serialized_value(
    proto: SerializedValue, bins: Dict[str, bytes]
) -> None:
    extract_from_dtype(proto.type, bins)
    extract_from_value(proto.value, bins)


def extract_from_dtype(proto: DataTypeValue, bins: Dict[str, bytes]) -> None:
    extract_from_dict(proto.attrs, bins)


def extract_from_dict(proto: Struct, bins: Dict[str, bytes]) -> None:
    for key in proto.fields:
        extract_from_value(proto.fields[key], bins)


def extract_from_list(proto: ListValue, bins: Dict[str, bytes]) -> None:
    for value in proto.values:
        extract_from_value(value, bins)


def extract_bins(proto: NodeDef, bins: Dict[str, bytes]) -> None:
    extract_from_dict(proto.attrs, bins)
    for port in proto.input_ports:
        extract_from_dict(port.type.attrs, bins)
    for edge in proto.edges:
        extract_from_dict(edge.attrs, bins)
    for op in proto.ops:
        extract_bins(op, bins)


def save_to_h5(proto: NodeDef, path: str):
    bins: Dict[str, np.ndarray] = {}
    extract_bins(proto, bins)
    with h5py.File(path, "w") as file:
        for digest, value in bins.items():
            file[digest] = np.frombuffer(value, dtype=np.uint8)


def save_to_proto(graph: Graph, path: Union[str, Path]) -> None:
    proto_path = ensure_dir(str(path) + ".amanda.pb")
    bin_path = ensure_dir(str(path) + ".amanda.h5")
    proto = to_proto(graph)
    save_to_h5(proto, bin_path)
    with open(proto_path, "wb") as file:
        file.write(proto.SerializeToString())


def recover_to_value(proto: Value, bins: Dict[str, bytes]) -> None:
    kind = proto.WhichOneof("kind")
    if kind == "ref_value":
        if proto.ref_value.schema == "h5":
            proto.bytes_value = bins[proto.ref_value.value]
    elif kind == "struct_value":
        recover_to_dict(proto.struct_value, bins)
    elif kind == "list_value":
        recover_to_list(proto.list_value, bins)
    elif kind == "serialized_value":
        recover_to_serialized_value(proto.serialized_value, bins)
    elif kind == "type_value":
        recover_to_dtype(proto.type_value, bins)


def recover_to_serialized_value(proto: SerializedValue, bins: Dict[str, bytes]) -> None:
    recover_to_dtype(proto.type, bins)
    recover_to_value(proto.value, bins)


def recover_to_dtype(proto: DataTypeValue, bins: Dict[str, bytes]) -> None:
    recover_to_dict(proto.attrs, bins)


def recover_to_dict(proto: Struct, bins: Dict[str, bytes]) -> None:
    for key in proto.fields:
        recover_to_value(proto.fields[key], bins)


def recover_to_list(proto: ListValue, bins: Dict[str, bytes]) -> None:
    for value in proto.values:
        recover_to_value(value, bins)


def recover_bins(proto: NodeDef, bins: Dict[str, bytes]) -> None:
    recover_to_dict(proto.attrs, bins)
    for port in proto.input_ports:
        recover_to_dict(port.type.attrs, bins)
    for edge in proto.edges:
        recover_to_dict(edge.attrs, bins)
    for op in proto.ops:
        recover_bins(op, bins)


def load_from_h5(proto: NodeDef, path: str):
    with h5py.File(path, "r") as file:
        bins = {digest: file[digest][()].tobytes() for digest in file}
    recover_bins(proto, bins)


def load_from_proto(path: Union[str, Path]) -> Graph:
    proto_path = str(path) + ".amanda.pb"
    bin_path = ensure_dir(str(path) + ".amanda.h5")
    proto = NodeDef()
    with open(proto_path, "rb") as file:
        proto.ParseFromString(file.read())
    load_from_h5(proto, bin_path)
    return from_proto(proto)


def save_to_yaml(graph: Graph, path: Union[str, Path]) -> None:
    yaml_path = ensure_dir(str(path) + ".amanda.yaml")
    bin_path = ensure_dir(str(path) + ".amanda.h5")
    proto = to_proto(graph)
    save_to_h5(proto, bin_path)
    with open(yaml_path, "w") as file:
        yaml.dump(from_proto_to_text(proto), stream=file)


def load_from_yaml(path: Union[str, Path]) -> Graph:
    yaml_path = str(path) + ".amanda.yaml"
    bin_path = ensure_dir(str(path) + ".amanda.h5")
    with open(yaml_path, "r") as file:
        text = yaml.load(file)
    proto = from_text_to_proto(text)
    load_from_h5(proto, bin_path)
    return from_proto(proto)


def dtype_representer(dumper: Representer, data: DTypeWrapper):
    dtype = data.dtype
    if isinstance(dtype, str):
        return dumper.represent_scalar("!type", dtype)
    else:
        return dumper.represent_mapping("!type", dtype)


def dtype_constructor(loader: Constructor, node: Node):
    if isinstance(node, ScalarNode):
        return DTypeWrapper(loader.construct_scalar(node))
    else:
        with contextmanager(loader.construct_yaml_map)(node) as dtype:
            wrapper = DTypeWrapper(dtype)
        return wrapper


def ref_representer(dumper: Representer, data: RefWrapper):
    return dumper.represent_scalar(f"!{data.tag}", data.ref)


def ref_constructor(loader: Constructor, node: Node):
    return RefWrapper(loader.construct_scalar(node), tag=node.tag)


def binary_representer(dumper: Representer, data: bytes):
    binary = base64.encodebytes(data).decode("ascii")
    return dumper.represent_scalar("tag:yaml.org,2002:binary", binary)


def list_representer(dumper: Representer, data: List[Any]):
    flow_style = True
    for item in data:
        if item is not None and not isinstance(item, (int, float, bool)):
            flow_style = False
            break
    if flow_style:
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
    else:
        return dumper.represent_list(data)


def none_representer(dumper: Representer, data):
    return dumper.represent_scalar("tag:yaml.org,2002:null", "null")


yaml = YAML()
yaml.sort_base_mapping_type_on_output = False
yaml.representer.add_representer(DTypeWrapper, dtype_representer)
yaml.representer.add_representer(RefWrapper, ref_representer)
yaml.representer.add_representer(bytes, binary_representer)
yaml.representer.add_representer(list, list_representer)
yaml.representer.add_representer(type(None), none_representer)
yaml.constructor.add_constructor("!type", dtype_constructor)
yaml.constructor.add_constructor("!ref", ref_constructor)
yaml.constructor.add_constructor("!node", ref_constructor)
yaml.constructor.add_constructor("!in", ref_constructor)
yaml.constructor.add_constructor("!out", ref_constructor)
yaml.constructor.add_constructor("!edge", ref_constructor)


def ensure_dir(path: str) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except FileExistsError:
            pass
    return path


def root_dir() -> Path:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return Path(current_dir).parents[2]
