import jsondiff
import numpy as np
import onnx
import onnxruntime as rt
import pytest
from onnx import numpy_helper

import amanda
from amanda.conversion.onnx import export_to_model_def, import_from_model_def
from amanda.conversion.utils import (
    diff_graph_def,
    diff_proto,
    repeated_fields_to_dict,
    to_proto,
)
from amanda.io.file import ensure_dir, root_dir


@pytest.fixture(params=["mobilenetv2-1.0", "resnet18-v1-7"])
def arch_name(request):
    return request.param


def test_onnx_import_export(arch_name):
    model_dir = root_dir() / "downloads" / "onnx_model" / arch_name
    model_path = str(model_dir / f"{arch_name}.onnx")
    new_model_path = root_dir() / "tmp" / "onnx_model" / arch_name / "model.onnx"
    graph_path = root_dir() / "tmp" / "onnx_graph" / arch_name / arch_name
    model_def = onnx.load(str(model_path))
    session = rt.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_tensor = numpy_helper.to_array(
        to_proto(model_dir / "test_data_set_0" / "input_0.pb", onnx.TensorProto)
    )
    output_tensor = numpy_helper.to_array(
        to_proto(model_dir / "test_data_set_0" / "output_0.pb", onnx.TensorProto)
    )
    prediction = session.run(None, {input_name: input_tensor})[0]
    np.testing.assert_allclose(prediction, output_tensor, atol=1.0e-3)
    graph = import_from_model_def(model_path)
    amanda.io.save_to_proto(graph, graph_path)
    graph = amanda.io.load_from_proto(graph_path)
    amanda.io.save_to_yaml(graph, graph_path)
    graph = amanda.io.load_from_yaml(graph_path)
    new_model_def = export_to_model_def(graph, ensure_dir(new_model_path))
    new_session = rt.InferenceSession(str(new_model_path))
    new_prediction = new_session.run(None, {input_name: input_tensor})[0]
    np.testing.assert_allclose(prediction, new_prediction)
    assert new_model_def == onnx.load(str(new_model_path))
    for key in [
        "float_data",
        "int32_data",
        "string_data",
        "int64_data",
        "raw_data",
        "double_data",
        "uint64_data",
    ]:
        for initializer in model_def.graph.initializer:
            initializer.ClearField(key)
        for initializer in model_def.graph.sparse_initializer:
            initializer.ClearField(key)
        for initializer in new_model_def.graph.initializer:
            initializer.ClearField(key)
        for initializer in new_model_def.graph.sparse_initializer:
            initializer.ClearField(key)
    initializer = repeated_fields_to_dict(model_def.graph.initializer)
    model_def.graph.ClearField("initializer")
    new_initializer = repeated_fields_to_dict(new_model_def.graph.initializer)
    new_model_def.graph.ClearField("initializer")
    assert (
        jsondiff.diff(
            repeated_fields_to_dict(model_def.graph.input),
            repeated_fields_to_dict(new_model_def.graph.input),
            syntax="explicit",
        )
        == {}
    )
    assert (
        jsondiff.diff(
            repeated_fields_to_dict(model_def.graph.output),
            repeated_fields_to_dict(new_model_def.graph.output),
            syntax="explicit",
        )
        == {}
    )
    model_def.graph.ClearField("input")
    new_model_def.graph.ClearField("input")
    model_def.graph.ClearField("output")
    new_model_def.graph.ClearField("output")
    assert diff_graph_def(model_def.graph, new_model_def.graph) == {}
    assert jsondiff.diff(initializer, new_initializer, syntax="explicit") == {}
    model_def.ClearField("graph")
    new_model_def.ClearField("graph")
    assert diff_proto(model_def, new_model_def) == {}
