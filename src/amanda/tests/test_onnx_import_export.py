import numpy as np
import onnx
import onnxruntime as rt
import pytest
from onnx import numpy_helper

from amanda.conversion.onnx import export_to_model_def, import_from_model_def
from amanda.conversion.utils import diff_graph_def, diff_proto, to_proto
from amanda.tests.utils import root_dir


@pytest.fixture(params=["mobilenetv2-1.0", "resnet18v2"])
def arch_name(request):
    return request.param


def test_onnx_import_export(arch_name, tmp_path):
    model_dir = root_dir() / "downloads" / "onnx_model" / arch_name
    model_path = str(model_dir / f"{arch_name}.onnx")
    new_model_path = tmp_path / "model.onnx"
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
    assert np.allclose(prediction, output_tensor, atol=1.0e-3)
    graph = import_from_model_def(model_path).to_default_namespace()
    new_model_def = export_to_model_def(graph, new_model_path)
    new_session = rt.InferenceSession(str(new_model_path))
    new_prediction = new_session.run(None, {input_name: input_tensor})[0]
    assert np.allclose(prediction, new_prediction)
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
    assert diff_graph_def(model_def.graph, new_model_def.graph) == {}
    model_def.ClearField("graph")
    new_model_def.ClearField("graph")
    assert diff_proto(model_def, new_model_def) == {}
