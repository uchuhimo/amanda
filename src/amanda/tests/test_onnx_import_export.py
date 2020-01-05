import numpy as np
import onnx
import onnxruntime as rt
import pytest
from onnx import numpy_helper

from amanda.conversion.utils import to_proto
from amanda.tests.utils import root_dir


@pytest.fixture(params=["mobilenetv2-1.0", "resnet18v2"])
def arch_name(request):
    return request.param


def test_onnx_import_export(arch_name):
    model_dir = root_dir() / "tmp" / "onnx_model" / arch_name
    sess = rt.InferenceSession(str(model_dir / f"{arch_name}.onnx"))
    input_name = sess.get_inputs()[0].name
    input_tensor = numpy_helper.to_array(
        to_proto(model_dir / "test_data_set_0" / "input_0.pb", onnx.TensorProto)
    )
    output_tensor = numpy_helper.to_array(
        to_proto(model_dir / "test_data_set_0" / "output_0.pb", onnx.TensorProto)
    )
    prediction = sess.run(None, {input_name: input_tensor})[0]
    assert np.allclose(prediction, output_tensor, atol=1.0e-3)
