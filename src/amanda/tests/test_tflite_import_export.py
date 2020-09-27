import numpy as np
import pytest
import tensorflow as tf

from amanda.io.file import root_dir


@pytest.fixture(
    params=["mobilenet_v2_1.0_224_quant", "mobilenet_v2_1.0_224", "nasnet_mobile"]
)
def arch_name(request):
    return request.param


@pytest.mark.slow
def test_tflite_import_export(arch_name):
    model_dir = root_dir() / "downloads" / "tflite_model" / arch_name
    interpreter = tf.lite.Interpreter(model_path=str(model_dir / f"{arch_name}.tflite"))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]
    input_data = np.array(
        np.random.random_sample(input_shape), dtype=input_details[0]["dtype"]
    )
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    np.testing.assert_allclose(output_data, output_data)
