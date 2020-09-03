from pathlib import Path

import numpy as np
import torch.jit
import torch.utils.cpp_extension
import torchvision.models as models
from loguru import logger

import amanda
from amanda.tests.utils import root_dir

op_source = """
#include <torch/script.h>
#include <iostream>
#include <memory>

torch::Tensor store_tensor_to_file(
  torch::Tensor input
) {
  std::string store_dir = "/tmp";
  std::string file_name = "tensor_data";
  auto bytes = torch::jit::pickle_save(input);
  auto filename = store_dir + "/" + file_name;
  std::ofstream fout(filename, std::ios::out | std::ios::binary);
  fout.write(bytes.data(), bytes.size());
  fout.close();
  return input;
}

static auto registry = torch::RegisterOperators(
  "amanda::store_tensor_to_file(Tensor input) -> Tensor",
  &store_tensor_to_file,
  torch::RegisterOperators::options().aliasAnalysis(
    torch::AliasAnalysisKind::FROM_SCHEMA)
  );
"""

arch_name = "vgg11"
store_dir = root_dir() / "tmp" / "debug_info_pytorch" / arch_name

if not Path(store_dir).exists():
    Path(store_dir).mkdir(mode=0o755, parents=True, exist_ok=True)

input = torch.randn(1, 3, 224, 224)
model = models.vgg11(pretrained=False, progress=False)
model.eval()
global traced_model
global new_model


def init():
    torch.utils.cpp_extension.load_inline(
        name="store_tensor_to_file",
        cpp_sources=op_source,
        is_python_module=False,
        verbose=True,
    )
    logger.info(f"load {torch.ops.amanda.store_tensor_to_file} successfully")
    global traced_model
    traced_model = torch.jit.trace(model, (input,))


def run_original_model():
    return traced_model(input)


def run_modified_model():
    return new_model(input)


def verify_output(output, new_output):
    np.testing.assert_allclose(output.detach().numpy(), new_output.detach().numpy())


def modify_graph(graph: amanda.Graph):
    for op in graph.ops:
        for tensor in op.output_tensors:
            if tensor.attrs["type"].kind() == "TensorType":
                debug_op = amanda.create_op(
                    attrs={"type": "amanda::store_tensor_to_file"},
                    input_tensors=[tensor],
                    control_dependencies=[],
                    output_num=1,
                )
                debug_op.output_tensors[0].attrs["type"] = tensor.attrs["type"]

                for output_op in graph.ops:
                    for index, input_tensor in enumerate(output_op.input_tensors):
                        if tensor == input_tensor:
                            output_op.update_input_tensor(
                                index, debug_op.output_tensors[0]
                            )
                graph.add_op(debug_op)


def main():
    global new_model

    init()

    output = run_original_model()

    graph = amanda.pytorch.import_from_module(traced_model)
    modify_graph(graph)
    new_model = amanda.pytorch.export_to_module(graph)

    new_output = run_modified_model()
    verify_output(output, new_output)


if __name__ == "__main__":
    main()
