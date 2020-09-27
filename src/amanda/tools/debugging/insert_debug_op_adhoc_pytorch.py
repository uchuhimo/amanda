from pathlib import Path

import numpy as np
import torch.jit
import torch.utils.cpp_extension
import torchvision.models as models
from loguru import logger

import amanda
from amanda.io.file import root_dir
from amanda.tools.debugging.insert_debug_op_adhoc import modify_graph

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


def init(input):
    model = models.vgg11(pretrained=False, progress=False)
    model.eval()
    torch.utils.cpp_extension.load_inline(
        name="store_tensor_to_file",
        cpp_sources=op_source,
        is_python_module=False,
        verbose=True,
    )
    logger.info(f"load {torch.ops.amanda.store_tensor_to_file} successfully")
    return torch.jit.trace(model, (input,))


def verify_output(output, new_output):
    np.testing.assert_allclose(output.detach().numpy(), new_output.detach().numpy())


def main():
    input = torch.randn(1, 3, 224, 224)
    traced_model = init(input)
    output = traced_model(input)
    graph = amanda.pytorch.import_from_module(traced_model)
    new_graph = modify_graph(graph)
    new_model = amanda.pytorch.export_to_module(new_graph)
    new_output = new_model(input)
    verify_output(output, new_output)


if __name__ == "__main__":
    main()
