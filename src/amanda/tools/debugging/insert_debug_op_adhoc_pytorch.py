from pathlib import Path

import numpy as np
import torch.jit
import torch.utils.cpp_extension
import torchvision.models as models

import amanda
from amanda.tests.utils import root_dir
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

torch.utils.cpp_extension.load_inline(
    name="store_tensor_to_file",
    cpp_sources=op_source,
    is_python_module=False,
    verbose=True,
)
print(torch.ops.amanda.store_tensor_to_file)

arch_name = "vgg11"
store_dir = root_dir() / "tmp" / "debug_info_pytorch" / arch_name

if not Path(store_dir).exists():
    Path(store_dir).mkdir(mode=0o755, parents=True, exist_ok=True)

input = torch.randn(1, 3, 224, 224)
model = models.vgg11(pretrained=False, progress=False)
model.eval()
traced_model = torch.jit.trace(model, (input,))
global new_model


def run_original_model():
    return traced_model(input)


def run_modified_model():
    return new_model(input)


def verify_output(output, new_output):
    np.testing.assert_allclose(output.detach().numpy(), new_output.detach().numpy())


def main():
    global new_model

    output = run_original_model()

    graph = amanda.pytorch.import_from_module(traced_model)
    new_graph = modify_graph(graph)
    new_model = amanda.pytorch.export_to_module(new_graph)

    new_output = run_modified_model()
    verify_output(output, new_output)


if __name__ == "__main__":
    main()
