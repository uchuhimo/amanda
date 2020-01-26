from pathlib import Path

import numpy as np
import torch
import torch.jit
import torch.utils.cpp_extension
import torchvision.models as models
from torch._C import StringType

from amanda import Op
from amanda.conversion.pytorch import export_to_module, import_from_module
from amanda.tests.utils import root_dir

op_source = """
#include <torch/script.h>
#include <iostream>
#include <memory>

torch::Tensor store_tensor_to_file(
  torch::Tensor input,
  std::string store_dir,
  std::string file_name
) {
  auto bytes = torch::jit::pickle_save(input);
  auto filename = store_dir + "/" + file_name;
  std::ofstream fout(filename, std::ios::out | std::ios::binary);
  fout.write(bytes.data(), bytes.size());
  fout.close();
  return input;
}

static auto registry = torch::RegisterOperators(
  "amanda::store_tensor_to_file(Tensor input, str store_dir, str file_name) -> Tensor",
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


def modify_graph(graph):
    for op in graph.ops:
        for tensor in op.output_tensors:
            if tensor.attrs["type"].kind() == "TensorType":
                store_dir_op = Op(
                    attrs={"type": "prim::Constant", "value": str(store_dir)}
                )
                store_dir_op.output_tensors[0].attrs["type"] = StringType.get()
                file_name_op = Op(
                    attrs={
                        "type": "prim::Constant",
                        "value": op.attrs["/amanda/name"].replace(":", "_"),
                    }
                )
                file_name_op.output_tensors[0].attrs["type"] = StringType.get()
                debug_op = Op(
                    attrs={"type": "amanda::store_tensor_to_file"},
                    input_tensors=[
                        tensor,
                        store_dir_op.output_tensors[0],
                        file_name_op.output_tensors[0],
                    ],
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
                graph.add_op(store_dir_op)
                graph.add_op(file_name_op)
                graph.add_op(debug_op)


def main():
    input = torch.randn(1, 3, 224, 224)
    model = models.vgg11(pretrained=False, progress=False)
    model.eval()
    traced_model = torch.jit.trace(model, (input,))
    output = traced_model(input)
    graph = import_from_module(traced_model)
    modify_graph(graph)
    new_model = export_to_module(graph)
    new_output = new_model(input)
    np.testing.assert_allclose(output.detach().numpy(), new_output.detach().numpy())


if __name__ == "__main__":
    main()
