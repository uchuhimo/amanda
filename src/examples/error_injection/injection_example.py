from typing import Callable

import torch
import torchvision
from injection_tool import ErrorInjectionTool

import amanda


def gen_op_names_filter(op_names) -> Callable:
    def op_names_filter(op_name: str) -> bool:
        return op_name in op_names

    return op_names_filter


def zero_out(tensor):
    t = torch.zeros_like(tensor)
    t[0] = 1
    return t


def test_error_injection():

    device = "cuda"

    model = torchvision.models.inception_v3().to(device)
    input = torch.rand((2, 3, 299, 299)).to(device)

    injector = ErrorInjectionTool(
        filter_fn=gen_op_names_filter(["relu_"]), modify_fn=zero_out
    )

    with amanda.tool.apply(injector):

        output = model(input)

    print(output[0].shape)


if __name__ == "__main__":
    test_error_injection()
