import torch
import torchvision

import amanda
from examples.trace.pytorch.trace_tool import TraceTool


def main():
    model = torchvision.models.resnet50()
    x = torch.rand((2, 3, 227, 227))

    tool = TraceTool()
    with amanda.tool.apply(tool):

        y = model(x)
        y.backward(torch.rand_like(y))


if __name__ == "__main__":
    main()
