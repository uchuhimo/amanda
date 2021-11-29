import amanda
import torch
import torchvision

from examples.flops_profiler.mapping.tool import FlopsProfileTool


def main():

    device = "cuda"

    model = torchvision.models.resnet50().to(device)
    x = torch.rand((32, 3, 227, 227)).to(device)

    profiler = FlopsProfileTool()

    with amanda.tool.apply(profiler):
        y = model(x)
        y = torch.split(y, 2, dim=0)
        y = y[0] + y[1]
        y.backward(torch.rand_like(y))


if __name__ == "__main__":
    main()
