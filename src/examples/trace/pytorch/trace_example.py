import amanda
import torch
import torchvision

from examples.trace.pytorch.trace_tool import TraceTool


def main():

    device = "cuda"

    model = torchvision.models.resnet50().to(device)
    x = torch.rand((2, 3, 227, 227)).to(device)

    tool = TraceTool(output_dir="tmp/trace_resnet50/tracetool.txt")

    with amanda.tool.apply(tool):

        y = model(x)
        y.backward(torch.rand_like(y))


if __name__ == "__main__":
    main()
