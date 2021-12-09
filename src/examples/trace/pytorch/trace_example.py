import amanda
import torch
import torchvision
from amanda.cache import cache_disabled

from examples.trace.pytorch.trace_tool import TraceTool
from examples.utils.timer import Timer


def main():
    device = "cuda"

    model = torchvision.models.resnet50().to(device)
    x = torch.rand((2, 3, 227, 227)).to(device)

    tool = TraceTool(output_dir="tmp/trace_resnet50/tracetool.txt")
    # tool = None

    with amanda.tool.apply(tool), amanda.cache.cache_disabled():
        for i in range(5):
            y = model(x)
            y.backward(torch.rand_like(y))

        with Timer(verbose=True) as t:

            y = model(x)
            y.backward(torch.rand_like(y))


if __name__ == "__main__":
    main()
