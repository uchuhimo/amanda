import amanda
import torch
import torchvision

from examples.profile.pytorch.profiler import Profiler


def main():

    device = "cuda"

    model = torchvision.models.resnet50().to(device)
    x = torch.rand((32, 3, 227, 227)).to(device)

    profiler = Profiler()

    with amanda.tool.apply(profiler):
        y = model(x)
        y.backward(torch.rand_like(y))

    profiler.export_chrome_trace("tmp/profile/pytorch_resnet50.json")


if __name__ == "__main__":
    main()
