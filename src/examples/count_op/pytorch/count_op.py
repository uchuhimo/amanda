import torch
from torchvision.models import resnet50

import amanda


class CountConvTool(amanda.Tool):
    def __init__(self):
        self.counter = 0
        self.add_inst_for_op(self.callback)

    def callback(self, context: amanda.OpContext):
        op = context.get_op()
        if op.__name__ == "conv2d":
            context.insert_before_op(self.count)

    def count(self, *inputs):
        self.counter += 1
        return inputs


def main():
    tool = CountConvTool()
    with amanda.apply(tool):
        model = resnet50()
        x = torch.rand((2, 3, 227, 227))
        model(x)
        print(tool.counter)


if __name__ == "__main__":
    main()
