import os
import sys

import amanda
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from loguru import logger

from examples.pruning.pytorch.pruning_tool import PruneTool
from examples.utils.timer import Timer


class CountConvTool(amanda.Tool):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.add_inst_for_op(self.callback)
        self.add_inst_for_op(self.callback_bw, backward=True)

    def callback(self, context: amanda.OpContext):
        op = context.get_op()
        if op.__name__ == "conv2d":
            context.insert_before_op(self.count)
            context.insert_after_op(self.count)

    def callback_bw(self, context: amanda.OpContext):
        op = context.get_op()
        backward_op = context.get_backward_op()
        if (
            op.__name__ == "conv2d"
            and backward_op.__name__ == "CudnnConvolutionBackward"
        ):
            context.insert_before_backward_op(self.count)
            context.insert_after_backward_op(self.count)

    def count(self, *tensors):
        self.counter += 1
        return tensors


def main():

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    torch.manual_seed(42)

    save_path = "tmp/model/resnet50_cifar100"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    batch_size = 128
    learning_rate = 0.1
    num_epochs = 350

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    model = torchvision.models.resnet50(num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    # For updating learning rate

    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    # Train the model
    total_step = len(train_loader)

    tool = PruneTool()

    with amanda.tool.apply(tool):
        for epoch in range(num_epochs):

            for i, (images, labels) in enumerate(train_loader):

                model.train()
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                if (i + 1) % 100 == 0:
                    print(
                        "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
                            epoch + 1, num_epochs, i + 1, total_step, loss.item()
                        )
                    )

            if epoch == 150:
                update_lr(optimizer, 0.01)
            elif epoch == 250:
                update_lr(optimizer, 0.001)

            # Test the model
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print(
                    "Accuracy of the model on the test images: {} %".format(
                        100 * correct / total
                    )
                )

            # Save the model checkpoint
            torch.save(model.state_dict(), f"{save_path}/epoch_{epoch}.ckpt")


if __name__ == "__main__":
    main()
