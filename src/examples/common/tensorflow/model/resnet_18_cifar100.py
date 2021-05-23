from examples.common.tensorflow.model.resnet_cifar import CifarModel


class ResNet18Cifar100(CifarModel):
    def __init__(self):
        super().__init__(resnet_size=18, num_classes=100, data_format="channels_first")
