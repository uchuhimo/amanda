from examples.common.tensorflow.model.resnet_imagenet import ImagenetModel


class ResNet50(ImagenetModel):
    def __init__(self):
        super().__init__(resnet_size=50, data_format="channels_first")
