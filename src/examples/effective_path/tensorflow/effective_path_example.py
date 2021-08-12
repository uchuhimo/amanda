import amanda
import tensorflow as tf

from examples.common.tensorflow.model.resnet_18_cifar100 import ResNet18Cifar100
from examples.effective_path.tensorflow.effective_path_tool import EffectivePathTool


def main():
    model = ResNet18Cifar100()
    x = tf.random.uniform(shape=[4, 32, 32, 3])

    tool = EffectivePathTool()

    with amanda.tool.apply(tool):
        y = model(x)
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())
            session.run(y)

    tool.extract_path(entry_points=[y.op.name], batch=4)
    density = tool.calc_density_per_layer()
    print(density)


if __name__ == "__main__":
    main()
