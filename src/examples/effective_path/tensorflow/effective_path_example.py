import amanda
import tensorflow as tf

from examples.common.tensorflow.model.resnet_18_cifar100 import ResNet18Cifar100
from examples.effective_path.tensorflow.effective_path_tool import EffectivePathTool


def main():
    batch_size = 4
    model = ResNet18Cifar100()
    x = tf.random.uniform(shape=[batch_size, 32, 32, 3])

    tool = EffectivePathTool()

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    with amanda.tool.apply(tool):
        y = model(x)
        with tf.Session(config=session_config) as session:
            session.run(tf.initialize_all_variables())
            session.run(y)

    tool.extract_path(entry_points=[y.op.name], batch=batch_size)
    density = tool.calc_density_per_layer()
    print(density)


if __name__ == "__main__":
    main()
