import amanda
import tensorflow as tf

from examples.common.tensorflow.model.resnet_50 import ResNet50
from examples.flops_profiler.mapping.tool import FlopsProfileTool


def main():
    model = ResNet50()
    x = tf.random.uniform(shape=[8, 224, 224, 3])

    profiler = FlopsProfileTool("tensorflow")

    with amanda.tool.apply(profiler):
        y = model(x)
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())
            session.run(y)
            session.run(tf.gradients(y, x))


if __name__ == "__main__":
    main()
