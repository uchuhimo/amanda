import tensorflow as tf

import amanda
from examples.common.tensorflow.model.resnet_50 import ResNet50
from examples.profile.tensorflow.profiler import Profiler


def main():
    model = ResNet50()
    x = tf.random.uniform(shape=[1, 224, 224, 3])

    profiler = Profiler()

    with amanda.tool.apply(profiler):
        y = model(x)
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())
            session.run(y)
            session.run(tf.gradients(y, x))

    profiler.export_chrome_trace("tmp/profile/tf_resnet50.json")


if __name__ == "__main__":
    main()
