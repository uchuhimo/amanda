import tensorflow as tf

import amanda
from examples.common.tensorflow.model.resnet_50 import ResNet50
from examples.trace.tensorflow.trace_tool import TraceTool


def main():
    model = ResNet50()
    x = tf.random.uniform(shape=[8, 224, 224, 3])

    tool = TraceTool(output_dir="tmp/trace_resnet50_tf/tracetool.txt")

    with amanda.tool.apply(tool):
        y = model(x)
        z = y + 1
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())
            print(session.run(y).shape)
            session.run(tf.gradients(y, x))
            session.run(tf.gradients(z, x))


if __name__ == "__main__":
    main()
