import amanda
import tensorflow as tf

from examples.flops_profiler.mapping.tool import FlopsProfileTool


class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(tf.random.normal([in_features, out_features]), name="w")
        self.b = tf.Variable(tf.zeros([out_features]), name="b")

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


class SequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


def main():
    model = SequentialModule(name="example_model")
    x = tf.random.uniform(shape=[1, 3])

    profiler = FlopsProfileTool("tensorflow")

    with amanda.tool.apply(profiler):
        y = model(x)
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())
            session.run(y)
            session.run(tf.gradients(y, x))


if __name__ == "__main__":
    main()
