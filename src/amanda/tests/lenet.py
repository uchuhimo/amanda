import tensorflow as tf

import amanda


class LeNet:
    def __init__(self, data_format: str = "channels_first"):
        # if data_format == "channels_first":
        #     self._input_shape = [-1, 1, 28, 28]
        # else:
        #     assert data_format == "channels_last"
        #     self._input_shape = [-1, 28, 28, 1]

        # self.conv1 = tf.layers.Conv2D(
        #     6, 5, data_format=data_format, activation=tf.nn.relu
        # )
        # self.conv2 = tf.layers.Conv2D(
        #     16, 5, data_format=data_format, activation=tf.nn.relu
        # )
        # self.fc1 = tf.layers.Dense(120, activation=tf.nn.relu)
        # self.fc2 = tf.layers.Dense(84, activation=tf.nn.relu)
        # self.fc3 = tf.layers.Dense(10)
        self.fc3 = tf.layers.Dense(10, activation=tf.nn.relu, use_bias=False)
        # self.max_pool2d = tf.layers.MaxPooling2D(
        #     (2, 2), (2, 2), padding="same", data_format=data_format
        # )
        # self.dropout = tf.layers.Dropout(rate=0.5)

    def __call__(self, inputs, training=False):
        """Add operations to classify a batch of input images.

        Args:
          inputs: A Tensor representing a batch of input images.
          training: A boolean. Set to True to add operations required only when
            training the classifier.

        Returns:
          A logits Tensor with shape [<batch_size>, 10].
        """
        # y = tf.reshape(inputs, self._input_shape)
        # y = self.conv1(y)

        # y = self.conv1(inputs)
        # y = self.max_pool2d(y)
        # y = self.conv2(y)
        # y = self.max_pool2d(y)
        # y = tf.layers.flatten(y)
        # y = self.fc1(y)
        # y = self.dropout(y, training=training)
        # y = self.fc2(y)
        # y = self.dropout(y, training=training)
        # return self.fc3(y)
        return self.fc3(inputs)


if __name__ == "__main__":
    with tf.Graph().as_default() as tf_graph:
        with tf.Session() as sess:
            # input = tf.placeholder(dtype=tf.float32, shape=(1, 1, 28, 28))
            input = tf.placeholder(dtype=tf.float32, shape=(1, 28 * 28))
            fc = tf.layers.Dense(units=10, activation=tf.nn.relu, use_bias=False)
            logits = fc(input)
            graph = amanda.tensorflow.import_from_graph(tf_graph, session=sess)
            graph.print()
