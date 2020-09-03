import tensorflow as tf
from tensorflow.contrib.slim.nets import inception, resnet_v1, resnet_v2, vgg

from amanda.tests.tensorflow.models import (
    inception_resnet_v1,
    inception_resnet_v2,
    mobilenet_v1,
    nasnet,
    test_rnn,
)
from amanda.tests.tensorflow.models.mobilenet import mobilenet_v2

slim = tf.contrib.slim

architecture_map = {
    "vgg16": {
        "url": "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
        "filename": "vgg_16.ckpt",
        "builder": lambda: vgg.vgg_16,
        "arg_scope": vgg.vgg_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 224, 224, 3]
        ),
        "num_classes": 1000,
    },
    "vgg19": {
        "url": "http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz",
        "filename": "vgg_19.ckpt",
        "builder": lambda: vgg.vgg_19,
        "arg_scope": vgg.vgg_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 224, 224, 3]
        ),
        "num_classes": 1000,
    },
    "inception_v1": {
        "url": "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz",
        "filename": "inception_v1.ckpt",
        "builder": lambda: inception.inception_v1,
        "arg_scope": inception.inception_v3_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 224, 224, 3]
        ),
        "num_classes": 1001,
    },
    "inception_v1_frozen": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/"
        "inception_v1_2016_08_28_frozen.pb.tar.gz",
        "filename": "inception_v1_2016_08_28_frozen.pb",
        "tensor_out": ["InceptionV1/Logits/Predictions/Reshape_1:0"],
        "tensor_in": ["input:0"],
        "input_shape": [[224, 224, 3]],  # input_shape of the elem in tensor_in
        "feed_dict": lambda img: {"input:0": img},
        "num_classes": 1001,
    },
    "inception_v3": {
        "url": "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz",
        "filename": "inception_v3.ckpt",
        "builder": lambda: inception.inception_v3,
        "arg_scope": inception.inception_v3_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 299, 299, 3]
        ),
        "num_classes": 1001,
    },
    "inception_v3_frozen": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/"
        "inception_v3_2016_08_28_frozen.pb.tar.gz",
        "filename": "inception_v3_2016_08_28_frozen.pb",
        "tensor_out": ["InceptionV3/Predictions/Softmax:0"],
        "tensor_in": ["input:0"],
        "input_shape": [[299, 299, 3]],  # input_shape of the elem in tensor_in
        "feed_dict": lambda img: {"input:0": img},
        "num_classes": 1001,
    },
    "resnet_v1_50": {
        "url": "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz",
        "filename": "resnet_v1_50.ckpt",
        "builder": lambda: resnet_v1.resnet_v1_50,
        "arg_scope": resnet_v2.resnet_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 224, 224, 3]
        ),
        "num_classes": 1000,
    },
    "resnet_v1_152": {
        "url": "http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz",
        "filename": "resnet_v1_152.ckpt",
        "builder": lambda: resnet_v1.resnet_v1_152,
        "arg_scope": resnet_v2.resnet_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 224, 224, 3]
        ),
        "num_classes": 1000,
    },
    "resnet_v2_50": {
        "url": "http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz",
        "filename": "resnet_v2_50.ckpt",
        "builder": lambda: resnet_v2.resnet_v2_50,
        "arg_scope": resnet_v2.resnet_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 299, 299, 3]
        ),
        "num_classes": 1001,
    },
    "resnet_v2_101": {
        "url": "http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz",
        "filename": "resnet_v2_101.ckpt",
        "builder": lambda: resnet_v2.resnet_v2_101,
        "arg_scope": resnet_v2.resnet_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 299, 299, 3]
        ),
        "num_classes": 1001,
    },
    "resnet_v2_152": {
        "url": "http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz",
        "filename": "resnet_v2_152.ckpt",
        "builder": lambda: resnet_v2.resnet_v2_152,
        "arg_scope": resnet_v2.resnet_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 299, 299, 3]
        ),
        "num_classes": 1001,
    },
    "resnet_v2_200": {
        "url": "http://download.tensorflow.org/models/resnet_v2_200_2017_04_14.tar.gz",
        "filename": "resnet_v2_200.ckpt",
        "builder": lambda: resnet_v2.resnet_v2_200,
        "arg_scope": resnet_v2.resnet_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 299, 299, 3]
        ),
        "num_classes": 1001,
    },
    "mobilenet_v1_1.0": {
        "url": "http://download.tensorflow.org/models/"
        "mobilenet_v1_1.0_224_2017_06_14.tar.gz",
        "filename": "mobilenet_v1_1.0_224.ckpt",
        "builder": lambda: mobilenet_v1.mobilenet_v1,
        "arg_scope": mobilenet_v1.mobilenet_v1_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 224, 224, 3]
        ),
        "num_classes": 1001,
    },
    "mobilenet_v1_1.0_frozen": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/"
        "mobilenet_v1_1.0_224_frozen.tgz",
        "filename": "mobilenet_v1_1.0_224/frozen_graph.pb",
        "tensor_out": ["MobilenetV1/Predictions/Softmax:0"],
        "tensor_in": ["input:0"],
        "input_shape": [[224, 224, 3]],  # input_shape of the elem in tensor_in
        "feed_dict": lambda img: {"input:0": img},
        "num_classes": 1001,
    },
    "mobilenet_v2_1.0_224": {
        "url": "https://storage.googleapis.com/mobilenet_v2/checkpoints/"
        "mobilenet_v2_1.0_224.tgz",
        "filename": "mobilenet_v2_1.0_224.ckpt",
        "builder": lambda: mobilenet_v2.mobilenet,
        "arg_scope": mobilenet_v2.training_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 224, 224, 3]
        ),
        "num_classes": 1001,
    },
    "inception_resnet_v2": {
        "url": "http://download.tensorflow.org/models/"
        "inception_resnet_v2_2016_08_30.tar.gz",
        "filename": "inception_resnet_v2_2016_08_30.ckpt",
        "builder": lambda: inception_resnet_v2.inception_resnet_v2,
        "arg_scope": inception_resnet_v2.inception_resnet_v2_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 299, 299, 3]
        ),
        "num_classes": 1001,
    },
    "nasnet-a_large": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/"
        "nasnet-a_large_04_10_2017.tar.gz",
        "filename": "model.ckpt",
        "builder": lambda: nasnet.build_nasnet_large,
        "arg_scope": nasnet.nasnet_large_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 331, 331, 3]
        ),
        "num_classes": 1001,
    },
    "facenet": {
        "url": "http://mmdnn.eastasia.cloudapp.azure.com:89/models/tensorflow/facenet/"
        "20180408-102900.zip",
        "filename": "20180408-102900/model-20180408-102900.ckpt-90",
        "builder": lambda: inception_resnet_v1.inception_resnet_v1,
        "arg_scope": inception_resnet_v1.inception_resnet_v1_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.float32, shape=[None, 160, 160, 3]
        ),
        "feed_dict": lambda img: {"input:0": img, "phase_train:0": False},
        "num_classes": 0,
    },
    "facenet_frozen": {
        "url": "http://mmdnn.eastasia.cloudapp.azure.com:89/models/tensorflow/facenet/"
        "20180408-102900.zip",
        "filename": "20180408-102900/20180408-102900.pb",
        "tensor_out": ["InceptionResnetV1/Logits/AvgPool_1a_8x8/AvgPool:0"],
        "tensor_in": ["input:0", "phase_train:0"],
        "input_shape": [[160, 160, 3], 1],  # input_shape of the elem in tensor_in
        "feed_dict": lambda img: {"input:0": img, "phase_train:0": False},
        "num_classes": 0,
    },
    "rnn_lstm_gru_stacked": {
        "url": "http://mmdnn.eastasia.cloudapp.azure.com:89/models/tensorflow/tf_rnn/"
        "tf_rnn.zip",
        # Note this is just a model used for test, not a standard rnn model.
        "filename": "tf_rnn/tf_lstm_gru_stacked.ckpt",
        "builder": lambda: test_rnn.create_symbol,
        "arg_scope": test_rnn.dummy_arg_scope,
        "input": lambda: tf.placeholder(
            name="input", dtype=tf.int32, shape=[None, 150]
        ),
        "feed_dict": lambda x: {"input:0": x},
        "num_classes": 0,
    },
}


def handle_checkpoint(architecture, path):
    with slim.arg_scope(architecture_map[architecture]["arg_scope"]()):
        data_input = architecture_map[architecture]["input"]()
        logits, endpoints = architecture_map[architecture]["builder"]()(
            data_input,
            num_classes=architecture_map[architecture]["num_classes"],
            is_training=False,
        )

        if logits.op.type == "Squeeze":
            tf.identity(logits, name="Output")
        else:
            tf.squeeze(logits, name="Output")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, path + architecture_map[architecture]["filename"])
        save_path = saver.save(sess, path + "imagenet_{}.ckpt".format(architecture))
        print("Model saved in file: %s" % save_path)

    import tensorflow.contrib.keras as keras

    keras.backend.clear_session()
