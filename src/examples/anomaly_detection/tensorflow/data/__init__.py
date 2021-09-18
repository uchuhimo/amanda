import examples.anomaly_detection.tensorflow.data.cifar10 as cifar10
import examples.anomaly_detection.tensorflow.data.imdb as imdb
import examples.anomaly_detection.tensorflow.data.mnist as mnist
import examples.anomaly_detection.tensorflow.data.reuters as reuters
import examples.anomaly_detection.tensorflow.data.simplednn as simplednn

datasets = {
    "cifar10": {
        "load_data": cifar10.load_data,
        "preprocess": cifar10.preprocess,
    },
    "imdb": {
        "load_data": imdb.load_data,
        "preprocess": imdb.preprocess,
    },
    "mnist": {
        "load_data": mnist.load_data,
        "preprocess": mnist.preprocess,
    },
    "reuters": {
        "load_data": reuters.load_data,
        "preprocess": reuters.preprocess,
    },
    "simplednn": {
        "load_data": simplednn.load_data,
        "preprocess": simplednn.preprocess,
    },
}
