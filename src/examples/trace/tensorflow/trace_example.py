import amanda
import tensorflow as tf
import tensorflow_hub as hub  # pip install --upgrade tensorflow-hub
from amanda.cache import cache_disabled
from tensorflow.keras.models import Model

from examples.common.tensorflow.model.resnet_50 import ResNet50

# from examples.trace.tensorflow.trace_tool import TraceTool
from examples.pruning.tensorflow.pruning_tool import PruningTool as TraceTool
from examples.utils.timer import Timer


def main_origin(model, x):
    # tool = TraceTool(output_dir="tmp/trace_resnet50_tf/tracetool.txt")
    tool = None

    with amanda.cache.cache_disabled():
        y = model(x)
        z = y + 1
        with tf.Session() as session:
            with amanda.disabled():
                session.run(tf.initialize_all_variables())
            g = tf.gradients(y, x)

            session.run(g)

            with Timer(verbose=True) as t:
                session.run(g)

    return t.elapsed


def main_core(model, x):
    # tool = TraceTool(output_dir="tmp/trace_resnet50_tf/tracetool.txt")
    tool = None

    with amanda.tool.apply(tool):
        y = model(x)
        z = y + 1
        with tf.Session() as session:
            with amanda.disabled():
                session.run(tf.initialize_all_variables())
            g = tf.gradients(y, x)

            session.run(g)

            with Timer(verbose=True) as t:
                session.run(g)

    return t.elapsed


def main_usecase(model, x):
    # tool = TraceTool(output_dir="tmp/trace_resnet50_tf/tracetool.txt")
    tool = TraceTool()
    # tool = None

    with amanda.tool.apply(tool):
        y = model(x)
        z = y + 1
        with tf.Session() as session:
            with amanda.disabled():
                session.run(tf.initialize_all_variables())
            g = tf.gradients(y, x)

            session.run(g)

            with Timer(verbose=True) as t:
                session.run(g)
    return t.elapsed


def main_core_nocache(model, x):
    # tool = TraceTool(output_dir="tmp/trace_resnet50_tf/tracetool.txt")
    tool = None

    with amanda.tool.apply(tool), amanda.cache.cache_disabled():
        y = model(x)
        z = y + 1
        with tf.Session() as session:
            with amanda.disabled():
                session.run(tf.initialize_all_variables())
            g = tf.gradients(y, x)

            session.run(g)

            with Timer(verbose=True) as t:
                session.run(g)
    return t.elapsed


def main_usecase_nocache(model, x):
    tool = TraceTool()
    # tool = TraceTool(output_dir="tmp/trace_resnet50_tf/tracetool.txt")
    # tool = None

    with amanda.tool.apply(tool), amanda.cache.cache_disabled():
        y = model(x)
        z = y + 1
        with tf.Session() as session:
            with amanda.disabled():
                session.run(tf.initialize_all_variables())
            g = tf.gradients(y, x)

            session.run(g)

            with Timer(verbose=True) as t:
                session.run(g)
    return t.elapsed


def main_bert():
    max_seq_length = 128
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids"
    )
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_mask"
    )
    segment_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="segment_ids"
    )
    bert_layer = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True
    )
    pooled_output, sequence_output = bert_layer(
        [input_word_ids, input_mask, segment_ids]
    )
    model = Model(
        inputs=[input_word_ids, input_mask, segment_ids],
        outputs=[pooled_output, sequence_output],
    )

    input_ids = [0] * max_seq_length
    input_masks = [0] * max_seq_length
    input_segments = [0] * max_seq_length

    tool = TraceTool(output_dir="tmp/trace_resnet50_tf/tracetool.txt")
    # tool = None

    # with amanda.tool.apply(tool), amanda.cache.cache_disabled():
    with amanda.tool.apply(tool):

        pool_embs, all_embs = model.predict(
            [[input_ids], [input_masks], [input_segments]]
        )

        with Timer(verbose=True) as t:
            pool_embs, all_embs = model.predict(
                [[input_ids], [input_masks], [input_segments]]
            )


def main(model, x):
    # tool = TraceTool(output_dir="tmp/trace_resnet50_tf/tracetool.txt")
    tool = TraceTool()
    # tool = None

    with amanda.tool.apply(tool):
        # with amanda.disabled():
        # with amanda.tool.apply(tool), amanda.cache.cache_disabled(), amanda.disabled():
        y = model(x)
        z = y + 1
        with tf.Session() as session:
            with amanda.disabled():
                session.run(tf.initialize_all_variables())
            g = tf.gradients(y, x)

            session.run(g)

            with Timer(verbose=True) as t:
                session.run(g)
    return t.elapsed


if __name__ == "__main__":
    # main()

    batch_size = 128

    # model = ResNet50()
    # x = tf.random.uniform(shape=[batch_size, 224, 224, 3])

    from examples.common.tensorflow.model.alexnet import AlexNet

    # main(model,x)
    # tf.train.export_meta_graph("downloads/model/alexnet/model.meta")
    model = AlexNet()
    # x = tf.random.uniform(shape=[batch_size, 224, 224, 3])
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 224, 224, 3])
    y = model(x)
    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        saver.save(session, "downloads/model/alexnet/alexnet")

    # origin_time = main_origin(model, x)
    # usecase_time = main_usecase(model, x)
    # core_time = main_core(model, x)
    # usecase_time_nocache = main_usecase_nocache(model,x)
    # core_time_nocache = main_core_nocache(model, x)

    # print(f"origin time {origin_time}")
    # print(f"core time {(core_time-origin_time)/origin_time}")
    # print(f"usecase time {(usecase_time-core_time)/origin_time}")
    # print(f"core no cache time {(core_time_nocache-origin_time)/origin_time}")
    # print(f"usecase no cache time {(usecase_time_nocache-core_time_nocache)/origin_time}")
