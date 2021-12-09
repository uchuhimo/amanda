import os
import shutil
import time

import amanda
import tensorflow as tf
from amanda.io.file import abspath

from examples.common.tensorflow.dataset import alexnet_preprocess_image, imagenet
from examples.common.tensorflow.dataset.cifar100_main import input_fn
from examples.common.tensorflow.dataset.envs import CIFAR100_RAW_DIR, IMAGENET_DIR
from examples.common.tensorflow.utils import new_session_config
from examples.pruning.tensorflow import (
    cifar100_model_fn,
    resnet_50_model_fn,
    validate_batch_size_for_multi_gpu,
)
from examples.pruning.tensorflow.alexnet_imagenet_train import alexnet_model_fn
from examples.pruning.tensorflow.pruning_tool import PruningTool


def train_resnet_18_cifar100(
    batch_size: int = 128,
    train_epochs: int = 182,
    epochs_between_evals: int = 10,
    multi_gpu: bool = False,
    label: str = None,
    take_num: int = None,
):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
    tf.logging.set_verbosity(tf.logging.INFO)
    if label is None:
        model_dir = abspath("tmp/tf/resnet-18-cifar100/model_overhead/")
    else:
        model_dir = abspath(f"tmp/tf/resnet-18-cifar100/model_{label}/")
    data_dir = abspath(CIFAR100_RAW_DIR)

    model_function = cifar100_model_fn
    if multi_gpu:
        # There are two steps required if using multi-GPU: (1) wrap the model_fn,
        # and (2) wrap the optimizer. The first happens here, and (2) happens
        # in the model_fn itself when the optimizer is defined.
        model_function = tf.contrib.estimator.replicate_model_fn(
            model_function, loss_reduction=tf.losses.Reduction.MEAN
        )

    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_secs=60 * 60,
        keep_checkpoint_max=None,
        session_config=new_session_config(parallel=0),
    )
    classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=model_dir,
        config=estimator_config,
        params={"batch_size": batch_size, "multi_gpu": multi_gpu, "loss_scale": 1},
    )

    # Train the model
    def train_input_fn():
        input = input_fn(
            is_training=True,
            data_dir=data_dir,
            batch_size=batch_size,
            num_epochs=epochs_between_evals,
        )
        if take_num is not None:
            input = input.take(take_num)
        return input

    def train_input_fn_warmup():
        input = input_fn(
            is_training=True,
            data_dir=data_dir,
            batch_size=batch_size,
            num_epochs=epochs_between_evals,
        )
        return input.take(1)

    # Set up training hook that logs the training accuracy every 100 steps.
    tensors_to_log = {"train_accuracy": "train_accuracy"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    classifier.train(input_fn=train_input_fn_warmup, hooks=[logging_hook])
    start_time = time.time()
    classifier.train(input_fn=train_input_fn, hooks=[logging_hook])
    end_time = time.time()
    return end_time - start_time


def train_alexnet_imagenet(
    batch_size: int = 64,
    train_epochs: int = 10,
    epochs_between_evals: int = 1,
    multi_gpu: bool = False,
    label: str = None,
    take_num: int = None,
):
    tf.logging.set_verbosity(tf.logging.INFO)
    if label is None:
        model_dir = abspath("tmp/tf/alexnet/model_overhead/")
    else:
        model_dir = abspath(f"tmp/tf/alexnet/model_{label}/")
    data_dir = abspath(IMAGENET_DIR)

    model_function = alexnet_model_fn
    if multi_gpu:
        validate_batch_size_for_multi_gpu(batch_size)

        # There are two steps required if using multi-GPU: (1) wrap the model_fn,
        # and (2) wrap the optimizer. The first happens here, and (2) happens
        # in the model_fn itself when the optimizer is defined.
        model_function = tf.contrib.estimator.replicate_model_fn(
            model_function, loss_reduction=tf.losses.Reduction.MEAN
        )

    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_secs=60 * 60,
        keep_checkpoint_max=None,
        session_config=new_session_config(parallel=0),
    )
    classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=model_dir,
        config=estimator_config,
        params={"multi_gpu": multi_gpu, "batch_size": batch_size},
    )

    # Evaluate the model and print results
    def train_input_fn():
        input = imagenet.train(
            data_dir,
            batch_size,
            num_epochs=epochs_between_evals,
            num_parallel_calls=40,
            is_shuffle=True,
            multi_gpu=multi_gpu,
            preprocessing_fn=alexnet_preprocess_image,
            transform_fn=lambda ds: ds.map(lambda image, label: (image, label - 1)),
        )
        if take_num is not None:
            input = input.take(take_num)
        return input

    tensors_to_log = {"train_accuracy": "train_accuracy"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    start_time = time.time()
    classifier.train(input_fn=train_input_fn, hooks=[logging_hook])
    end_time = time.time()
    return end_time - start_time


class TrackerHook(tf.train.SessionRunHook):
    def __init__(self, tr):
        self.tr = tr

    def after_run(self, run_context, run_values):
        self.tr.print_diff()


def resnet_50_train(
    batch_size: int = 64,
    train_epochs: int = 10,
    epochs_between_evals: int = 1,
    multi_gpu: bool = False,
    label: str = None,
    take_num: int = None,
):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
    tf.logging.set_verbosity(tf.logging.INFO)
    if label is None:
        model_dir = abspath("tmp/tf/resnet-50-v2/model_overhead/")
    else:
        model_dir = abspath(f"tmp/tf/resnet-50-v2/model_{label}/")
    data_dir = abspath(IMAGENET_DIR)

    model_function = resnet_50_model_fn
    if multi_gpu:
        validate_batch_size_for_multi_gpu(batch_size)

        # There are two steps required if using multi-GPU: (1) wrap the model_fn,
        # and (2) wrap the optimizer. The first happens here, and (2) happens
        # in the model_fn itself when the optimizer is defined.
        model_function = tf.contrib.estimator.replicate_model_fn(
            model_function, loss_reduction=tf.losses.Reduction.MEAN
        )

    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_secs=60 * 60,
        keep_checkpoint_max=None,
        session_config=new_session_config(parallel=0),
    )
    classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=model_dir,
        config=estimator_config,
        params={"batch_size": batch_size, "multi_gpu": multi_gpu},
    )

    # Train the model
    def train_input_fn():
        input = imagenet.train(
            data_dir,
            batch_size,
            num_epochs=epochs_between_evals,
            num_parallel_calls=40,
            is_shuffle=True,
            multi_gpu=multi_gpu,
        )
        if take_num is not None:
            input = input.take(take_num)
        return input

    # Set up training hook that logs the training accuracy every 100 steps.
    tensors_to_log = {"train_accuracy": "train_accuracy"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    # tr = tracker.SummaryTracker()
    # tr_hook = TrackerHook(tr)
    start_time = time.time()
    classifier.train(input_fn=train_input_fn, hooks=[logging_hook])
    # tr.print_diff()
    end_time = time.time()
    return end_time - start_time


def main_v1(train_fn, model_name):
    take_num = 1000
    epochs_between_evals = 10
    model_dir = abspath(f"tmp/tf/{model_name}/model_overhead/")
    train = train_fn

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    with amanda.disabled():
        train_time_warmup = train(
            take_num=1,
            epochs_between_evals=epochs_between_evals,
        )

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    tool = PruningTool(disabled=True)
    with amanda.tool.apply(tool):
        train_with_hook_time_warmup = train(
            take_num=1,
            epochs_between_evals=epochs_between_evals,
        )

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    tool = PruningTool(disabled=False)
    with amanda.tool.apply(tool):
        train_with_pruning_time_warmup = train(
            take_num=1,
            epochs_between_evals=epochs_between_evals,
        )

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    with amanda.disabled():
        train_time = train(
            take_num=take_num + 1,
            epochs_between_evals=epochs_between_evals,
        )

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    tool = PruningTool(disabled=True)
    with amanda.tool.apply(tool):
        train_with_hook_time = train(
            take_num=take_num + 1,
            epochs_between_evals=epochs_between_evals,
        )

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    tool = PruningTool(disabled=False)
    with amanda.tool.apply(tool):
        train_with_pruning_time = train(
            take_num=take_num + 1,
            epochs_between_evals=epochs_between_evals,
        )

    hook_overhead = (
        (train_with_hook_time - train_with_hook_time_warmup)
        / (train_time - train_time_warmup)
    ) - 1
    pruning_overhead = (
        (
            (train_with_pruning_time - train_with_pruning_time_warmup)
            / (train_time - train_time_warmup)
        )
        - hook_overhead
        - 1
    )
    hook_overhead_warmup = (train_with_hook_time_warmup / train_time_warmup) - 1
    pruning_overhead_warmup = (
        (train_with_pruning_time_warmup / train_time_warmup) - hook_overhead_warmup - 1
    )
    print(f"train: {(train_time - train_time_warmup) / take_num}s")
    time = (train_with_hook_time - train_with_hook_time_warmup) / take_num
    print(f"train_with_hook: {time}s")
    time = (train_with_pruning_time - train_with_pruning_time_warmup) / take_num
    print(f"train_with_pruning: {time}s")
    print(f"train/warmup: {train_time_warmup}s")
    print(f"train_with_hook/warmup: {train_with_hook_time_warmup}s")
    print(f"train_with_pruning/warmup: {train_with_pruning_time_warmup}s")
    print(f"hook_overhead: {hook_overhead}")
    print(f"pruning_overhead: {pruning_overhead}")
    print(f"hook_overhead/warmup: {hook_overhead_warmup}")
    print(f"pruning_overhead/warmup: {pruning_overhead_warmup}")


def main(train_fn, model_name, prune_matmul: bool = True):
    take_num = 100
    epochs_between_evals = 10
    model_dir = abspath(f"tmp/tf/{model_name}/model_overhead/")
    train = train_fn

    # warmup
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    with amanda.disabled():
        train(
            take_num=take_num,
            epochs_between_evals=epochs_between_evals,
        )

    # full
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    tool = PruningTool()
    with amanda.tool.apply(tool):
        train_with_pruning_time = train(
            take_num=take_num,
            epochs_between_evals=epochs_between_evals,
        )
        train_with_pruning_time /= take_num

    # origin
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    with amanda.disabled():
        train_time = train(
            take_num=take_num,
            epochs_between_evals=epochs_between_evals,
        )
        train_time /= take_num

    # core
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    tool = None
    with amanda.tool.apply(tool):
        train_with_hook_time = train(
            take_num=take_num,
            epochs_between_evals=epochs_between_evals,
        )
        train_with_hook_time /= take_num

    # if os.path.exists(model_dir):
    #     shutil.rmtree(model_dir, ignore_errors=True)
    # tool = PruningTool(disabled=False, prune_matmul=prune_matmul)
    # with amanda.tool.apply(tool):
    #     train_with_pruning_time_warmup = train(
    #         with_hook=True,
    #         take_num=1,
    #         epochs_between_evals=epochs_between_evals,
    #     )

    hook_overhead = (train_with_hook_time / train_time) - 1
    pruning_overhead = (train_with_pruning_time / train_time) - hook_overhead - 1
    # pruning_overhead_warmup = (
    #     (train_with_pruning_time_warmup / train_time) - hook_overhead - 1
    # )
    print(f"train: {train_time*1000}ms")
    print(f"train_with_hook: {train_with_hook_time*1000}ms")
    print(f"train_with_pruning: {train_with_pruning_time*1000}ms")
    # print(f"train_with_pruning/warmup: {train_with_pruning_time_warmup}s")
    print(f"hook_overhead: {hook_overhead}")
    print(f"pruning_overhead: {pruning_overhead}")
    # print(f"pruning_overhead/warmup: {pruning_overhead_warmup}")


if __name__ == "__main__":
    main(train_resnet_18_cifar100, "resnet-18-cifar100")
    # main(train_alexnet_imagenet, "alexnet")
    # main(resnet_50_train, "resnet-50-v2")
    # main(resnet_50_train, "resnet-50-v2", prune_matmul=False)
