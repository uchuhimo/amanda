#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

import os
import time
from typing import Set
import amanda
import shutil

import tensorflow as tf

from examples.common.tensorflow.dataset.cifar100_main import input_fn
from examples.common.tensorflow.utils import new_session_config
from examples.common.tensorflow.dataset.envs import CIFAR100_RAW_DIR
from amanda.io.file import abspath
# from examples.pruning.tensorflow.pruning import PruningTool
from examples.pruning.tensorflow.pruning_test import PruningTool
from examples.pruning.tensorflow.resnet_18_cifar100_train import cifar100_model_fn


def train(
    batch_size: int = 128,
    train_epochs: int = 182,
    epochs_between_evals: int = 1,
    multi_gpu: bool = False,
    label: str = None,
    with_hook: bool = False,
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
    if with_hook:
        amanda.tensorflow.inject_hook(classifier)

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

    # Set up training hook that logs the training accuracy every 100 steps.
    tensors_to_log = {"train_accuracy": "train_accuracy"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100
    )
    classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

    print(label)
    print()


class DummyTool(amanda.Tool):
    def __init__(self, debug: bool = False):
        super(DummyTool, self).__init__(namespace="amanda/tensorflow")

        def filter_fn(event, context):
            op = context["op"]
            if op.type not in [
                "Conv2D",
                "MatMul",
            ]:
                return False
            if event in [amanda.event.before_op_added, amanda.event.after_op_added]:
                if not debug and event == amanda.event.after_op_added:
                    return False
                else:
                    return True
            else:
                backward_op = context["backward_op"]
                if backward_op.type in [
                    "Conv2DBackpropFilter",
                    "MatMul",
                ]:
                    if not debug and event == amanda.event.before_backward_op_added:
                        return False
                    else:
                        return True
                else:
                    return False

        self.depends_on(
            amanda.tools.FilterOpTool(filter_fn=filter_fn),
            amanda.tools.EagerContextTool(),
        )
        self.register_event(
            amanda.event.before_op_executed,
            self.test_before_op_executed
        )
        self.register_event(
            amanda.event.after_op_executed,
            self.test_after_op_executed
        )
        self.register_event(
            amanda.event.before_backward_op_executed,
            self.test_before_backward_op_executed
        )
        self.register_event(
            amanda.event.after_backward_op_executed,
            self.test_after_backward_op_executed
        )
        self.debug = debug
        self.before_executed_ops: Set[str] = set()
        self.after_executed_ops: Set[str] = set()
        self.before_executed_backward_ops: Set[str] = set()
        self.after_executed_backward_ops: Set[str] = set()

    def test_before_op_executed(self, context: amanda.EventContext):
        if self.debug:
            op = context["op"]
            print("before", op.type, [input.dtype for input in context["inputs"]], op.name)
            self.before_executed_ops.add(op.name)
        return

    def test_after_op_executed(self, context: amanda.EventContext):
        if self.debug:
            op = context["op"]
            self.after_executed_ops.add(op.name)
        return

    def test_before_backward_op_executed(self, context: amanda.EventContext):
        if self.debug:
            op = context["op"]
            backward_op = context["backward_op"]
            print("before_backward", op.type, op.name, backward_op.type, backward_op.name)
            self.before_executed_backward_ops.add(backward_op.name)
        return

    def test_after_backward_op_executed(self, context: amanda.EventContext):
        if self.debug:
            backward_op = context["backward_op"]
            self.after_executed_backward_ops.add(backward_op.name)
        return


def main():
    take_num = None
    epochs_between_evals = 10
    model_dir = abspath("tmp/tf/resnet-18-cifar100/model_overhead/")

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    start_time = time.time()
    with amanda.disabled():
        train(
            take_num=take_num,
            epochs_between_evals=epochs_between_evals,
        )
    end_time = time.time()
    train_time = end_time - start_time
    print(f"train: {train_time}")

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    start_time = time.time()
    tool = DummyTool()
    with amanda.tool.apply(tool):
        train(
            with_hook=True,
            take_num=take_num,
            epochs_between_evals=epochs_between_evals,
        )
    end_time = time.time()
    train_with_hook_time = end_time - start_time
    print(f"train_with_hook: {train_with_hook_time}")

    overhead = (train_with_hook_time / train_time)
    print(f"overhead: {overhead}")


def main_pruning():
    take_num = 1
    epochs_between_evals = 1
    model_dir = abspath("tmp/tf/resnet-18-cifar100/model_overhead/")

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    start_time = time.time()
    with amanda.disabled():
        train(
            take_num=take_num,
            epochs_between_evals=epochs_between_evals,
        )
    end_time = time.time()
    train_time = end_time - start_time

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    start_time = time.time()
    tool = PruningTool(disabled=True)
    with amanda.tool.apply(tool):
        train(
            with_hook=True,
            take_num=take_num,
            epochs_between_evals=epochs_between_evals,
        )
    end_time = time.time()
    train_with_hook_time = end_time - start_time

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    start_time = time.time()
    tool = PruningTool(disabled=False)
    with amanda.tool.apply(tool):
        train(
            with_hook=True,
            take_num=take_num,
            epochs_between_evals=epochs_between_evals,
        )
    end_time = time.time()
    train_with_pruning_time = end_time - start_time

    hook_overhead = (train_with_hook_time / train_time) - 1
    pruning_overhead = (train_with_pruning_time / train_time) - hook_overhead - 1
    print(f"train: {train_time} s")
    print(f"train_with_hook: {train_with_hook_time} s")
    print(f"train_with_pruning: {train_with_pruning_time} s")
    print(f"hook_overhead: {hook_overhead}")
    print(f"pruning_overhead: {pruning_overhead}")


def test():
    tool = DummyTool(debug=True)
    with amanda.tool.apply(tool):
        train(batch_size=1, with_hook=True, take_num=1)
        # train(batch_size=1, with_hook=False, take_num=1)
    assert len(tool.before_executed_ops) != 0
    assert tool.before_executed_ops == tool.after_executed_ops
    assert len(tool.before_executed_backward_ops) != 0
    assert tool.before_executed_backward_ops == tool.after_executed_backward_ops


def test_pruning():
    tool = PruningTool(disabled=False)
    with amanda.tool.apply(tool):
        train(batch_size=1, with_hook=True, take_num=1)


if __name__ == "__main__":
    # main()
    # main_pruning()
    # test()
    test_pruning()
