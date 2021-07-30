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

import tensorflow as tf

import amanda
from amanda.io.file import abspath
from examples.common.tensorflow.dataset.cifar100_main import input_fn
from examples.common.tensorflow.dataset.envs import CIFAR100_RAW_DIR
from examples.common.tensorflow.utils import new_session_config
from examples.pruning.tensorflow.pruning_tool import PruningTool
from examples.pruning.tensorflow.resnet_18_cifar100_train import cifar100_model_fn


def train(
    batch_size: int = 128,
    train_epochs: int = 182,
    epochs_between_evals: int = 1,
    multi_gpu: bool = False,
    label: str = None,
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
        return input

    # Set up training hook that logs the training accuracy every 100 steps.
    tensors_to_log = {"train_accuracy": "train_accuracy"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

    print(label)
    print()


def main():
    tool = PruningTool()
    with amanda.tool.apply(tool):
        train()


if __name__ == "__main__":
    main()
