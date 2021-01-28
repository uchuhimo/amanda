from apex.contrib.sparsity.amanda_tf.pruning_tool import PruningTool
from collections import OrderedDict
import amanda
import tensorflow as tf
import torch
from apex.optimizers import FusedAdam
from functools import partial
import numpy as np

class MLP:
    def __init__(self, args):
        self.fc1 = tf.layers.Dense(args.hidden_features, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(args.hidden_features, activation=tf.nn.relu)
        self.fc3 = tf.layers.Dense(args.hidden_features, activation=tf.nn.relu)
        self.fc4 = tf.layers.Dense(args.output_features)
        self.bn1 = tf.layers.BatchNormalization()
        self.bn2 = tf.layers.BatchNormalization()
        self.bn3 = tf.layers.BatchNormalization()

    def __call__(self, inputs, training=False):
        y = self.fc1(inputs)
        y = self.bn1(y, training=training)
        y = self.fc2(y)
        y = self.bn2(y, training=training)
        y = self.fc3(y)
        y = self.bn3(y, training=training)
        return self.fc4(y)

def model_fn(features, labels, mode, params, args):
    """The model_fn argument for creating an Estimator."""
    model = MLP(args)
    image = features

    logits = model(image, training=(mode == tf.estimator.ModeKeys.TRAIN))
    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    loss = tf.losses.mean_squared_error(labels, logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
    )


def train_loop(args, classifier, step, num_steps):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(
            (np.random.rand(args.batch_size, args.input_features).astype(np.float32),
             np.random.rand(args.batch_size, args.output_features).astype(np.float32)))
        return dataset.batch(1)

    classifier.train(input_fn=input_fn, steps=num_steps)
    step += num_steps
    return step

def get_checkpoint(estimator):
    return amanda.Checkpoint("tensorflow_checkpoint", estimator.latest_checkpoint())

def main(args):
    estimator = tf.estimator.Estimator(
        model_fn=partial(model_fn, args=args),
    )
    weight_name = "dense_2/kernel"

    pruning_tool = PruningTool(mask_calculator="m4n2_1d", whitelist=["MatMul"])

    step = 0

    pruning_tool.init_masks()

    # train for a few steps with dense weights
    step = train_loop(args, estimator, step, args.num_dense_steps)
    print("DENSE :: ",estimator.get_variable_value(weight_name))
    amanda.apply(get_checkpoint(estimator), pruning_tool)
    print(estimator.latest_checkpoint())

    # simulate sparsity by inserting zeros into existing dense weights
    pruning_tool.compute_masks()
    amanda.apply(get_checkpoint(estimator), pruning_tool)
    print(estimator.latest_checkpoint())

    # train for a few steps with sparse weights
    print("SPARSE :: ",estimator.get_variable_value(weight_name))
    step = train_loop(args, estimator, step, args.num_sparse_steps)
    print(estimator.latest_checkpoint())

    # recompute sparse masks
    pruning_tool.compute_masks()
    amanda.apply(get_checkpoint(estimator), pruning_tool)

    # train for a few steps with sparse weights
    print("SPARSE :: ",estimator.get_variable_value(weight_name))
    step = train_loop(args, estimator, step, args.num_sparse_steps_2)

    pruning_tool.remove_masks()
    amanda.apply(get_checkpoint(estimator), pruning_tool)

    print("SPARSE :: ",estimator.get_variable_value(weight_name))

if __name__ == '__main__':
    class Args:
        batch_size = 32
        input_features = 16
        output_features = 8
        hidden_features = 40
        num_layers = 4
        num_dense_steps = 2000
        num_sparse_steps = 3000
        num_sparse_steps_2 = 1000
        num_dense_steps_2 = 1500
    args = Args()

    main(args)
