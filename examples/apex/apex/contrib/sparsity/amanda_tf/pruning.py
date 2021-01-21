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
        self.fc3 = tf.layers.Dense(args.output_features)

    def __call__(self, inputs, training=False):
        y = self.fc1(inputs)
        y = self.fc2(y)
        return self.fc3(y)

def model_fn(features, labels, mode, params, args):
    """The model_fn argument for creating an Estimator."""
    model = MLP(args)
    image = features

    # Generate a summary node for the images
    # tf.summary.image("images", features, max_outputs=6)

    logits = model(image, training=False)
    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.mean_squared_error(labels, logits)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name="cross_entropy")
    tf.summary.scalar("cross_entropy", cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def loss_filter_fn(name):
        return "batch_normalization" not in name

    weight_decay = 1e-4
    # Add weight decay to the loss.
    loss = cross_entropy + weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables() if loss_filter_fn(v.name)]
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=0.001, momentum=0.9
        )
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

def main(args):
    classifier = tf.estimator.Estimator(
        model_fn=partial(model_fn, args=args),
    )
    weight_name = "dense_2/kernel"

    print(classifier.latest_checkpoint())

    pruning_tool = PruningTool(mask_calculator="m4n2_1d", whitelist=["MatMul"])

    step = 0

    # train for a few steps with dense weights
    step = train_loop(args, classifier, step, args.num_dense_steps)
    print("DENSE :: ",classifier.get_variable_value(weight_name))

    print(classifier.latest_checkpoint())

    amanda.apply(classifier, pruning_tool)
    # simulate sparsity by inserting zeros into existing dense weights
    pruning_tool.compute_sparse_masks()

    # train for a few steps with sparse weights
    print("SPARSE :: ",classifier.get_variable_value(weight_name))
    step = train_loop(args, classifier, step, args.num_sparse_steps)

    # recompute sparse masks
    pruning_tool.compute_sparse_masks()

    # train for a few steps with sparse weights
    print("SPARSE :: ",classifier.get_variable_value(weight_name))
    step = train_loop(args, classifier, step, args.num_sparse_steps_2)
    # pruning_tool.mask_weights()

    # turn off sparsity
    print("SPARSE :: ",classifier.get_variable_value(weight_name))
    pruning_tool.restore_pruned_weights()

    # train for a few steps with dense weights
    print("DENSE :: ",classifier.get_variable_value(weight_name))
    step = train_loop(args, classifier, step, args.num_dense_steps_2)

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
