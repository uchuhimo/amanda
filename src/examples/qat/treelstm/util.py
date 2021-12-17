"""Helper functions for running the TreeLSTM model
"""

import numpy
import torch


def calculate_evaluation_orders(adjacency_list, tree_size):
    """Calculates the node_order and edge_order from a tree adjacency_list and the tree_size.
    The TreeLSTM model requires node_order and edge_order to be passed into the model along
    with the node features and adjacency_list.  We pre-calculate these orders as a speed
    optimization.
    """
    adjacency_list = numpy.array(adjacency_list)

    node_ids = numpy.arange(tree_size, dtype=int)

    node_order = numpy.zeros(tree_size, dtype=int)
    unevaluated_nodes = numpy.ones(tree_size, dtype=bool)

    parent_nodes = adjacency_list[:, 0]
    child_nodes = adjacency_list[:, 1]

    n = 0
    while unevaluated_nodes.any():
        # Find which child nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[child_nodes]

        # Find the parent nodes of unevaluated children
        unready_parents = parent_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of parents with unevaluated child nodes
        nodes_to_evaluate = unevaluated_nodes & ~numpy.isin(node_ids, unready_parents)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    edge_order = node_order[parent_nodes]

    return node_order, edge_order


def batch_tree_input(batch):
    """Combines a batch of tree dictionaries into a single batched dictionary for use by the TreeLSTM model.
    batch - list of dicts with keys ('features', 'node_order', 'edge_order', 'adjacency_list')
    returns a dict with keys ('features', 'node_order', 'edge_order', 'adjacency_list', 'tree_sizes')
    """
    tree_sizes = [b["features"].shape[0] for b in batch]

    batched_features = torch.cat([b["features"] for b in batch])
    batched_node_order = torch.cat([b["node_order"] for b in batch])
    batched_edge_order = torch.cat([b["edge_order"] for b in batch])

    batched_adjacency_list = []
    offset = 0
    for n, b in zip(tree_sizes, batch):
        batched_adjacency_list.append(b["adjacency_list"] + offset)
        offset += n
    batched_adjacency_list = torch.cat(batched_adjacency_list)

    return {
        "features": batched_features,
        "node_order": batched_node_order,
        "edge_order": batched_edge_order,
        "adjacency_list": batched_adjacency_list,
        "tree_sizes": tree_sizes,
    }


def unbatch_tree_tensor(tensor, tree_sizes):
    """Convenience functo to unbatch a batched tree tensor into individual tensors given an array of tree_sizes.
    sum(tree_sizes) must equal the size of tensor's zeroth dimension.
    """
    return torch.split(tensor, tree_sizes, dim=0)
