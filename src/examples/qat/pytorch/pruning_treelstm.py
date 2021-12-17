# import amanda
# from amanda.conversion.pytorch_updater import apply

import sys
from timeit import default_timer as timer

import torch
import torch.nn.utils.prune as prune

from examples.pruning.treelstm import TreeLSTM, calculate_evaluation_orders
from examples.pruning.vector_wise_sparsity import create_mask


class VectorWisePruningMethod(prune.BasePruningMethod):
    def compute_mask(self, t, default_mask):
        # mask = create_mask(t)
        mask = torch.rand_like(t)
        return mask


if "amanda" in sys.modules:
    import amanda

    class PruneTool(amanda.Tool):
        def __init__(self):
            super(PruneTool, self).__init__(namespace="amanda/pytorch")
            self.register_event(
                amanda.event.before_op_executed, self.mask_forward_weight
            )
            self.register_event(
                amanda.event.after_backward_op_executed, self.mask_backward_gradient
            )

            self.conv_cnt = 0
            self.conv_masks = []

        def compute_mask(self, tensor):
            print(f"compute mask for {tensor.shape}")
            return torch.rand_like(tensor)
            return create_mask(tensor)

        def get_mask(self, tensor):
            if self.conv_cnt <= len(self.conv_masks):
                return self.conv_masks[self.conv_cnt - 1]
            else:
                mask = self.compute_mask(tensor)
                self.conv_masks.append(mask)
                return mask

        def mask_forward_weight(self, context):

            if (
                "conv2d" in context["op"].__name__
                and context["args"][1].shape[1] % 4 == 0
            ) or (
                "matmul" in context["op"].__name__
                and len(context["args"][1].shape) == 2
            ):
                self.conv_cnt += 1

                weight = context["args"][1]
                mask = self.get_mask(weight)
                with torch.no_grad():
                    weight.data = torch.mul(weight, mask)
                    context["mask"] = mask

        def mask_backward_gradient(self, context):
            if (
                "conv2d" in context["op"].__name__
                and context["args"][1].shape[1] % 4 == 0
            ) or (
                "matmul" in context["op"].__name__
                and len(context["args"][1].shape) == 2
            ):
                weight_grad = context["input_grad"][1]
                mask = context["mask"]
                # print(context['args'][0].shape, context['args'][1].shape)
                # print(context['input_grad'][0].shape, context['input_grad'][1].shape)
                with torch.no_grad():
                    weight_grad.data = torch.mul(weight_grad, mask)

        def reset_cnt(self):
            self.conv_cnt = 0


def _label_node_index(node, n=0):
    node["index"] = n
    for child in node["children"]:
        n += 1
        _label_node_index(child, n)


def _gather_node_attributes(node, key):
    features = [node[key]]
    for child in node["children"]:
        features.extend(_gather_node_attributes(child, key))
    return features


def _gather_adjacency_list(node):
    adjacency_list = []
    for child in node["children"]:
        adjacency_list.append([node["index"], child["index"]])
        adjacency_list.extend(_gather_adjacency_list(child))

    return adjacency_list


def convert_tree_to_tensors(tree, device=torch.device("cpu")):
    # Label each node with its walk order to match nodes to feature tensor indexes
    # This modifies the original tree as a side effect
    _label_node_index(tree)

    features = _gather_node_attributes(tree, "features")
    labels = _gather_node_attributes(tree, "labels")
    adjacency_list = _gather_adjacency_list(tree)

    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))

    return {
        "features": torch.tensor(features, device=device, dtype=torch.float32),
        "labels": torch.tensor(labels, device=device, dtype=torch.float32),
        "node_order": torch.tensor(node_order, device=device, dtype=torch.int64),
        "adjacency_list": torch.tensor(
            adjacency_list, device=device, dtype=torch.int64
        ),
        "edge_order": torch.tensor(edge_order, device=device, dtype=torch.int64),
    }


if __name__ == "__main__":
    # Toy example
    tree = {
        "features": [1, 0],
        "labels": [1],
        "children": [
            {"features": [0, 1], "labels": [0], "children": []},
            {
                "features": [0, 0],
                "labels": [0],
                "children": [{"features": [1, 1], "labels": [0], "children": []}],
            },
        ],
    }

    data = convert_tree_to_tensors(tree)

    model = TreeLSTM(2, 1).train()

    # for name, module in model.named_modules():
    #     if name == 'conv1' or name=='features.0': # skip input layer with input dim=3
    #         continue
    #     if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
    #         VectorWisePruningMethod.apply(module, 'weight')

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # tool = PruneTool()
    tool = None

    for n in range(16):
        optimizer.zero_grad()

        # with apply(tool):
        if True:

            if tool is not None and hasattr(tool, "reset_cnt"):
                tool.reset_cnt()

            start = timer()

            h, c = model(
                data["features"],
                data["node_order"],
                data["adjacency_list"],
                data["edge_order"],
            )

            labels = data["labels"]

            loss = loss_function(h, labels)
            loss.backward()
        optimizer.step()

        end = timer()

        print(end - start)

        # print(f'Iteration {n+1} Loss: {loss}')
    print(data)
