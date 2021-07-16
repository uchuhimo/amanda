import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

import amanda

dataset = "Cora"
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class TraceTool(amanda.Tool):
    def __init__(self):
        super(TraceTool, self).__init__(namespace="amanda/pytorch")
        self.register_event(amanda.event.before_op_executed, self.print_op_name)

    def print_op_name(self, context):
        if context["op"].__name__ in [
            "size",
            "shape",
            "reshape",
            "unbind",
            "item",
            "isfinite",
            "ne",
            "masked_select",
            "abs",
            "min",
            "max",
            "ceil",
            "gt",
            "lt",
        ]:
            return

        # print(f"input of {context['op'].__name__}: {context['args']})")
        print(f"input of {context['op'].__name__}: {len(context['args'])})")
        # print(context['args'])
        # for i in context['args']:
        #     print(type(i))
        #     if type(i) == torch.Tensor:
        #         print(i.shape)
        #         print(i.data)
        # print(context['op'].grad_fn)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(
            8 * 8, dataset.num_classes, heads=1, concat=False, dropout=0.6
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, data = Net().to(device), data.to(device)  # type: ignore

with amanda.tool.apply(TraceTool()):
    model(data.x, data.edge_index)  # type: ignore
