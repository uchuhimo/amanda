import amanda
from amanda.tools.pruning.model import train_loop
from amanda.tools.pruning.pruning_tool import PruningTool


def main(model, optimizer, args):
    pruning_tool = PruningTool(mask_calculator="m4n2_1d")
    amanda.apply(model, pruning_tool)

    step = 0
    # train for a few steps with dense weights
    step = train_loop(args, model, optimizer, step, args.num_dense_steps)
    # simulate sparsity by inserting zeros into existing dense weights
    pruning_tool.compute_sparse_masks()
    # train for a few steps with sparse weights
    step = train_loop(args, model, optimizer, step, args.num_sparse_steps)
