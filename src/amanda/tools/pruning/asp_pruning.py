# type: ignore
from amanda.tools.pruning.asp import ASP
from amanda.tools.pruning.model import train_loop


def main(model, optimizer, args):
    ASP.init_model_for_pruning(model, mask_calculator="m4n2_1d")
    ASP.init_optimizer_for_pruning(optimizer)

    step = 0
    # train for a few steps with dense weights
    step = train_loop(args, model, optimizer, step, args.num_dense_steps)
    # simulate sparsity by inserting zeros into existing dense weights
    ASP.compute_sparse_masks()
    # train for a few steps with sparse weights
    step = train_loop(args, model, optimizer, step, args.num_sparse_steps)
