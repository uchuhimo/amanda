from apex.contrib.sparsity.amanda.pruning_tool import PruningTool
from collections import OrderedDict
import amanda

import torch
from apex.optimizers import FusedAdam

def build_model(args):
    od = OrderedDict()
    for i in range(args.num_layers):
        if i == 0:
            od['linear_layer_%d' % (i+1)] = torch.nn.Linear(args.input_features, args.hidden_features)
            od['layer_norm_%d' % (i+1)] = torch.nn.LayerNorm([args.batch_size, args.hidden_features])
        elif i == args.num_layers-1:
            od['linear_layer_%d' % (i+1)] = torch.nn.Linear(args.hidden_features, args.output_features)
            od['layer_norm_%d' % (i+1)] = torch.nn.LayerNorm([args.batch_size, args.output_features])
        else:
            od['linear_layer_%d' % (i+1)] = torch.nn.Linear(args.hidden_features, args.hidden_features)
            od['layer_norm_%d' % (i+1)] = torch.nn.LayerNorm([args.batch_size, args.hidden_features])
    return torch.nn.Sequential(od)

def train_step(args, model, optimizer, input_batch, target_batch, step):
    predicted_target = model(input_batch)
    loss = ((predicted_target-target_batch)**2).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    step = step + 1
    #print("Step %d :: loss=%e" % (step, loss.item()))
    return step

def train_loop(args, model, optimizer, step, num_steps):
    for i in range(num_steps):
        input_batch = torch.randn([args.batch_size, args.input_features]).cuda()
        target_batch = torch.randn([args.batch_size, args.output_features]).cuda()
        step = train_step(args, model, optimizer, input_batch, target_batch, step)
    return step

def main(args):
    model = build_model(args).cuda()
    one_ll = next(model.children()).weight
    optimizer = FusedAdam(model.parameters())

    pruning_tool = PruningTool(mask_calculator="m4n2_1d", whitelist=[torch.nn.Linear])
    amanda.apply(model, pruning_tool)

    step = 0

    pruning_tool.init_masks()

    # train for a few steps with dense weights
    step = train_loop(args, model, optimizer, step, args.num_dense_steps)
    print("DENSE :: ",one_ll)

    # simulate sparsity by inserting zeros into existing dense weights
    pruning_tool.compute_masks()

    # train for a few steps with sparse weights
    print("SPARSE :: ",one_ll)
    step = train_loop(args, model, optimizer, step, args.num_sparse_steps)

    # recompute sparse masks
    pruning_tool.compute_masks()

    # train for a few steps with sparse weights
    print("SPARSE :: ",one_ll)
    step = train_loop(args, model, optimizer, step, args.num_sparse_steps_2)

    pruning_tool.remove_masks()
    
    print("SPARSE :: ",one_ll)

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
