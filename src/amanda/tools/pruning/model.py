from collections import OrderedDict

import torch


def build_model(args):
    od = OrderedDict()
    for i in range(args.num_layers):
        if i == 0:
            od["linear_layer_%d" % (i + 1)] = torch.nn.Linear(
                args.input_features, args.hidden_features
            )
            od["layer_norm_%d" % (i + 1)] = torch.nn.LayerNorm(
                [args.batch_size, args.hidden_features]
            )
        elif i == args.num_layers - 1:
            od["linear_layer_%d" % (i + 1)] = torch.nn.Linear(
                args.hidden_features, args.output_features
            )
            od["layer_norm_%d" % (i + 1)] = torch.nn.LayerNorm(
                [args.batch_size, args.output_features]
            )
        else:
            od["linear_layer_%d" % (i + 1)] = torch.nn.Linear(
                args.hidden_features, args.hidden_features
            )
            od["layer_norm_%d" % (i + 1)] = torch.nn.LayerNorm(
                [args.batch_size, args.hidden_features]
            )
    return torch.nn.Sequential(od)


def train_step(args, model, optimizer, input_batch, target_batch, step):
    predicted_target = model(input_batch)
    loss = ((predicted_target - target_batch) ** 2).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    step = step + 1
    # print("Step %d :: loss=%e" % (step, loss.item()))
    return step


def train_loop(args, model, optimizer, step, num_steps):
    for i in range(num_steps):
        input_batch = torch.randn([args.batch_size, args.input_features]).cuda()
        target_batch = torch.randn([args.batch_size, args.output_features]).cuda()
        step = train_step(args, model, optimizer, input_batch, target_batch, step)
    return step
