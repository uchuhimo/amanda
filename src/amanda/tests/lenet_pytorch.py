import torch.jit

import amanda

if __name__ == "__main__":
    input = torch.randn(1, 28 * 28)
    model = torch.nn.Sequential(
        torch.nn.Linear(28 * 28, 100, bias=False), torch.nn.ReLU(),
    )
    traced_model = torch.jit.trace(model, (input,))
    logits = traced_model(input)
    graph = amanda.pytorch.import_from_module(traced_model)
    graph.print()
