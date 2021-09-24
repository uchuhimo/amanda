import torch

from amanda.conversion.listener.build.amanda_pybind import amanda_add_pre_hook


def dummy_hook(inputs):
    print("hello world")
    print(inputs)
    return inputs


def test_grad_modification():
    a = torch.rand((2, 2), requires_grad=True)
    b = torch.rand((2, 2))

    c = a + b

    # amanda_hook.add_pre_hook(c.grad_fn, dummy_hook)
    amanda_add_pre_hook(c.grad_fn, dummy_hook)

    c.backward(torch.rand_like(c))

    print(a.grad)


def test_multiple_outputs():
    model = torch.nn.RNN(
        input_size=16,
        hidden_size=16,
        num_layers=2,
    )

    input_embedding = torch.rand((2, 32, 16))

    outputs = model(input_embedding)

    print([i.grad_fn for i in outputs])


def test_split():
    def split_hook(inputs):
        print("hello world")
        print(inputs)
        # return (inputs[0], torch.zeros_like(inputs[1]))
        return inputs

    # init_THPVariableClass()

    x = torch.rand((4, 4), requires_grad=True)

    outputs = torch.split(x, 2, dim=0)

    y = outputs[0] + outputs[1]

    print([(i.grad_fn, i.output_nr) for i in outputs])

    handle = amanda_add_pre_hook(outputs[0].grad_fn, split_hook)  # noqa: F841

    y.backward(torch.rand_like(y))
    # y.backward(torch.rand_like(y), retain_graph=True)

    print(x.grad)

    # amanda_remove_pre_hook(outputs[0].grad_fn, handle)

    # y.backward(torch.rand_like(y))


def test_resnet():
    def dummy_hook(inputs):
        print(f"{inputs[0].shape}")
        return inputs

    import torchvision

    # init_THPVariableClass()

    model = torchvision.models.resnet50()

    x = torch.rand((1, 3, 227, 227))

    y = model(x)

    # amanda_add_pre_hook(y.grad_fn, dummy_hook)
    grad_fns = [y.grad_fn]
    while grad_fns:
        fn = grad_fns.pop(0)
        # amanda_add_pre_hook(fn, lambda inputs: inputs)
        amanda_add_pre_hook(fn, dummy_hook)
        for next_fn, output_nr in fn.next_functions:
            if next_fn and next_fn not in grad_fns:
                grad_fns.append(next_fn)

    y.backward(torch.rand_like(y))

    print(y.shape)


if __name__ == "__main__":
    # test_multiple_outputs()
    # test_split()
    test_resnet()
