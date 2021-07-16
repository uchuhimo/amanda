import inspect

import torch

import amanda

# import torch.nn


def foo(*args, **kwargs):
    args[0].zero_()


def prune(*args, **kwargs):
    mask = torch.eye(3)
    weight_grad = args[0][1]
    weight_grad *= mask


def test_tensor_hook():
    # a = torch.rand((2,2), requires_grad=True)
    # b = torch.rand((2,2), requires_grad=True)

    # # b_mask = torch.eye(2, requires_grad=True)
    # # b_pruned = torch.mul(b, b_mask)

    # # c = a * b_pruned

    # # c.register_hook(foo)
    # # c.grad_fn._register_hook_dict(c)

    # c.backward(torch.rand(2,2))

    # print(a.grad, b.grad)

    weight = torch.rand((1, 1, 3, 3), requires_grad=True)

    # mask = torch.eye(3)

    # weight_pruned = torch.mul(weight, mask)

    # print(weight_pruned)
    with torch.no_grad():
        weight.data = weight * torch.eye(3)

    input = torch.rand((1, 1, 5, 5), requires_grad=True)
    output = torch.conv2d(input, weight)
    output.grad_fn.register_hook(prune)
    # print(output)

    output.backward(torch.rand(output.shape))
    print(input.grad, weight.grad)
    # print(a.grad, b.grad)


def test_hook_removal():
    a = torch.rand((2, 2), requires_grad=True)
    b = torch.rand((2, 2), requires_grad=True)

    c = torch.rand((2, 2), requires_grad=True)

    d = torch.mul(a, b)

    _ = torch.mul(c, a)

    print(d._backward_hooks)

    # print(c._backward_hooks)


def test_object_id():
    weight = torch.rand((1, 1, 3, 3), requires_grad=True)
    input = torch.rand((1, 1, 5, 5), requires_grad=True)
    _ = torch.conv2d(input, weight)
    _ = torch.conv2d(input, weight)


def test_vector_wise():
    from vector_wise_sparsity import create_mask

    weight = torch.rand((4, 4, 3, 3), device="cuda")

    print(create_mask(weight))


def test_import():
    print([i[0] for i in inspect.getmembers(torch.nn.modules.pooling.MaxPool2d)])
    # print(torch.nn.modules.pooling.MaxPool2d.__dict__)


def test_vgg():
    class TraceTool(amanda.Tool):
        def __init__(self):
            super(TraceTool, self).__init__(namespace="amanda/pytorch")
            self.register_event(amanda.event.before_op_executed, self.print_op_name)

        def print_op_name(self, context):
            if (
                "conv2d" in context["op"].__name__
                and context["args"][1].shape[1] % 4 == 0
            ) or (
                "matmul" in context["op"].__name__
                and len(context["args"][1].shape) == 2
            ):
                print(context["op"].__name__)
                # print(context['op'].grad_fn)

    import torchvision

    model = torchvision.models.vgg11(num_classes=100)

    input = torch.rand((8, 3, 128, 128))

    with amanda.conversion.pytorch_updater.apply(TraceTool()):

        output = model(input)

        output.backward(torch.rand_like(output))

    print(output.shape)


def test_broadcast():
    class TraceTool(amanda.Tool):
        def __init__(self):
            super(TraceTool, self).__init__(namespace="amanda/pytorch")
            self.register_event(amanda.event.before_op_executed, self.print_op_name)

        def print_op_name(self, context):
            print(context["op"].__name__)

    with amanda.conversion.pytorch_updater.apply(TraceTool()):

        weight = torch.rand((768, 768), requires_grad=True)
        input = torch.rand((1, 512, 768), requires_grad=True)

        output = torch.matmul(input, weight)

        # output.grad_fn.register_hook(lambda input, output: print(input[1].shape))

        output.backward(torch.rand_like(output))

        print(input.grad.shape, weight.grad.shape)


def test_function_hook():

    device = "cpu"

    x = torch.rand((1, 512, 768), requires_grad=True, device=device)
    weight = torch.rand((768, 768), requires_grad=True, device=device)

    print("shape of inputs", x.shape, weight.shape)

    y = torch.matmul(x, weight)

    y.grad_fn.next_functions[0][0].next_functions[0][0].register_hook(
        lambda input, output: print(
            f"shape of inputs grad in hook {[i.shape for i in input]}"
        )
    )

    print(y.grad_fn.next_functions[0][0].next_functions)

    print(y.grad_fn)

    y.backward(torch.rand_like(y))

    print("shape of inputs grad after backward", x.grad.shape, weight.grad.shape)


def test_forward():
    import torchvision

    model = torchvision.models.resnet50()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters())

    input = torch.rand((8, 3, 227, 227))
    label = torch.rand(8, 1)

    import apex

    apex.contrib.sparsity.prune_trained_model(model, optimizer)
    output = model(input)
    loss = criterion(output, label)

    loss.backward()
    optimizer.step()


def test_transformer():
    class TraceTool(amanda.Tool):
        def __init__(self):
            super(TraceTool, self).__init__(namespace="amanda/pytorch")
            self.register_event(
                amanda.event.after_backward_op_executed, self.print_op_name
            )

        def print_op_name(self, context):
            if "matmul" in context["op"].__name__:
                print(context["op"].__name__)
                print([i.shape for i in context["args"]])
                print([i.shape for i in context["input_grad"]])

    import transformers

    config = transformers.BertConfig.from_pretrained("bert-base-uncased")
    model = transformers.BertModel(config)

    input = torch.randint(1, 100, (2, 4))

    with amanda.conversion.pytorch_updater.apply(TraceTool()):
        output = model(input)
        output[0].backward(torch.rand_like(output[0]))


def test_vgg_conv_module():
    # device = 'cpu'
    device = "cuda"

    input = torch.rand(128, 3, 32, 32).to(device)

    # model = torchvision.models.vgg16(num_classes=100).to(device)
    model = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True).to(device)

    output = model(input)

    print(output.shape)

    print(output.grad_fn.next_functions)

    # print(output.grad_fn.next_functions)


if __name__ == "__main__":
    # test_tensor_hook()
    # test_hook_removal()
    # test_object_id()
    # test_vector_wise()

    # test_import()

    # test_vgg()
    # test_transformer()

    # test_function_hook()

    test_vgg_conv_module()
