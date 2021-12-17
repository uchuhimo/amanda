import amanda
import torch
import torchvision
from amanda.cache import cache_disabled

from examples.utils.timer import Timer


def main_origin(model, x):
    # tool = TraceTool(output_dir="tmp/trace_resnet50/tracetool.txt")
    tool = None

    with amanda.tool.apply(tool), amanda.disabled(), amanda.cache.cache_disabled():
        for i in range(5):
            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

        with Timer(verbose=True) as t:

            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

    return t.elapsed


def main_core(model, x):
    # tool = TraceTool(output_dir="tmp/trace_resnet50/tracetool.txt")
    tool = None

    with amanda.tool.apply(tool):
        for i in range(5):
            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

        with Timer(verbose=True) as t:

            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

    return t.elapsed


def main_usecase(model, x):
    tool = TraceTool()
    # tool = TraceTool(output_dir="tmp/trace_resnet50/tracetool.txt")
    # tool = None

    with amanda.tool.apply(tool):
        for i in range(5):
            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

        with Timer(verbose=True) as t:

            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

    return t.elapsed


def main_core_nocache(model, x):
    # tool = TraceTool(output_dir="tmp/trace_resnet50/tracetool.txt")
    tool = None

    with amanda.tool.apply(tool), amanda.cache.cache_disabled():
        for i in range(5):
            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

        with Timer(verbose=True) as t:

            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

    return t.elapsed


def main_usecase_nocache(model, x):
    tool = TraceTool()
    # tool = None

    with amanda.tool.apply(tool), amanda.cache.cache_disabled():
        for i in range(5):
            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

        with Timer(verbose=True) as t:

            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

    return t.elapsed


from examples.trace.pytorch.trace_tool import TraceTool

# from examples.effective_path.pytorch.trace_tool import TraceEffectivePathTool as TraceTool
# from examples.flops_profiler.mapping.tool import FlopsProfileTool as TraceTool
# from examples.qat.pytorch.pruning_tool import PruneTool as TraceTool
# from examples.pruning.pytorch.pruning_tool import PruneTool as TraceTool

if __name__ == "__main__":
    device = "cuda:0"
    batch_size = 128

    # model = torchvision.models.alexnet().to(device)
    # x = torch.zeros((batch_size, 3, 224, 224)).to(device)

    # model = torchvision.models.inception_v3().to(device)
    # x = torch.zeros((batch_size, 3, 299, 299)).to(device)

    # model = torchvision.models.vgg19_bn().to(device)
    # x = torch.zeros((batch_size, 3, 224, 224)).to(device)

    # model = torchvision.models.mobilenet_v2().to(device)
    # x = torch.zeros((batch_size, 3, 224, 224)).to(device)

    # model = torchvision.models.resnet50().to(device)
    # x = torch.rand((batch_size, 3, 227, 227)).to(device)

    import transformers

    model = transformers.models.bert.BertForMaskedLM.from_pretrained(
        "bert-base-uncased"
    ).to(device)
    x = torch.zeros((batch_size, 128), dtype=torch.long).to(device)

    origin_time = main_origin(model, x)
    usecase_time = main_usecase(model, x)
    core_time = main_core(model, x)
    usecase_time_nocache = main_usecase_nocache(model, x)
    core_time_nocache = main_core_nocache(model, x)

    print(f"origin time {origin_time}")
    print(f"core time {(core_time-origin_time)/origin_time}")
    print(f"usecase time {(usecase_time-core_time)/origin_time}")
    print(f"core no cache time {(core_time_nocache-origin_time)/origin_time}")
    print(
        f"usecase no cache time {(usecase_time_nocache-core_time_nocache)/origin_time}"
    )

    print("===========================")
    print(f"{abs(origin_time)}")
    print(f"{abs((core_time-origin_time)/origin_time)}")
    print(f"{abs((usecase_time-core_time)/origin_time)}")
    print(f"{abs((core_time_nocache-origin_time)/origin_time)}")
    print(f"{abs((usecase_time_nocache-core_time_nocache)/origin_time)}")
