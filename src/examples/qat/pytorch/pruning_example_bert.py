import amanda
import torch
import torchvision
from amanda.cache import cache_disabled

from examples.trace.pytorch.trace_tool import TraceTool
from examples.utils.timer import Timer


def main_origin(model, x):
    device = "cuda"
    batch_size = 128

    # tool = TraceTool(output_dir="tmp/trace_resnet50/tracetool.txt")
    tool = None
    # from examples.pruning.pytorch.pruning_tool import PruneTool
    # tool = PruneTool()

    # with amanda.tool.apply(tool):
    # with amanda.tool.apply(tool), amanda.disabled():
    with amanda.tool.apply(tool), amanda.disabled(), amanda.cache.cache_disabled():
        # with amanda.tool.apply(tool), amanda.cache.cache_disabled():
        for i in range(5):
            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

        num_steps = 10
        total_time = 0
        for i in range(num_steps):
            with Timer(verbose=True) as t:

                y = model(x)
                if not isinstance(y, torch.Tensor):
                    y = y[0]
                y.backward(torch.rand_like(y))
            total_time += t.elapsed
        print(f"avg time with warmup: {total_time/num_steps}")
    return total_time / num_steps


def main_core(model, x):
    device = "cuda"
    batch_size = 128

    # tool = TraceTool(output_dir="tmp/trace_resnet50/tracetool.txt")
    tool = None
    from examples.pruning.pytorch.pruning_tool import PruneTool

    # tool = PruneTool()

    with amanda.tool.apply(tool):
        # with amanda.tool.apply(tool), amanda.disabled():
        # with amanda.tool.apply(tool), amanda.disabled(), amanda.cache.cache_disabled():
        # with amanda.tool.apply(tool), amanda.cache.cache_disabled():
        for i in range(5):
            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

        num_steps = 10
        total_time = 0
        for i in range(num_steps):
            with Timer(verbose=True) as t:

                y = model(x)
                if not isinstance(y, torch.Tensor):
                    y = y[0]
                y.backward(torch.rand_like(y))
            total_time += t.elapsed
        print(f"avg time with warmup: {total_time/num_steps}")
    return total_time / num_steps


def main_usecase(model, x):
    device = "cuda"
    batch_size = 128

    # tool = TraceTool(output_dir="tmp/trace_resnet50/tracetool.txt")
    # tool = None
    from examples.pruning.pytorch.pruning_tool import PruneTool

    tool = PruneTool()

    with amanda.tool.apply(tool):
        # with amanda.tool.apply(tool), amanda.disabled():
        # with amanda.tool.apply(tool), amanda.disabled(), amanda.cache.cache_disabled():
        # with amanda.tool.apply(tool), amanda.cache.cache_disabled():
        for i in range(5):
            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

        num_steps = 10
        total_time = 0
        for i in range(num_steps):
            with Timer(verbose=True) as t:

                y = model(x)
                if not isinstance(y, torch.Tensor):
                    y = y[0]
                y.backward(torch.rand_like(y))
            total_time += t.elapsed
        print(f"avg time with warmup: {total_time/num_steps}")
    return total_time / num_steps


def main_core_nocache(model, x):
    device = "cuda"
    batch_size = 128

    # tool = TraceTool(output_dir="tmp/trace_resnet50/tracetool.txt")
    tool = None
    from examples.pruning.pytorch.pruning_tool import PruneTool

    # tool = PruneTool()
    # with amanda.tool.apply(tool):
    # with amanda.tool.apply(tool), amanda.disabled():
    # with amanda.tool.apply(tool), amanda.disabled(), amanda.cache.cache_disabled():
    with amanda.tool.apply(tool), amanda.cache.cache_disabled():
        for i in range(5):
            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

        num_steps = 10
        total_time = 0
        for i in range(num_steps):
            with Timer(verbose=True) as t:

                y = model(x)
                if not isinstance(y, torch.Tensor):
                    y = y[0]
                y.backward(torch.rand_like(y))
            total_time += t.elapsed
        print(f"avg time with warmup: {total_time/num_steps}")
    return total_time / num_steps


def main_usecase_nocache(model, x):
    device = "cuda"
    batch_size = 128

    # tool = TraceTool(output_dir="tmp/trace_resnet50/tracetool.txt")
    # tool = None
    from examples.pruning.pytorch.pruning_tool import PruneTool

    tool = PruneTool()

    # with amanda.tool.apply(tool):
    # with amanda.tool.apply(tool), amanda.disabled():
    # with amanda.tool.apply(tool), amanda.disabled(), amanda.cache.cache_disabled():
    with amanda.tool.apply(tool), amanda.cache.cache_disabled():
        for i in range(5):
            y = model(x)
            if not isinstance(y, torch.Tensor):
                y = y[0]
            y.backward(torch.rand_like(y))

        num_steps = 10
        total_time = 0
        for i in range(num_steps):
            with Timer(verbose=True) as t:

                y = model(x)
                if not isinstance(y, torch.Tensor):
                    y = y[0]
                y.backward(torch.rand_like(y))
            total_time += t.elapsed
        print(f"avg time with warmup: {total_time/num_steps}")
    return total_time / num_steps


if __name__ == "__main__":
    device = "cuda"
    batch_size = 128

    # model = torchvision.models.resnet50().to(device)
    # x = torch.rand((2, 3, 227, 227)).to(device)

    # import transformers
    # model = transformers.models.bert.BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    # x = torch.zeros((128,128),dtype=torch.long).to(device)

    # model = torchvision.models.inception_v3().to(device)
    # x = torch.zeros((batch_size, 3, 299, 299)).to(device)

    model = torchvision.models.alexnet().to(device)
    x = torch.zeros((batch_size, 3, 224, 224)).to(device)

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
