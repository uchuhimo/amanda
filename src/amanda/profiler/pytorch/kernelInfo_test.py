import amanda
import torch
import torchvision
from amanda_profiler import amandaProfiler

def main():

	device = "cuda"

	model = torchvision.models.resnet50().to(device)
	x = torch.rand((32, 3, 227, 227)).to(device)

	metric = "KernelInfo"
	profiler = amandaProfiler(metric)
	profiler.setConfigs(metric=metric)

	with amanda.tool.apply(profiler.counter):
		y = model(x)
	with amanda.tool.apply(profiler.tracer):
		y = model(x)

	profiler.showResults()

if __name__ == "__main__":
	main()