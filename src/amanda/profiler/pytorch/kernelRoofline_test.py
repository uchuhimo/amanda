import amanda
import torch
import torchvision
from amanda_profiler import amandaProfiler

def main():

	device = "cuda"

	model = torchvision.models.resnet50().to(device)
	x = torch.rand((32, 3, 227, 227)).to(device)

	metric = "KernelRoofline"
	# Nvidia Geforce RTX 2080 Ti: 1350MHz, 13.45 Single-Precision TFlops, 616GB/s
	supplyInfo = [1350, 13.45, 616]
	profiler = amandaProfiler(metric)
	profiler.setConfigs(metric=metric, supplyInfo=supplyInfo)

	with amanda.tool.apply(profiler.counter):
		y = model(x)

	profiler.showResults()

if __name__ == "__main__":
	main()