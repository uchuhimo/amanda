from asyncore import file_dispatcher
import amanda
import torch
import torchvision

from amanda_counter import amandaCounter

def main():

	device = "cuda"

	model = torchvision.models.resnet50().to(device)
	x = torch.rand((32, 3, 227, 227)).to(device)

	filePath = "kernel_metrics2.txt"
	counter = amandaCounter(filePath=filePath)

	# dram_read_bytes: dram__bytes_read.sum => 0x1 << 0
 	# dram_read_throughput: dram__bytes_read.sum.per_second = 0x1 << 1
	# dram_write_bytes: dram__bytes_write.sum => 0x1 << 2
	# dram_write_throughput: dram__bytes_write.sum.per_second => 0x1 << 3
	counter.setKindFlag(0xF)

	with amanda.tool.apply(counter):
		y = model(x)

if __name__ == "__main__":
	main()