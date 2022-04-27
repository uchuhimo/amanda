from sklearn.preprocessing import KernelCenterer
import amanda
import torch
import sys
import torchvision

sys.path.append("..")
from amanda_tracer import amandaTracer

def main():

	device = "cuda"

	model = torchvision.models.resnet50().to(device)
	x = torch.rand((32, 3, 227, 227)).to(device)

	filePath = "runtime_record_stampEnd.txt"
	tracer = amandaTracer(filePath=filePath)
	tracer.onlineAnalysisOnly()
	# tracer.setDataType(0x3)

	# 0x1 << 3 Kernel
	# 0x1 << 10 Conc Kernel
	# 0x1 << 5 Runtime
	# 0x1 << 4 Driver
	tracer.setKindFlag(0x1 << 5 | 0x1 << 10)

	with amanda.tool.apply(tracer):
		y = model(x)

	data_api = tracer.getTraceDataApi()
	timeLists_start = tracer.getStartTimeLists()
	numTotalDisorder = 0
	numTotalOrder = 0
	print("Len of timeList:", len(timeLists_start), "opCount:", tracer.opCount)
	timeIndex = 1
	for x in data_api:
		time = timeLists_start[timeIndex - 1]
		if (timeIndex < len(timeLists_start) and x.startTime > timeLists_start[timeIndex]):
			timeIndex = timeIndex + 1
			time = timeLists_start[timeIndex - 1]

		if x.name == "cudaLaunchKernel_v7000":
			if (x.startTime < time):
				numTotalDisorder += 1
				print(x.name, x.startTime, time, timeIndex, time-x.startTime)
			else:
				numTotalOrder += 1


	numTotalZero = 0
	numTotalNotZero = 0
	executionTimeTotal = 0
	data_rt = tracer.getTraceDataRt()
	for x in data_rt:
		if x.durationTime == 0:
			numTotalZero += 1
			print(x.kind, x.name, x.startTime, x.duratiuonTime)
		else:
			numTotalNotZero += 1
			executionTimeTotal += x.durationTime	
		
		
	print("Total zero-time kernel:", numTotalZero)
	print("Total notzero-time kernel:", numTotalNotZero)
	print("Total recording execution time:", executionTimeTotal)
	print("Total disorder API:", numTotalDisorder)
	print("Total order API:", numTotalOrder)

if __name__ == "__main__":
	main()