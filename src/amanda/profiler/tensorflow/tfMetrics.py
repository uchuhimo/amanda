import pandas as pd
from metrics import kernelInfoTracer, kernelInfoCounter, opInfoTracer, opInfoCounter

# Now Information: kernelInfoTracer, kernelInfoCounter, and Mapping
# ['OpIndex', 'OpName', 'KernelIndex', 'KernelType', 'KernelName', 'LaunchKernelTime(ns)', 'KernelExecutionTime(ns)', 'KernelNum',
#					'DramRead(MB)', 'DramWrite(MB)', 'CyclesElapsed(M)']
def kernelInfo(opListTracer, opListCounter, timeList, apiList, rtList, dataList):
	
	ansListTracer = kernelInfoTracer(opListTracer, timeList, apiList, rtList)
	opPosition, opKernelNum = kernelInfoCounter(dataList, flopCount=False)

	# We can not fully mapping for some ops' kernels. For example, Conv2D.
	# Maybe the reason is cudnn will do some optimizations at the underlying layer.
	# At this point, the results are not entirely plausible.
	def findOp(opName, opListCounter):
		for i in range(len(opListCounter)):
			if opListCounter[i] == opName:
				return i

	k = min(20, len(ansListTracer))

	for i in range(k):
		opName = ansListTracer[i][1]
		opIndex = findOp(opName=opName, opListCounter=opListCounter)
		
		if opKernelNum[opIndex] == 0:
			ansListTracer[i].append(-1)
			ansListTracer[i].append(-1)
			ansListTracer[i].append(-1)
			continue

		kernelPosition = opPosition[opIndex] + 3 * ansListTracer[i][2] + 1
		dramRead = dataList[kernelPosition].gpuValue / 1e6
		dramWrite = dataList[kernelPosition+1].gpuValue / 1e6
		cyclesElapsed = dataList[kernelPosition+2].gpuValue / 1e6

		ansListTracer[i].append(opKernelNum[opIndex])
		ansListTracer[i].append(round(dramRead, 2))
		ansListTracer[i].append(round(dramWrite,2))
		ansListTracer[i].append(round(cyclesElapsed, 2))

	res = pd.DataFrame(ansListTracer)
	res.columns = ['OpIndex', 'OpName', 'KernelIndex', 'KernelType', 'KernelName', 'LaunchKernelTime(ns)', 'KernelExecutionTime(ns)', 'KernelNum',
					'DramRead(MB)', 'DramWrite(MB)', 'CyclesElapsed(M)']
	print(res)
	res.to_csv("./Experiments/kernelInfo_result.csv", index=False, sep=',')




# Now information: opInfoTracer, opInfoCounter, and Mapping
# ['OpIndex', 'OpName', 'opExecutionTime', 'MaxKernelIndex', 'MaxKernelName', 'MaxKernelExecutionTime(ns)', 'KernelNumTracer',
#					'KernelNumCounter', 'TotalCyclesElapsed', 'TotalDramRead', 'TotalDramWrite']
def opInfo(opListTracer, opListCounter, startTimeList, endTimeList, apiList, rtList, dataList):

	ansListTracer = opInfoTracer(opListTracer, startTimeList, endTimeList, apiList, rtList)
	infoListCounter = opInfoCounter(dataList, flopCount=False)

	def findOp(opName, infoListCounter):
		for i in range(len(infoListCounter)):
			if infoListCounter[i][1] == opName:
				return i
	
	# Mapping by the op name
	for i in range(len(ansListTracer)):
		opName = ansListTracer[i][1]
		index = findOp(opName, infoListCounter)
		ansListTracer[i] += infoListCounter[index][2:]

	res = pd.DataFrame(ansListTracer)
	res.columns = ['OpIndex', 'OpName', 'opExecutionTime', 'MaxKernelIndex', 'MaxKernelName', 'MaxKernelExecutionTime(ns)', 'KernelNumTracer',
					'KernelNumCounter', 'TotalCyclesElapsed', 'TotalDramRead', 'TotalDramWrite']
	print(res)
	res.to_csv("./Experiments/opInfo_result.csv", index=False, sep=',')
