import pandas as pd
from metrics import kernelInfoTracer, kernelInfoCounter

# Now Information: kernelInfoTracer, kernelInfoCounter
def kernelInfo(opListTracer, opListCounter, timeList, apiList, rtList, dataList):
	
	ansListTracer = kernelInfoTracer(opListTracer, timeList, apiList, rtList)
	opPosition, opKernelNum = kernelInfoCounter(dataList, flopCount=False)

	# We can not maping now, since the kernel index can not be matched.

	# def findOp(opName, opListCounter):
	# 	for i in range(len(opListCounter)):
	# 		if opListCounter[i] == opName:
	# 			return i

	# k = min(20, len(ansListTracer))

	# for i in range(k):
	# 	opName = ansListTracer[i][1]
	# 	opIndex = findOp(opName=opName, opListCounter=opListCounter)
		
	# 	if opKernelNum[opIndex] == 0:
	# 		ansListTracer[i].append(-1)
	# 		ansListTracer[i].append(-1)
	# 		ansListTracer[i].append(-1)
	# 		continue

	# 	kernelPosition = opPosition[opIndex] + 3 * ansListTracer[i][2] + 1
	# 	print("Op: ", dataList[opPosition[opIndex]].metricName)
	# 	print("kernelPosition: ", kernelPosition)
	# 	print("kernelIndex: ", ansListTracer[i][2])
	# 	print("dramRead: ", dataList[kernelPosition].metricName)
	# 	dramRead = dataList[kernelPosition].gpuValue / 1e6
	# 	dramWrite = dataList[kernelPosition+1].gpuValue / 1e6
	# 	cyclesElapsed = dataList[kernelPosition+2].gpuValue / 1e6

	# 	ansListTracer[i].append(round(dramRead, 2))
	# 	ansListTracer[i].append(round(dramWrite,2))
	# 	ansListTracer[i].append(round(cyclesElapsed, 2))

	# res = pd.DataFrame(ansListTracer)
	# res.columns = ['OpIndex', 'OpName', 'KernelIndex', 'KernelType', 'KernelName', 'LaunchKernelTime(ns)', 'KernelExecutionTime(ns)', 'DramRead(MB)',
	# 				'DramWrite(MB)', 'CyclesElapsed(M)']
	# print(res)
	# res.to_csv("./Experiments/kernelInfo_result.csv", index=False, sep=',')
	
