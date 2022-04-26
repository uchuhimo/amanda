import pandas as pd
from metrics import kernelInfoTracer, kernelInfoCounter


# Now Information: OpIndex, OpName, KernelIndex, KernelName, cudaLaunchKernelDuration, kernelExecutionDuration, dramRead, dramWrite, SFOp, elapsedCycles
def kernelInfo(opList, timeList, apiList, rtList, dataList):

	ansListTracer = kernelInfoTracer(opList, timeList, apiList, rtList)
	opPosition, opKernelNum = kernelInfoCounter(dataList)

	k = min(20, len(ansListTracer))
		
	# Mapping Tracer data and Counter data, add counter data to TOP-K tracer ansList
	for i in range(k):
		if opKernelNum[ansListTracer[i][0]] == 0:
			ansListTracer[i].append(-1)
			ansListTracer[i].append(-1)
			ansListTracer[i].append(-1)
			ansListTracer[i].append(-1)
			continue

		# kernelPosition = opPosition[opIndex] + matricsNum * kernelIndex + 1
		kernelPosition = opPosition[ansListTracer[i][0]] + 6 * ansListTracer[i][2] + 1
		dramRead = dataList[kernelPosition].gpuValue / 1e6
		dramWrite = dataList[kernelPosition+1].gpuValue / 1e6
		spAdd = dataList[kernelPosition+2].gpuValue
		spFma = dataList[kernelPosition+3].gpuValue
		spMul = dataList[kernelPosition+4].gpuValue
		spOp = (spAdd + spMul + spFma * 2) / 1e6
		cyclesElapsed = dataList[kernelPosition+5].gpuValue / 1e6

		ansListTracer[i].append(round(dramRead,2))
		ansListTracer[i].append(round(dramWrite,2))
		ansListTracer[i].append(round(spOp,2))
		ansListTracer[i].append(round(cyclesElapsed,2))

	res = pd.DataFrame(ansListTracer)
	res.columns = ['OpIndex', 'OpName', 'KernelIndex', 'KernelType', 'KernelName', 'LaunchKernelTime(ns)', 'KernelExecutionTime(ns)', 'DramRead(MB)',
					'DramWrite(MB)', 'SpOps(M)', 'CyclesElapsed(M)']
	print(res)
	res.to_csv("./Experiments/kernelInfo_result.csv", index=False, sep=',')
