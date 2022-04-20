import heapq


def findTopKKernel(infoList, k):
	heap = []
	for i in range(k):
		heapq.heappush(heap, (infoList[i][6], infoList[i]))
	for i in range(k, len(infoList)):
		heapq.heappushpop(heap, (infoList[i][6], infoList[i]))
	
	ansList = []
	for i in range(k):
		item = heapq.heappop(heap)
		ansList.append(item[1])
	ansList.reverse()
	return ansList


# Now Information: OpIndex, OpName, KernelIndex, KernelName, cudaLaunchKernelDuration, kernelExecutionDuration, dramRead, dramWrite, SFOp, elapsedCycles
def kernelInfo(opList, timeList, apiList, rtList, dataList):

	# Filter apiList and rtList, only reserve cudaLaunchKernel and CONC KERNEL / KERNEL
	launchKernelApiList = []
	kernelList = []
	for x in apiList:
		if x.name == "cudaLaunchKernel_v7000":
			launchKernelApiList.append(x)
	for x in rtList:
		if x.kind == "KERNEL" or x.kind == "CONC KERNEL":
			kernelList.append(x)

	# Match kernel and op
	infoList = []
	opIndex = 0
	kernelIndex = 0
	timeListLen = len(timeList)
	for i in range(len(launchKernelApiList)):
		record = []
		if (opIndex < timeListLen - 1 and launchKernelApiList[i].startTime > timeList[opIndex + 1]):
			opIndex += 1
			kernelIndex = 0
		
		record.append(opIndex)
		record.append(opList[opIndex])

		record.append(kernelIndex)
		record.append(kernelList[i].kind)
		record.append(kernelList[i].name)
		record.append(launchKernelApiList[i].durationTime)
		record.append(kernelList[i].durationTime)

		infoList.append(record)
		kernelIndex += 1

	# Find Top-K kernel according to kernel execution time
	k = min(10, len(kernelList))
	ansList = findTopKKernel(infoList, k)
	
	# Add Counter Data
	opPosition=[]
	for i in range(len(dataList)):
		if dataList[i].rangeName == "NEW OP":
			opPosition.append(i)
			
	for i in range(k):
		# kernelPosition = opPosition[opIndex] + matricsNum * kernelIndex + 1
		kernelPosition = opPosition[ansList[i][0]] + 6 * ansList[i][2] + 1
		dramRead = dataList[kernelPosition].gpuValue
		dramWrite = dataList[kernelPosition+1].gpuValue
		spAdd = dataList[kernelPosition+2].gpuValue
		spMul = dataList[kernelPosition+3].gpuValue
		spFma = dataList[kernelPosition+4].gpuValue
		spOp = spAdd + spMul + spFma * 2
		cyclesElapsed = dataList[kernelPosition+5].gpuValue

		ansList[i].append(dramRead)
		ansList[i].append(dramWrite)
		ansList[i].append(spOp)
		ansList[i].append(cyclesElapsed)
		print(ansList)