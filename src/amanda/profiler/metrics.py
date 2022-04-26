import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from utils import findTopK

def drawRoofline(hardwareTFlops, hardwareIntensity, X, Y):
	maxright=0
	for x in X:
		if x > maxright:
			maxright = x
	maxright *= 1.2
	maxtop = 0
	for y in Y:
		if y > maxtop:
			maxtop = y
	maxtop *= 1.35

	# draw basic elements
	def func1(x):
		return (hardwareTFlops/hardwareIntensity) * x
	def func2(x):
		return hardwareTFlops * x / x
	
	fig, ax = plt.subplots()
	x1 = np.linspace(0, hardwareIntensity)
	y1 = func1(x1)
	ax.plot(x1, y1, 'black', linewidth=2)
	x2 = np.linspace(hardwareIntensity, maxright)
	y2 = func2(x2)
	ax.plot(x2, y2, "black", linewidth=2)
	ax.set_ylim(bottom=0, top=maxtop)
	ax.set_xlim(left=0, right=maxright)
	verts1 = [(0,0), *zip(x1, y1), (hardwareIntensity, 0)]
	poly1 = Polygon(verts1, linestyle="--", color="#CAFFBF", alpha=0.75, zorder=1)
	ax.add_patch(poly1)
	verts2 = [(hardwareIntensity, 0), *zip(x2, y2), (maxright, 0)]
	poly2 = Polygon(verts2, linestyle="--", color="#9BF6FF", alpha=0.75, zorder=1)
	ax.add_patch(poly2)
	ax.plot([hardwareIntensity, hardwareIntensity], [0, hardwareTFlops], color='black', linewidth=1, linestyle="--")
	ax.plot([0, hardwareIntensity], [hardwareTFlops, hardwareTFlops], color='black', linewidth=1, linestyle="--")
	ax.text(hardwareIntensity, 0.5, round(hardwareIntensity,2), fontsize=8)
	ax.text(2, hardwareTFlops, round(hardwareTFlops,2), fontsize=8)
	ax.set(xlabel="Arithmetic Intensity(Flop:Byte)", ylabel="Attainable TFlops", title="Kernel Roofline Analysis")

	# draw count data
	for i in range(len(X)):
		ax.scatter(X[i], Y[i], color="black", alpha=1, zorder=2, s=10)

	plt.savefig("./Experiments/kernelRoofline_result.png")



# TOP-20 Tracer Kernel Information, return ansListTracer for mapping
def kernelInfoTracer(opList, timeList, apiList, rtList):
	# Filter apiList and rtList, only reserve cudaLaunchKernel and CONC KERNEL / KERNEL
	launchKernelApiList = []
	kernelList = []
	for x in apiList:
		if x.name == "cudaLaunchKernel_v7000":
			launchKernelApiList.append(x)
	for x in rtList:
		if x.kind == "KERNEL" or x.kind == "CONC KERNEL":
			kernelList.append(x)

	# Tracer match kernel and op
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
	k = min(20, len(kernelList))
	ansListTracer = findTopK(infoList, k, key = 6)
	resTracer = pd.DataFrame(ansListTracer)
	resTracer.columns = ['OpIndex', 'OpName', 'KernelIndex', 'KernelType', 'KernelName', 'LaunchKernelTime(ns)', 'KernelExecutionTime(ns)']
	print("Number of tracer records: " + str(len(infoList)))
	print(resTracer)
	resTracer.to_csv("./Experiments/kernelInfoTracer_result.csv", index=False, sep=',')

	return ansListTracer



# TOP-20 Counter Kernel Information, return opPosition for mapping
def kernelInfoCounter(dataList, flopCount=True):
	# Filter Count Data, aggregate the information
	opPosition = []
	opKernelNum = []
	for i in range(len(dataList)):
		if dataList[i].rangeName == "NEW OP":
			opPosition.append(i)

	numValue = 6
	if flopCount == False:
		numValue = 3
	for i in range(len(opPosition) - 1):
		kernelNum = (opPosition[i+1] - opPosition[i] -1) / numValue
		opKernelNum.append(kernelNum)
	kernelNum = (len(dataList) - opPosition[-1] - 1) / numValue
	opKernelNum.append(kernelNum)

	infoList = []
	for i in range(len(opPosition)):
		for j in range(int(opKernelNum[i])):
			opName = dataList[opPosition[i]].metricName
			kernelPosition = j * numValue + opPosition[i] + 1
			dramRead = dataList[kernelPosition].gpuValue / 1e6
			dramWrite = dataList[kernelPosition+1].gpuValue / 1e6
			if flopCount == True:
				cyclesElapsed = dataList[kernelPosition+5].gpuValue / 1e6		
				spAdd = dataList[kernelPosition+2].gpuValue / 1e6
				spFma = dataList[kernelPosition+3].gpuValue / 1e6
				spMul = dataList[kernelPosition+4].gpuValue / 1e6
				infoList.append([i, opName, j, cyclesElapsed, dramRead, dramWrite, spAdd, spFma, spMul])
			
			else:
				cyclesElapsed = dataList[kernelPosition+2].gpuValue / 1e6
				infoList.append([i, opName, j, cyclesElapsed, dramRead, dramWrite])

	# Find Top-K kernel according to kernel execution time
	k = min(20, len(infoList))
	ansListCounter = findTopK(infoList, k, key = 3)
	resCounter = pd.DataFrame(ansListCounter)
	if flopCount == True:
		resCounter.columns = ['OpIndex', 'OpName', 'KernelIndex', 'cyclesElapsed', 'dramRead', 'dramWrite', 'spAdd', 'spFma', 'spMul']
	else:
		resCounter.columns = ['OpIndex', 'OpName', 'KernelIndex', 'cyclesElapsed', 'dramRead', 'dramWrite']

	print("Number of counter records: " + str(len(infoList)))
	print(resCounter)
	resCounter.to_csv("./Experiments/kernelInfoCounter_result.csv", index=False, sep=',')

	return opPosition, opKernelNum

# Kernel Roofline Analysis:
# supplyInfo: [Frequency(MHz), Peak-Performance(TFlops), Throughput(GB/s)]
def kernelRoofline(supplyInfo, countData):
	hardwareTFlops = supplyInfo[1]
	hardwareIntensity = (hardwareTFlops * 1000) / supplyInfo[2]

	# Filter Op
	dataList = []
	for x in countData:
		if x.rangeName == "NEW OP":
			continue
		dataList.append(x)

	# Calculate CountData
	kernelX=[]
	kernelY=[]
	for index in range(0, len(dataList), 6):
		dramRead = dataList[index].gpuValue
		dramWrite = dataList[index+1].gpuValue
		spAdd = dataList[index+2].gpuValue
		spFma = dataList[index+3].gpuValue
		spMul = dataList[index+4].gpuValue
		spOp = spAdd + spMul + spFma * 2
		cyclesElapsed = dataList[index+5].gpuValue

		time = cyclesElapsed / (supplyInfo[0] * 1024 * 1024)
		tflops = (spOp / time) / (1024 * 1024 * 1024)
		kernelY.append(round(tflops, 2))
		intensity = spOp / (dramRead + dramWrite)
		kernelX.append(intensity)

	drawRoofline(hardwareTFlops, hardwareIntensity, kernelX, kernelY)



# TOP-20 Tracer OP Information [key: executionTime], return ansListTracer for mapping
# Information collected by tracer: OpIndex, OpName, TotalExecutionTime, MaxKernelIndex, MaxKernelName, MaxKernelExecutionTime, kernelNumTracer 
def opInfoTracer(opList, startTimeList, endTimeList, apiList, rtList):
	# Filter cudaLaunchKernel and kernel
	launchKernelApiList = []
	kernelList = []
	for x in apiList:
		if x.name == "cudaLaunchKernel_v7000":
			launchKernelApiList.append(x)
	for x in rtList:
		if x.kind == "KERNEL" or x.kind == "CONC KERNEL":
			kernelList.append(x)

	# Calculate number of kernels for each op
	kernelNumList = []
	kernelCount = 0
	opIndex = 0
	kernelIndex = 0
	timeListLen = len(startTimeList)
	
	while opIndex < timeListLen - 1 and startTimeList[opIndex] < launchKernelApiList[0].startTime and startTimeList[opIndex+1] < launchKernelApiList[0].startTime:
		kernelNumList.append(0)
		opIndex += 1

	while kernelIndex < len(launchKernelApiList):
		if (opIndex < timeListLen - 1 and launchKernelApiList[kernelIndex].startTime > startTimeList[opIndex + 1]):
			kernelNumList.append(kernelCount)
			opIndex += 1
			kernelCount = 0
			continue
		kernelCount += 1
		kernelIndex += 1
	kernelNumList.append(kernelCount)

	while opIndex < timeListLen - 1:
		kernelNumList.append(0)
		opIndex += 1
	
	for i in range(len(opList)):
		print(opList[i])
		print(kernelNumList[i])

	# Get information for each op
	kernelCount = 0
	infoList = []
	for i in range(len(opList)):
		executionTime = endTimeList[i] - startTimeList[i]
		maxExeTime = 0
		maxIndex = 0
		for j in range(kernelCount, kernelCount+kernelNumList[i]):
			if kernelList[j].durationTime > maxExeTime:
				maxExeTime = kernelList[j].durationTime
				maxIndex = j

		if kernelNumList[i] != 0:
			infoList.append([i, opList[i], executionTime, maxIndex-kernelCount, kernelList[maxIndex].name, maxExeTime, kernelNumList[i]])
		else:
			infoList.append([i, opList[i], executionTime, -1, "NONE", 0, 0])
		kernelCount += kernelNumList[i]
	
	# TOP-20, use executionTime as key
	k = min(20, len(infoList))
	ansListTracer = findTopK(infoList, k, 2)
	resTracer = pd.DataFrame(ansListTracer)
	resTracer.columns = ['OpIndex', 'OpName', 'opExecutionTime', 'MaxKernelIndex', 'MaxKernelName', 'MaxKernelExecutionTime(ns)', 'KernelNumTracer']
	print("Number of tracer records: " + str(len(infoList)))
	print(resTracer)
	resTracer.to_csv("./Experiments/opInfoTracer_result.csv", index=False, sep=',')
	
	return ansListTracer




def opInfoCounter(dataList, flopCount=True):
	# Filter Count Data, aggregate the information
	opPosition = []
	opKernelNum = []
	for i in range(len(dataList)):
		if dataList[i].rangeName == "NEW OP":
			opPosition.append(i)

	numValue = 6
	if flopCount == False:
		numValue = 3
	for i in range(len(opPosition) - 1):
		kernelNum = (opPosition[i+1] - opPosition[i] -1) / numValue
		opKernelNum.append(kernelNum)
	kernelNum = (len(dataList) - opPosition[-1] - 1) / numValue
	opKernelNum.append(kernelNum)

	infoList = []
	for i in range(len(opPosition)):
		totalDramRead = 0
		totalDramWrite = 0
		totalCyclesElapsed = 0
		totalFlopCount = 0
		for j in range(int(opKernelNum[i])):
			kernelPosition = j * numValue + opPosition[i] + 1
			totalDramRead += dataList[kernelPosition].gpuValue / 1e6
			totalDramWrite += dataList[kernelPosition+1].gpuValue / 1e6
			if flopCount == True:
				totalCyclesElapsed += dataList[kernelPosition+5].gpuValue / 1e6		
				spAdd = dataList[kernelPosition+2].gpuValue / 1e6
				spFma = dataList[kernelPosition+3].gpuValue / 1e6
				spMul = dataList[kernelPosition+4].gpuValue / 1e6
				totalFlopCount += spAdd + spMul + spFma * 2
			
			else:
				totalCyclesElapsed += dataList[kernelPosition+2].gpuValue / 1e6

		if flopCount:
			infoList.append([i, dataList[opPosition[i]].metricName, int(opKernelNum[i]), round(totalCyclesElapsed,2), round(totalDramRead,2), round(totalDramWrite,2), round(totalFlopCount,2)])
		else:
			infoList.append([i, dataList[opPosition[i]].metricName, int(opKernelNum[i]), round(totalCyclesElapsed,2), round(totalDramRead,2), round(totalDramWrite,2)])	

	# Find Top-K kernel according to kernel execution time
	k = min(20, len(infoList))
	ansListCounter = findTopK(infoList, k, key = 3)
	resCounter = pd.DataFrame(ansListCounter)
	if flopCount == True:
		resCounter.columns = ['OpIndex', 'OpName', 'KernelNumCounter', 'TotalCyclesElapsed', 'TotalDramRead', 'TotalDramWrite', 'TotalFlopCount']
	else:
		resCounter.columns = ['OpIndex', 'OpName', 'KernelNumCounter', 'TotalCyclesElapsed', 'TotalDramRead', 'TotalDramWrite']

	print("Number of counter records: " + str(len(infoList)))
	print(resCounter)
	resCounter.to_csv("./Experiments/opInfoCounter_result.csv", index=False, sep=',')

	return infoList	
