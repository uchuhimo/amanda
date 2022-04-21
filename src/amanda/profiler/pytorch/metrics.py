import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

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
		dramRead = dataList[kernelPosition].gpuValue / 1e6
		dramWrite = dataList[kernelPosition+1].gpuValue / 1e6
		spAdd = dataList[kernelPosition+2].gpuValue
		spFma = dataList[kernelPosition+3].gpuValue
		spMul = dataList[kernelPosition+4].gpuValue
		spOp = (spAdd + spMul + spFma * 2) / 1e6
		cyclesElapsed = dataList[kernelPosition+5].gpuValue / 1e6

		ansList[i].append(round(dramRead,2))
		ansList[i].append(round(dramWrite,2))
		ansList[i].append(round(spOp,2))
		ansList[i].append(round(cyclesElapsed,2))

	res = pd.DataFrame(ansList)
	res.columns = ['OpIndex', 'OpName', 'KernelIndex', 'KernelType', 'KernelName', 'LaunchKernelTime(ns)', 'KernelExecutionTime(ns)', 'DramRead(MB)',
					'DramWrite(MB)', 'SpOps(M)', 'CyclesElapsed(M)']
	print(res)
	res.to_csv("./Experiments/kernelInfo_result.csv", index=False, sep=',')

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