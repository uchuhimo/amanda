import sys

import amanda

sys.path.append('../')
from counter import counter

class amandaCounter(amanda.Tool):
	def __init__(self, filePath="kernel_metrics.txt", kindFlag=0) -> None:
		super().__init__(namespace="pytorch")
		self.add_inst_for_op(self.forward_instrumentation)
		self.counter = counter.counter(kindFlag, filePath)
		self.opCount = 0
		self.opList = []

	def forward_instrumentation(self, context: amanda.OpContext):
		op = context.get_op()
		self.opCount += 1

		# if self.opCount > 10:
		# 	return

		self.opList.append(op.__name__)
		context.insert_before_op(
			self.start_profiling,
			opName = op.__name__
		)
		context.insert_after_op(
			self.stop_profiling,
		)

	def start_profiling(self, *inputs, opName):
		self.counter.startProfilingKernel(opName)

	def stop_profiling(self, *outputs):
		self.counter.stopProfiling()

	def setKindFlag(self, kindFlag):
		self.counter.setKindFlag(kindFlag)

	def getKindFlag(self):
		return self.counter.getKindFlag()

	def setFilePath(self, filePath):
		self.counter.setFilePath(filePath)

	def getFilePath(self):
		return self.counter.getFilePath()
		
	def onlineAnalysisOnly(self):
		self.counter.onlineAnalysisOnly()

	def offlineAnalysisOnly(self):
		self.counter.offlineAnalysisOnly()

	def setCountDevice(self, deviceNum):
		self.counter.setCountDevice(deviceNum)

	def setCountParams(self, deviceNum, metricNames):
		self.counter.setCountParams(deviceNum, metricNames)

	def getCountData(self):
		return self.counter.countData

	def clearData(self):
		self.opCount = 0
		self.opList.clear()
		self.counter.clearData()
