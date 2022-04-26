import sys

import amanda

sys.path.append('../')
from tracer import tracer

class amandaTracer(amanda.Tool):
	def __init__(self, filePath="activity_records.txt", kindFlag=0) -> None:
		super().__init__(namespace="pytorch")
		self.add_inst_for_op(self.forward_instrumentation)
		self.opCount = 0
		self.opList=[]
		self.tracer = tracer.tracer(kindFlag, filePath)


	def forward_instrumentation(self, context: amanda.OpContext):
		op = context.get_op()
		self.opCount += 1

		# if self.opCount > 10:
		# 	return

		self.opList.append(op.__name__)		
		context.insert_before_op(
			self.init_trace,
		)
		context.insert_after_op(
			self.finish_trace,
		)

	def init_trace(self, *input):
		self.tracer.initTrace()

	def finish_trace(self, *output):
		self.tracer.activityFlushAll()
		self.tracer.finishTrace()

	def setKindFlag(self, kindFlag):
		self.tracer.setKindFlag(kindFlag)

	def getKindFlag(self):
		return self.tracer.getKindFlag()

	def setFilePath(self, filePath):
		self.tracer.setFilePath(filePath)

	def getFilePath(self):
		return self.tracer.getFilePath()
	
	def setDataTypeFlag(self, dataTypeFlag):
		self.tracer.setDataTypeFlag(dataTypeFlag)

	def getDataTypeFlag(self):
		return self.tracer.getDataTypeFlag()

	def onlineAnalysisOnly(self):
		self.tracer.onlineAnalysisOnly()

	def offlineAnalysisOnly(self):
		self.tracer.offlineAnalysisOnly()

	def activityFlushAll(self):
		self.tracer.activityFlushAll()

	def clearData(self):
		self.opCount = 0
		self.opList.clear()
		self.tracer.clearData()

	def getTraceDataRt(self):
		return self.tracer.traceData_rt

	def getTraceDataApi(self):
		return self.tracer.traceData_api
	
	def getTraceDataOh(self):
		return self.tracer.traceData_oh
	
	def getStartTimeLists(self):
		return self.tracer.startTimeLists

	def getEndTimeLists(self):
		return self.tracer.endTimeLists