import sys

from amanda_tracer import amandaTracer
from amanda_counter import amandaCounter

from utils import setConfigsMetric
from metrics import kernelRoofline, opRoofline
from torchMetrics import kernelInfo, opInfo

class amandaProfiler():
	def __init__(self, metric) -> None:
		self.__metric = metric

		self.tracer = amandaTracer()
		self.counter = amandaCounter()
		
		self.opList = []
		self.startTimeList = []
		self.endTimeList = []
		self.traceDataApi = []
		self.traceDataRt = []
		self.traceDataOh = []
		self.countData = []

		self.supplyInfo = []

	def setConfigs(self, metric, supplyInfo, onlineOnly=False, offlineOnly=False):
		self.__metric = metric
		self.tracer.activityFlushAll()
		self.tracer.clearData()
		self.counter.clearData()

		self.supplyInfo = supplyInfo
		setConfigsMetric(metric, self.tracer, self.counter)

		if onlineOnly == True:
			self.tracer.onlineAnalysisOnly()
			self.counter.onlineAnalysisOnly()

		if offlineOnly == True:
			self.tracer.offlineAnalysisOnly()
			self.counter.offlineAnalysisOnly()

	def showResults(self):
		if self.__metric == "KernelInfo":
			self.tracer.activityFlushAll()
			self.opList = self.tracer.opList
			self.startTimeList = self.tracer.getStartTimeLists()
			self.traceDataRt = self.tracer.getTraceDataRt()
			self.traceDataApi = self.tracer.getTraceDataApi()
			self.countData = self.counter.getCountData()
			kernelInfo(self.opList, self.startTimeList, self.traceDataApi, self.traceDataRt, self.countData)
			return

		if self.__metric == "KernelRoofline":
			self.countData = self.counter.getCountData()
			assert len(self.supplyInfo) == 3, "Please provide correct hardware parameters"
			kernelRoofline(self.supplyInfo, self.countData)
			return
		
		if self.__metric == "OpInfo":
			self.tracer.activityFlushAll()
			self.opList = self.tracer.opList
			self.startTimeList = self.tracer.getStartTimeLists()
			self.endTimeList = self.tracer.getEndTimeLists()
			self.traceDataApi = self.tracer.getTraceDataApi()
			self.traceDataRt = self.tracer.getTraceDataRt()
			self.countData = self.counter.getCountData()
			opInfo(self.opList, self.startTimeList, self.endTimeList, self.traceDataApi, self.traceDataRt, self.countData)
			return

		if self.__metric == "OpRoofline":
			self.countData = self.counter.getCountData()
			assert len(self.supplyInfo) == 3, "Please provide correct hardware parameters"
			opRoofline(self.supplyInfo, self.countData)
			return

		sys.exit("Profiler.Metric: " + self.__metric + " not supported")
		


