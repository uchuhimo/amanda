from amanda_tracer import amandaTracer
from amanda_counter import amandaCounter

from utils import setConfigsMetric
from pytorch.metrics import kernelInfo

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

	def setConfigs(self, metric, onlineOnly=False, offlineOnly=False):
		self.__metric = metric
		self.tracer.clearData()
		self.counter.clearData()

		setConfigsMetric(metric, self.tracer, self.counter)

		if onlineOnly == True:
			self.tracer.onlineAnalysisOnly()
			self.counter.onlineAnalysisOnly()

		if offlineOnly == True:
			self.tracer.offlineAnalysisOnly()
			self.counter.offlineAnalysisOnly()

	def showResults(self):
		if self.__metric == "KernelInfo":
			self.opList = self.tracer.opList
			self.startTimeList = self.tracer.getStartTimeLists()
			self.traceDataRt = self.tracer.getTraceDataRt()
			self.traceDataApi = self.tracer.getTraceDataApi()
			self.countData = self.counter.getCountData()
			kernelInfo(self.opList, self.startTimeList, self.traceDataApi, self.traceDataRt, self.countData)
			return
		


