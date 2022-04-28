import sys
import tensorflow as tf

from amanda_tracer import amandaTracer
from amanda_counter import amandaCounter

from utils import setConfigsMetric
from tfMetrics import kernelInfo, opInfo
from metrics import opInfoTracer, opInfoCounter, kernelInfoTracer, kernelInfoCounter

class amandaProfiler():
	def __init__(self, metric) -> None:
		self.__metric = metric

		self.tracer = amandaTracer()
		self.counter = amandaCounter()
		
		self.opListTracer = []
		self.opListCounter = []

		self.startTimeList = []
		self.endTimeList = []
		self.traceDataApi = []
		self.traceDataRt = []
		self.traceDataOh = []
		self.countData = []

		self.supplyInfo = []

	def setConfigs(self, metric, supplyInfo, onlineOnly=False, offlineOnly=False):
		self.__metric = metric
		self.tracer.clearData()
		self.counter.clearData()

		self.supplyInfo = supplyInfo
		setConfigsMetric(metric, self.tracer, self.counter, flopCount=False)

		if onlineOnly == True:
			self.tracer.onlineAnalysisOnly()
			self.counter.onlineAnalysisOnly()

		if offlineOnly == True:
			self.tracer.offlineAnalysisOnly()
			self.counter.offlineAnalysisOnly()
	
	def createSessionTracer(self):
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

		config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=0, inter_op_parallelism_threads=1)
		session = tf.Session(config=config)
		return session
	
	def createSessionCounter(self):
		gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

		config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
		session = tf.Session(config=config)
		return session

	def showResults(self):
		if self.__metric.find("KernelInfo") != -1:
			self.tracer.activityFlushAll()
			self.opListTracer = self.tracer.opList
			self.opListCounter = self.counter.opList
			self.startTimeList = self.tracer.getStartTimeLists()
			self.traceDataRt = self.tracer.getTraceDataRt()
			self.traceDataApi = self.tracer.getTraceDataApi()
			self.countData = self.counter.getCountData()
			if self.__metric == "KernelInfo":
				kernelInfo(self.opListTracer, self.opListCounter, self.startTimeList, self.traceDataApi, self.traceDataRt, self.countData)
			elif self.__metric == "KernelInfoTracer":
				kernelInfoTracer(self.opListTracer, self.startTimeList, self.traceDataApi, self.traceDataRt)
			elif self.__metric == "KernelInfoCounter":
				kernelInfoCounter(self.countData, flopCount=False)
			else:
				sys.exit("Profiler.Metric: " + self.__metric + " not supported")
			return

		if self.__metric.find("OpInfo") != -1:
			self.tracer.activityFlushAll()
			self.opListTracer = self.tracer.opList
			self.opListCounter = self.counter.opList
			self.startTimeList = self.tracer.getStartTimeLists()
			self.endTimeList = self.tracer.getEndTimeLists()
			self.traceDataRt = self.tracer.getTraceDataRt()
			self.traceDataApi = self.tracer.getTraceDataApi()
			self.countData = self.counter.getCountData()
			if self.__metric == "OpInfo":
				opInfo(self.opListTracer, self.opListCounter, self.startTimeList, self.endTimeList, self.traceDataApi, self.traceDataRt, self.countData)
			elif self.__metric == "OpInfoTracer":
				opInfoTracer(self.opListTracer, self.startTimeList, self.endTimeList, self.traceDataApi, self.traceDataRt)
			elif self.__metric == "OpInfoCounter":
				opInfoCounter(self.countData, flopCount=False)
			else:
				sys.exit("Profiler.Metric: " + self.__metric + " not supported")
			return

		sys.exit("Profiler.Metric: " + self.__metric + " not supported")
		