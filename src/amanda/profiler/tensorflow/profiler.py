import sys
import tensorflow as tf

from amanda_tracer import amandaTracer
from amanda_counter import amandaCounter

from utils import setConfigsMetric
from tfMetrics import kernelInfo, opInfo

class Profiler():
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

		config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
		session = tf.Session(config=config)
		return session
	
	def createSessionCounter(self):
		gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

		config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
		session = tf.Session(config=config)
		return session

	def showResults(self):
		if self.__metric == "KernelInfo":
			self.opListTracer = self.tracer.opList
			self.opListCounter = self.counter.opList
			self.startTimeList = self.tracer.getStartTimeLists()
			self.traceDataRt = self.tracer.getTraceDataRt()
			self.traceDataApi = self.tracer.getTraceDataApi()
			self.countData = self.counter.getCountData()
			kernelInfo(self.opListTracer, self.opListCounter, self.startTimeList, self.traceDataApi, self.traceDataRt, self.countData)
			return

		if self.__metric == "OpInfo":
			self.opListTracer = self.tracer.opList
			self.opListCounter = self.counter.opList
			self.startTimeList = self.tracer.getStartTimeLists()
			self.endTimeList = self.tracer.getEndTimeLists()
			self.traceDataRt = self.tracer.getTraceDataRt()
			self.traceDataApi = self.tracer.getTraceDataApi()
			self.countData = self.counter.getCountData()
			opInfo(self.opListTracer, self.opListCounter, self.startTimeList, self.endTimeList, self.traceDataApi, self.traceDataRt, self.countData)
			return

		sys.exit("Profiler.Metric: " + self.__metric + " not supported")
		