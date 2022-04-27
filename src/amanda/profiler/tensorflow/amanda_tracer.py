import enum
from operator import index
import sys
import tensorflow as tf

import amanda
sys.path.append('../')
from tracer import tracer


class amandaTracer(amanda.Tool):
	def __init__(self, filePath="activity_records.txt", kindFlag=0) -> None:
		super().__init__(namespace="tensorflow")
		self.add_inst_for_op(
			self.forward_instrumentation,
			require_outputs=True
		)
		self.opCount = 0
		self.beforeCount = 0
		self.afterCount = 0
		self.tracer = tracer.tracer(kindFlag, filePath)
		self.opList = []

	def forward_instrumentation(self, context: amanda.OpContext):
		op = context.get_op()
		self.opCount += 1
		if op.type in ["Identify", "Const"]:
			return

		op_inputs=[
				index
				for index, tensor in enumerate(context.get_inputs())
				if not tensor.dtype._is_ref_dtype
			]
		op_outputs=[
				index
				for index, tensor in enumerate(context.get_outputs())
				if not tensor.dtype._is_ref_dtype
			]

		if len(op_outputs) != 0 and len(op_inputs) != 0:
		# if len(op_outputs) != 0 and len(op_inputs) != 0 and op.name.find("Relu") != -1:	
			self.opList.append(op.name)
			context.insert_before_op(
				self.init_trace,
				inputs=op_inputs,
				op=op,
			)

			context.insert_after_op(
				self.finish_trace,
				outputs=op_outputs,
				op=op,
			)

	def init_trace(self, *inputs, op):
		def extract_fn(*inputs):
			self.tracer.initTrace()
			self.beforeCount += 1
			return inputs
		
		new_inputs = tf.py_function(
			extract_fn,
			inputs,
			[tensor.dtype for tensor in inputs],
			name="before_" + op.name,
		)
		return new_inputs

	def finish_trace(self, *outputs, op):	
		def extract_fn(*outputs):
			self.tracer.finishTrace()
			self.afterCount += 1
			return outputs
		
		with tf.control_dependencies([op]):
			new_outputs = tf.py_function(
				extract_fn,
				outputs,
				[tensor.dtype for tensor in outputs],
				name="after_" + op.name,
			)
			return new_outputs

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
