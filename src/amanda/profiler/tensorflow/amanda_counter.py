import enum
from operator import index
import sys
import tensorflow as tf

import amanda
sys.path.append('../')
from counter import counter


class amandaCounter(amanda.Tool):
	def __init__(self, filePath="kernel_metrics.txt", kindFlag=0) -> None:
		super().__init__(namespace="tensorflow")
		self.add_inst_for_op(
			self.forward_instrumentation,
			require_outputs=True
		)

		self.opCount = 0
		self.beforeCount = 0
		self.afterCount = 0
		self.counter = counter.counter(kindFlag, filePath)
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

		# if len(op_outputs) != 0 and len(op_inputs) != 0:
		if len(op_outputs) != 0 and len(op_inputs) != 0 and op.name.find("conv2d/Conv2D") != -1:	
			self.opList.append(op.name)
			context.insert_before_op(
				self.start_profile,
				inputs=op_inputs,
				op=op,
			)

			context.insert_after_op(
				self.stop_profile,
				outputs=op_outputs,
				op=op,
			)

	def start_profile(self, *inputs, op):
		def extract_fn(*inputs):
			self.counter.startProfilingKernel(op.name)
			self.beforeCount += 1
			return inputs
		
		new_inputs = tf.py_function(
			extract_fn,
			inputs,
			[tensor.dtype for tensor in inputs],
			name="before_" + op.name,
		)
		return new_inputs

	def stop_profile(self, *outputs, op):	
		def extract_fn(*outputs):
			self.counter.stopProfiling()
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
		self.counter.clearData()