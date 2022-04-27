import sys

from torch import int32
import amanda
import tensorflow as tf
from tensorflow.python.client import timeline
from examples.common.tensorflow.model.resnet_50 import ResNet50

sys.path.append("..")
from amanda_tracer import amandaTracer

def main():

	filePath = "trace_record01.txt"
	tracer = amandaTracer(filePath=filePath)

	# 0x1 << 3 Kernel
	# 0x1 << 5 Runtime
	# 0x1 << 4 Driver
	# 0x1 << 10 Conc Kernel	
	tracer.setKindFlag(0x1 << 3 | 0x1 << 5)

	model = ResNet50()
	x = tf.random.uniform(shape = [1, 224, 224, 3])
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
	# gpu_options = tf.GPUOptions(allow_growth=True)

	# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=0, inter_op_parallelism_threads=1)) as session:
	config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=0, inter_op_parallelism_threads=1)
	session = tf.Session(config=config)

	# runOptions = tf.compat.v1.RunOptions(inter_op_thread_pool=-1, trace_level=tf.RunOptions.SOFTWARE_TRACE)
	with amanda.tool.apply(tracer):
		y = model(x)
		session.run(tf.initialize_all_variables())
		session.run(y)

	data_api = tracer.getTraceDataApi()
	timeLists_start = tracer.getStartTimeLists()
	numTotalDisorder = 0
	numTotalOrder = 0
	timeIndex = 0
	for x in data_api:
		time = timeLists_start[timeIndex]
		if (x.startTime < time):
			timeIndex = timeIndex + 1
			time = timeLists_start[timeIndex]

		if x.name == "cudaLaunchKernel_v7000":
			if (x.startTime < time):
				numTotalDisorder += 1
				print(x.name, x.startTime, time, timeIndex, time-x.startTime)
			else:
				numTotalOrder += 1


	numTotalZero = 0
	numTotalNotZero = 0
	executionTimeTotal = 0
	data_rt = tracer.getTraceDataRt()
	for x in data_rt:
		if x.durationTime == 0:
			numTotalZero += 1
			print(x.kind, x.name, x.startTime, x.duratiuonTime)
		else:
			numTotalNotZero += 1
			executionTimeTotal += x.durationTime	
				
	print("Len of timeList:", len(timeLists_start), "opCount:", tracer.opCount)
	print("Total zero-time kernel:", numTotalZero)
	print("Total notzero-time kernel:", numTotalNotZero)
	print("Total recording execution time:", executionTimeTotal)
	print("Total disorder API:", numTotalDisorder)
	print("Total order API:", numTotalOrder)
	print("Before-op Count: ", tracer.beforeCount)
	print("After-op Count: ", tracer.afterCount)


def model():
	model = ResNet50()
	x = tf.random.uniform(shape = [1, 224, 224, 3])
	with tf.Session() as session:
		options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		run_metadata = tf.RunMetadata()
		y = model(x)
		session.run(tf.initialize_all_variables())
		session.run(y, options=options, run_metadata=run_metadata)
		time_line = timeline.Timeline(run_metadata.step_stats)
		trace = time_line.generate_chrome_trace_format()
		with open('timeline_full.json', 'w') as f:
			f.write(trace)

if __name__ == "__main__":
	main()
	# model()