import amanda
import tensorflow as tf
from tensorflow.python.client import timeline

from amanda_counter import amandaCounter
from examples.common.tensorflow.model.resnet_50 import ResNet50

def main():

	filePath = "kernel_metrics_11.txt"
	counter = amandaCounter(filePath=filePath)

	# dram_read_bytes: dram__bytes_read.sum => 0x1 << 0
 	# dram_read_throughput: dram__bytes_read.sum.per_second = 0x1 << 1
	# dram_write_bytes: dram__bytes_write.sum => 0x1 << 2
	# dram_write_throughput: dram__bytes_write.sum.per_second => 0x1 << 3
	counter.setKindFlag(0xF)

	model = ResNet50()
	x = tf.random.uniform(shape = [1, 224, 224, 3])
	# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
	gpu_options = tf.GPUOptions(allow_growth=True)
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)) as session:
		# runOptions = tf.compat.v1.RunOptions(inter_op_thread_pool=-1, trace_level=tf.RunOptions.SOFTWARE_TRACE)
		with amanda.tool.apply(counter):
			y = model(x)
			session.run(tf.initialize_all_variables())
			session.run(y)

if __name__ == "__main__":
	main()