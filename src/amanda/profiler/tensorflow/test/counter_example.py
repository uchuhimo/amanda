import os
import sys
import amanda
import tensorflow as tf
from tensorflow.python.client import timeline
from examples.common.tensorflow.model.resnet_50 import ResNet50

sys.path.append("..")
from amanda_counter import amandaCounter

def main():

	filePath = "kernel_metrics.txt"
	counter = amandaCounter(filePath=filePath)

	# dram_read_bytes: dram__bytes_read.sum => 0x1 << 0
 	# dram_read_throughput: dram__bytes_read.sum.per_second = 0x1 << 1
	# dram_write_bytes: dram__bytes_write.sum => 0x1 << 2
	# dram_write_throughput: dram__bytes_write.sum.per_second => 0x1 << 3
	counter.setKindFlag(0x5 | 0x1 << 42)

	model = ResNet50()
	x = tf.random.uniform(shape = [1, 224, 224, 3])

	experimental = tf.GPUOptions.Experimental(timestamped_allocator=True)	
	# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
	gpu_options = tf.GPUOptions(allow_growth=True, experimental=experimental)
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=0, inter_op_parallelism_threads=1)) as session:
		with amanda.tool.apply(counter):
			y = model(x)
			session.run(tf.initialize_all_variables())
			session.run(y)

if __name__ == "__main__":
	main()