import sys

import amanda
import tensorflow as tf
from examples.common.tensorflow.model.resnet_50 import ResNet50

sys.path.append("..")
from profiler import Profiler

def main():

	model = ResNet50()
	x = tf.random.uniform(shape = [1, 224, 224, 3])
	y = model(x)

	metric = "KernelInfo"
	profiler = Profiler(metric)
	profiler.setConfigs(metric=metric, supplyInfo=[])

	session_ = profiler.createSessionCounter()
	with amanda.tool.apply(profiler.counter):
		session_.run(tf.initialize_all_variables())
		session_.run(y)
	session_.close()	
	
	session = profiler.createSessionTracer()
	with amanda.tool.apply(profiler.tracer):
		session.run(tf.initialize_all_variables())
		session.run(y)
	session.close()	

	profiler.showResults()

if __name__ == "__main__":
	main()