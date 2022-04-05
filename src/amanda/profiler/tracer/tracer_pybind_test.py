from tracer import Tracer
from tracer import tracer
import os
import sys

# test: Tracer::trace_Mode
def test_trace_Mode_enum():
	off_on = Tracer.OFFLINE_AND_ONLINE
	assert off_on == Tracer.OFFLINE_AND_ONLINE

# test: Tracer::traceData (use traceData as an example)
def test_traceData_struct():
	traceData = Tracer.traceData_rt()
	traceData.startTime = 1
	traceData.deviceId = 0
	traceData.kind = "KERNEL"
	assert traceData.startTime == 1
	assert traceData.deviceId == 0
	assert traceData.kind == "KERNEL"

# test: tracer class
def test_constrctor():
	_tracer = tracer()
	assert _tracer.getKindFlag() == 0
	assert _tracer.getFilePath() == "activity_record.txt"
	assert _tracer.getDataTypeFlag() == 0
	assert not _tracer.traceData_rt
	del _tracer

	_tracer = tracer(0xF)
	assert _tracer.getKindFlag() == 15
	assert not _tracer.traceData_api
	del _tracer

	_tracer = tracer("record.txt")
	assert _tracer.getFilePath() == "record.txt"
	assert not _tracer.traceData_oh
	del _tracer

	_tracer = tracer(0x7, "record.txt")
	_tracer.setDataTypeFlag(0xF)
	assert _tracer.getKindFlag() == 0x7
	assert _tracer.getFilePath() == "record.txt"
	assert _tracer.getDataTypeFlag() == 0x7
	del _tracer

# The vector are read-only now, still some problems...
def test_trace_process():
	_tracer = tracer()
	_tracer.onlineAnalysisOnly()
	_tracer.initTrace()
	_tracer.finishTrace()
	assert not _tracer.traceData_rt
	os.remove("activity_record.txt")
	# traceData = Tracer.traceData_rt()
	# traceData.deviceId = 1
	# _tracer.traceData_rt.push_back(traceData)
	# assert _tracer.traceData_rt
	# assert len(_tracer.traceData_rt) == 1
	# _tracer.traceData_rt.clear()
	# assert not _tracer.traceData_rt

def test_exception():
	_tracer_a = tracer()
	try:
		_tracer_b = tracer()
	except RuntimeError as re:
		print("Runtime Error", re)
		sys.exit()

def test_tracer_class():
	test_constrctor()
	test_trace_process()
	test_exception()


if __name__ == "__main__":
	test_trace_Mode_enum()
	test_traceData_struct()
	test_tracer_class()