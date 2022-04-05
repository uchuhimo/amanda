from counter import Counter
from counter import counter

# test: Counter::countData_t read/write
def test_countData_struct():
	countData = Counter.countData_t();
	countData.rangeName = "0"
	countData.metricName = "dram__bytes_read.sum"
	countData.gpuValue = 3.30
	assert countData.rangeName == "0"
	assert countData.metricName == "dram__bytes_read.sum"
	assert countData.gpuValue == 3.30

# test: counter.countData clear/transfer
def test_counter_countData():
	_counter = counter()
	assert not _counter.countData
	_counter.testClearData()
	assert _counter.countData
	assert _counter.countData[0].gpuValue == 3.30
	_counter.clearData()
	assert not _counter.countData

if __name__ == "__main__":
	test_countData_struct()
	test_counter_countData()