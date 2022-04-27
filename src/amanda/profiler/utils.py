import pathlib
import sys
import heapq

def findTopK(infoList, k, key):
	heap = []
	for i in range(k):
		heapq.heappush(heap, (infoList[i][key], infoList[i]))
	for i in range(k, len(infoList)):
		heapq.heappushpop(heap, (infoList[i][key], infoList[i]))
	
	ansList = []
	for i in range(k):
		item = heapq.heappop(heap)
		ansList.append(item[1])
	ansList.reverse()
	return ansList	

def setConfigsMetric(metric, tracer, counter, flopCount=True):

	pathlib.Path("Experiments").mkdir(parents=True, exist_ok=True)

	# set configs for kernel information
	if metric == "KernelInfo":
		# RUNTIME API => 0x1 << 5
		# CONC KERNEL => 0x1 << 10
		tracer.setKindFlag(0x1 << 5 | 0x1 << 3)
		tracer.setFilePath("./Experiments/activity_records.txt")

		# dram_read_bytes: dram__bytes_read.sum => 0x1 << 0
		# dram_write_bytes: dram__bytes_write.sum => 0x1 << 2
		# eg. flop_count_sp: smsp__sass_thread_inst_executed_op_fadd_pred_on.sum + 
		# 			   smsp__sass_thread_inst_executed_op_fmul_pred_on.sum + 
		# 			   smsp__sass_thread_inst_executed_op_ffma_pred_on.sum * 2

		# flop_count_dp_add: smsp__sass_thread_inst_executed_op_dadd_pred_on.sum => 0x1 << 4
		# flop_count_dp_fma: smsp__sass_thread_inst_executed_op_dfma_pred_on.sum => 0x1 << 5
		# flop_count_dp_mul: smsp__sass_thread_inst_executed_op_dmul_pred_on.sum => 0x1 << 6
		# flop_count_hp_add: smsp__sass_thread_inst_executed_op_hadd_pred_on.sum => 0x1 << 7
		# flop_count_hp_fma: smsp__sass_thread_inst_executed_op_hfma_pred_on.sum => 0x1 << 8
		# flop_count_hp_mul: smsp__sass_thread_inst_executed_op_hmul_pred_on.sum => 0x1 << 9
		# flop_count_sp_add: smsp__sass_thread_inst_executed_op_fadd_pred_on.sum => 0x1 << 10
 		# flop_count_sp_fma: smsp__sass_thread_inst_executed_op_ffma_pred_on.sum => 0x1 << 11
 		# flop_count_sp_mul: smsp__sass_thread_inst_executed_op_fmul_pred_on.sum => 0x1 << 12

		# elapsed_cycles: sm__cycles_elapsed.sum => 0x1 << 42
		if flopCount == True:
			counter.setKindFlag(0x1c05 | 0x1 << 42)
		else:
			counter.setKindFlag(0x5 | 0x1 << 42)
		counter.setFilePath("./Experiments/kernel_metrics.txt")
		return
	
	# set configs for kernel roofline analysis
	if metric == "KernelRoofline" and flopCount:
		counter.setKindFlag(0x1C05 | 0x1 << 42)
		counter.setFilePath("./Experiments/kernel_metrics.txt")
		return

	# set configs for op information
	if metric == "OpInfo":
		tracer.setKindFlag(0x1 << 5 | 0x1 << 3)
		tracer.setFilePath("./Experiments/activity_records.txt")
		if flopCount:
			counter.setKindFlag(0x1C05 | 0x1 << 42)
		else:
			counter.setKindFlag(0x5 | 0x42 << 42)
		counter.setFilePath("./Experiments/kernel_metrics.txt")
		return

	# set configs for op roofline analysis
	if metric == "OpRoofline" and flopCount:
		counter.setKindFlag(0x1C05 | 0x1 << 42)
		counter.setFilePath("./Experiments/kernel_metrics.txt")
		return
	
	sys.exit("Invalid Metric")