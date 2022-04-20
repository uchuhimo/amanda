import sys

def setConfigsMetric(metric, tracer, counter):
	# set configs for kernel information
	if metric == "KernelInfo":
		# RUNTIME API => 0x1 << 5
		# CONC KERNEL => 0x1 << 10
		tracer.setKindFlag(0x1 << 5 | 0x1 << 10)
		tracer.setFilePath("./Experiments/activity_records.txt")

		# dram_read_bytes: dram__bytes_read.sum => 0x1 << 0
		# dram_write_bytes: dram__bytes_write.sum => 0x1 << 2
		# flop_count_sp: smsp__sass_thread_inst_executed_op_fadd_pred_on.sum + 
		# 			   smsp__sass_thread_inst_executed_op_fmul_pred_on.sum + 
		# 			   smsp__sass_thread_inst_executed_op_ffma_pred_on.sum * 2
		# flop_count_sp_add: smsp__sass_thread_inst_executed_op_fadd_pred_on.sum => 0x1 << 10
 		# flop_count_sp_fma: smsp__sass_thread_inst_executed_op_ffma_pred_on.sum => 0x1 << 11
 		# flop_count_sp_mul: smsp__sass_thread_inst_executed_op_fmul_pred_on.sum => 0x1 << 12
		# elapsed_cycles: sm__cycles_elapsed.sum => 0x1 << 42
		counter.setKindFlag(0xE05 | 0x1 << 42)
		counter.setFilePath("./Experiments/kernel_metrics.txt")
		return
	
	sys.exit("Invalid Matrix")