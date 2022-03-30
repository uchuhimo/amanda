#include "counter.h"

/**
 * @brief Set metrics according to the flag.
 * One bit in flag correspond to one event/metric cupti can provide. 
 * 
 * @param flag
 * roofline analysis:
 * 	default: all => 0
 * 	dram_read_bytes: dram__bytes_read.sum => 0x1 << 0
 * 	dram_read_throughput: dram__bytes_read.sum.per_second = 0x1 << 1
 *  dram_write_bytes: dram__bytes_write.sum => 0x1 << 2
 *  dram_write_throughput: dram__bytes_write.sum.per_second => 0x1 << 3
 * 
 * 	flop_count_dp: 	smsp__sass_thread_inst_executed_op_dadd_pred_on.sum + 
 * 					smsp__sass_thread_inst_executed_op_dmul_pred_on.sum + 
 * 					smsp__sass_thread_inst_executed_op_dfma_pred_on.sum * 2
 * 	flop_count_dp_add: smsp__sass_thread_inst_executed_op_dadd_pred_on.sum => 0x1 << 4
 *  flop_count_dp_fma: smsp__sass_thread_inst_executed_op_dfma_pred_on.sum => 0x1 << 5
 * 	flop_count_dp_mul: smsp__sass_thread_inst_executed_op_dmul_pred_on.sum => 0x1 << 6
 * 
 *  flop_count_hp: 	smsp__sass_thread_inst_executed_op_hadd_pred_on.sum + 
 * 					smsp__sass_thread_inst_executed_op_hmul_pred_on.sum + 
 * 					smsp__sass_thread_inst_executed_op_hfma_pred_on.sum * 2
 * 	flop_count_hp_add: smsp__sass_thread_inst_executed_op_hadd_pred_on.sum => 0x1 << 7
 *  flop_count_hp_fma: smsp__sass_thread_inst_executed_op_hfma_pred_on.sum => 0x1 << 8
 * 	flop_count_hp_mul: smsp__sass_thread_inst_executed_op_hmul_pred_on.sum => 0x1 << 9
 * 
 *  flop_count_sp: 	smsp__sass_thread_inst_executed_op_fadd_pred_on.sum + 
 * 					smsp__sass_thread_inst_executed_op_fmul_pred_on.sum + 
 * 					smsp__sass_thread_inst_executed_op_ffma_pred_on.sum * 2
 * 	flop_count_sp_add: smsp__sass_thread_inst_executed_op_fadd_pred_on.sum => 0x1 << 10
 *  flop_count_sp_fma: smsp__sass_thread_inst_executed_op_ffma_pred_on.sum => 0x1 << 11
 * 	flop_count_sp_mul: smsp__sass_thread_inst_executed_op_fmul_pred_on.sum => 0x1 << 12
 * 
 * 	per_second
 * 	smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_second => 0x1 << 13
 *  smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_second => 0x1 << 14
 * 	smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_second => 0x1 << 15
 *  smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_second => 0x1 << 16
 *  smsp__sass_thread_inst_executed_op_hfma_pred_on.sum.per_second => 0x1 << 17
 * 	smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_second => 0x1 << 18
 *  smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_second => 0x1 << 19
 *  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_second => 0x1 << 20
 * 	smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_second => 0x1 << 21
 *  
 * per_cycle_elapsed
 *  smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed => 0x1 << 22
 *  smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed => 0x1 << 23
 * 	smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed => 0x1 << 24
 *  smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_elapsed => 0x1 << 25
 *  smsp__sass_thread_inst_executed_op_hfma_pred_on.sum.per_cycle_elapsed => 0x1 << 26
 * 	smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_elapsed => 0x1 << 27
 *  smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed => 0x1 << 28
 *  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed => 0x1 << 29
 * 	smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed => 0x1 << 30
 * 
 * per_cycle_active
 *  smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_active => 0x1 << 31
 *  smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_active => 0x1 << 32
 * 	smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_active => 0x1 << 33
 *  smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_active => 0x1 << 34
 *  smsp__sass_thread_inst_executed_op_hfma_pred_on.sum.per_cycle_active => 0x1 << 35
 * 	smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_active => 0x1 << 36
 *  smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_active => 0x1 << 37
 *  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_active => 0x1 << 38
 * 	smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_active => 0x1 << 39  
 * 
 */

static std::vector<std::string> allMetrics;
static void initAllMetrics() {
	allMetrics.push_back("dram__bytes_read.sum");
	allMetrics.push_back("dram__bytes_read.sum.per_second");
	allMetrics.push_back("dram__bytes_write.sum");
	allMetrics.push_back("dram__bytes_write.sum.per_second");

	allMetrics.push_back("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum");

	allMetrics.push_back("smsp__sass_thread_inst_executed_op_hadd_pred_on.sum");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_hmul_pred_on.sum");

	allMetrics.push_back("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum");

	allMetrics.push_back("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_second");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_second");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_second");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_second");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum.per_second");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_second");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_second");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_second");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_second");

	allMetrics.push_back("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_elapsed");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum.per_cycle_elapsed");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_elapsed");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed");

	allMetrics.push_back("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_active");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_active");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_active");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_active");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum.per_cycle_active");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_active");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_active");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_active");
	allMetrics.push_back("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_active");
} 

void counter::setMetrics(unsigned long flag) {
	if (allMetrics.empty()) {
		initAllMetrics();
	}

	if (flag == 0) {
		this->controler.metricNames = allMetrics;
		return;
	}

	int len = allMetrics.size();
	this->controler.metricNames.clear();
	for (int i = 0; i < allMetrics.size(); ++i) {
		if (flag & (0x1 << i)) {
			this->controler.metricNames.push_back(allMetrics[i]);
		}
	}
	return;
}