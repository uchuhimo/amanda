// Each activity record kind represents information about a GPU or an activity occurring on a CPU or GPU. 
// Each kind is associated with a activity record structure that holds the information associated with the kind.
// For more information about the activity kind, please refer: 
// https://docs.nvidia.com/cuda/archive/11.0_GA/cupti/Cupti/modules.html#group__CUPTI__ACTIVITY__API_1gefed720d5a60c3e8b286cd386c4913e3

#include <cupti_activity.h>
#include <cupti.h>
#include <iostream>

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

/**
 * The relationship between flag and cupti activity's kind is based on the flag's binary representation.
 * Details:
 * DEFAULT = 0
 * CUPTI_ACTIVITY_KIND_INVALID => 0
 * CUPTI_ACTIVITY_KIND_MEMCPY => 1<<1
 * CUPTI_ACTIVITY_KIND_MEMSET => 1<<2
 * CUPTI_ACTIVITY_KIND_KERNEL => 1<<3
 * CUPTI_ACTIVITY_KIND_DRIVER => 1<<4
 * CUPTI_ACTIVITY_KIND_RUNTIME => 1<<5
 * CUPTI_ACTIVITY_KIND_EVENT => 1<<6
 * CUPTI_ACTIVITY_KIND_METRIC => 1<<7
 * CUPTI_ACTIVITY_KIND_DEVICE => 1<<8
 * CUPTI_ACTIVITY_KIND_CONTEXT => 1<<9
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL => 1<<10
 * CUPTI_ACTIVITY_KIND_NAME => 1<<11
 * CUPTI_ACTIVITY_KIND_MARKER => 1<<12
 * CUPTI_ACTIVITY_KIND_MARKER_DATA => 1<<13
 * CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR => 1<<14
 * CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS => 1<<15
 * CUPTI_ACTIVITY_KIND_BRANCH => 1<<16
 * CUPTI_ACTIVITY_KIND_OVERHEAD => 1<<17
 * CUPTI_ACTIVITY_KIND_CDP_KERNEL => 1<<18
 * CUPTI_ACTIVITY_KIND_PREEMPTION => 1<<19
 * CUPTI_ACTIVITY_KIND_ENVIRONMENT => 1<<20
 * CUPTI_ACTIVITY_KIND_EVENT_INSTANCE => 1<<21
 * CUPTI_ACTIVITY_KIND_MEMCPY2 => 1<<22
 * CUPTI_ACTIVITY_KIND_METRIC_INSTANCE => 1<<23
 * CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION => 1<<24
 * CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER => 1<<25
 * CUPTI_ACTIVITY_KIND_FUNCTION => 1<<26
 * CUPTI_ACTIVITY_KIND_MODULE => 1<<27
 * CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE => 1<<28
 * CUPTI_ACTIVITY_KIND_SHARED_ACCESS => 1<<29
 * CUPTI_ACTIVITY_KIND_PC_SAMPLING => 1<<30
 * CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO => 1<<31
 * CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION => 1<<32
 * CUPTI_ACTIVITY_KIND_OPENACC_DATA => 1<<33
 * CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH => 1<<34
 * CUPTI_ACTIVITY_KIND_OPENACC_OTHER => 1<<35
 * CUPTI_ACTIVITY_KIND_CUDA_EVENT => 1<<36
 * CUPTI_ACTIVITY_KIND_STREAM => 1<<37
 * CUPTI_ACTIVITY_KIND_SYNCHRONIZATION => 1<<38
 * CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION => 1<<39
 * CUPTI_ACTIVITY_KIND_NVLINK => 1<<40
 * CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT => 1<<41
 * CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE => 1<<42
 * CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC => 1<<43
 * CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE => 1<<44
 * CUPTI_ACTIVITY_KIND_MEMORY => 1<<45
 * CUPTI_ACTIVITY_KIND_PCIE => 1<<46
 * CUPTI_ACTIVITY_KIND_OPENMP => 1<<47
 * CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API => 1<<48
 * 
 */

void enableActivityKind(unsigned long flag) {
	// This is the default kinds of activities(the first 17 kinds), and we haved implemented recording 
	// methods for these activities. [TO DO]If other kinds of activities be chosen, we may need to 
	// implement the recording methods for that kind.
	if (flag == 0) {
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));

		// A kernel executing on the GPU. This activity kind may significantly change the overall performance 
		// characteristics of the application because all kernel executions are serialized on the GPU.
		/* CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL)); 	    */

		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));

		// Event/Metric activity can be recorded only when profiling activity is invoked. [NOT SUPPORTED NOW]
		// This activity record kind is not produced by the activity API but is included for completeness and ease-of-use. 
		// Profile frameworks built on top of CUPTI that collect event data may choose to use this type to store the collected event data.  
		/* CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EVENT));		    */
		/* CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_METRIC));         */

		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
		
		// Extended, optional, data about a marker
		/* CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER_DATA));	*/

		// Source level result/global-access/branch
		// Invalid Now.
		/* CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR)); */
		/* CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS));  */
		/* CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_BRANCH));			*/

		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
		return;
	}

	// Currently, we only support the first 17 activities. If a activity beyond 17 is chosen,
	// this function will print an warning.
	if (flag > (0x1 << 17)) {
		std::cout << "[WARNING] Only part of first 17 activities are supported now! " << std::endl;
		std::cout << "[WARNING] The results may not be what your expected." << std::endl;
	}

	if (flag & (0x1 << 1)) {
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));	
	}
	if (flag & (0x1 << 2)) {
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
	}
	if (flag & (0x1 << 3)) {
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
	}
	if (flag & (0x1 << 4)) {
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
	}
	if (flag & (0x1 << 5)) {
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
	}
	if (flag & (0x1 << 8)) {
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
	}
	if (flag & (0x1 << 9)) {
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
	}
	if (flag & (0x1 << 10)) {
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
	}
	if (flag & (0x1 << 11)) {
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
	}
	if (flag & (0x1 << 12)) {
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
	}
	if (flag & (0x1 << 17)) {
		CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
	}
}