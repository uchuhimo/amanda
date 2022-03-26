// This file describes the api[class and function] of trace process.

#include <string>
#include <vector>
#include <pthread.h>
#include <fstream>

namespace Tracer{

	// This is used to store the data values during the tracing process. Only contains those
	// types all the activities will record.
	typedef struct traceData_st
	{
	
		unsigned long long startTime;
		unsigned long long endTime;
		unsigned long long durationTime;

		unsigned int deviceId;
		unsigned int contextId;
		unsigned int streamId;
		unsigned int correlationId;

		std::string domain;
		std::string name;
	} traceData_t;

	/** 
	 * Type of trace mode.
	 * OFFLINE: All the values will write to a file.
	 * ONLINE: We will store data in the given data structure.
	 */
	typedef enum {
		OFFLINE_AND_ONLINE = 0,
		OFFLINE_ONLY = 1,
		ONLINE_ONLY = 2
	} trace_Mode;
};


// This is the trace controler class.
// And this class is the only thing that explode to upstream user.	
class tracer {
	/**
	 * Domain_flag and File_path is two parametes that user can set.
	 * File_path: the path user can set to record fully activity data.
	 * 			We should attention that only online mode be seted
	 * 			the file can be used. Also if the offline be set and
	 * 			file path not be given, we will write data to a fixed
	 * 			file called "activity_data.txt" under current directory.
	 * kind_flag: we use kind flag to confirm the activities belong to which
	 * 			kinds should be recorded. Please refer to kind_enable.h for the
	 * 			specific relationship.
	 */
	unsigned long kindFlag;
	std::string filePath;

	/**
	 * traceMode will control online/offline data record of current trace process.
	 * Please refer to Trace::trace_Mode for more details.
	 * 
	 * traceCount will record how many threads are under execution of current tracer.
	 * It will used to control open/close file.
	 * 
	 */
	Tracer::trace_Mode traceMode;
	int traceCount = 0;

public:
	std::vector<Tracer::traceData_t> traceData;

	tracer();
	tracer(unsigned long kindFlag);
	tracer(std::string filePath);
	tracer(unsigned long kindFlag, std::string filePath);
	~tracer();

	void setKindFlag(unsigned long kindFlag);
	void setFilePath(std::string filePath);
	/**
	 * The following two functions is used to set trace mode.
	 * Defalut mode will be set OFFLIJNE and ONLINE if both aren't
	 * be called.
	 * 
	 */
	void onlineAnalysisOnly();
	void offlineAnalysisOnly();

	unsigned long getKindFlag();
	std::string getFilePath();

	/**
	 * Flush all remaining CUPTI buffers before resetting the device.
	 * This can also be called in the cudaDeviceReset callback.
	 * 
	 */
	void activityFlushAll();

	/**
	 * Init and finish tracing process.
	 * After initTrace is called, tracing process of current thread starts.
	 * And the tracing process will last until finishTrace. They should occur in pair.
	 * 
	 */
	void initTrace();
	void finishTrace();
};

// Pointer to trace controler, which means we only allowed one tracing process
// at a time.
// We do this with constructors and destructors. If there has exist one tracer, then
// the second one will not be construct successfully. 
extern tracer *globalTracer_pointer;
extern pthread_mutex_t tracer_mutex;