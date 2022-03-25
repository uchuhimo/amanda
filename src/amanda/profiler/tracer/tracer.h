// This file describes the api[class and function] of trace process.

#include <string>
#include <vector>

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
	 * Domian_flag: we use domain flag to confirm the activities belong to which
	 * 			domains should be recorded. Please refer to domain_enable.h for the
	 * 			specific relationship.
	 */
	unsigned short domainFlag;
	std::string filePath;

	/**
	 * traceMode will control online/offline data record of current trace process.
	 * Please refer to Trace::trace_Mode for more details.
	 * 
	 */
	Tracer::trace_Mode traceMode;

public:
	std::vector<Tracer::traceData_t> traceData;

	tracer();
	tracer(unsigned short domainFlag);
	tracer(std::string filePath);
	tracer(unsigned short domainFlag, std::string filePath);
	~tracer();

	void setDomainFlag(unsigned short domainFlag);
	void setFilePath(std::string filePath);
	/**
	 * The following two functions is used to set trace mode.
	 * Defalut mode will be set OFFLIJNE and ONLINE if both aren't
	 * be called.
	 * 
	 */
	void onlineAnalysisOnly();
	void offlineAnalysisOnly();

	unsigned short getDomainFlag();
	std::string getFilePath();
	Tracer::trace_Mode getTeaceMode();
};

// Flush all remaining CUPTI buffers before resetting the device.
// This can also be called in the cudaDeviceReset callback.
void activityFlushAll();

// Init the trace context
void initTrace();
// Finish the trace context, just flush the buffer
void finishTrace();