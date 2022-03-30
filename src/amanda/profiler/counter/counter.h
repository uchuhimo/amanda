#pragma once

#include <vector>
#include <string>
#include <iostream>

typedef struct counterControler
{
	// Parameters needed to be set before profiling
	int deviceNum = 0;
	std::vector<std::string> metricNames;

	// Profiling values 
	std::string chipName;
	std::vector<uint8_t> counterDataImage;

	// Assisted parameters, their lifetime last during the whole profiling period
	std::vector<uint8_t> counterDataImagePrefix;
    std::vector<uint8_t> configImage;
    std::vector<uint8_t> counterDataScratchBuffer;
    std::vector<uint8_t> counterAvailabilityImage;
}counterControler;

namespace Counter {
	typedef struct countData_s {
		std::string rangeName;
		std::string metricName;
		double gpuValue;
	} countData_t;

	typedef enum {
		OFFLINE_AND_ONLINE = 0,
		OFFLINE_ONLY = 1,
		ONLINE_ONLY = 2,
	} count_Mode;
};

class counter {
	std::string filePath;
	unsigned long kindFlag;

	counterControler controler;
	Counter::count_Mode countMode;
	void setMetrics(unsigned long flag);
	counterControler* getControler();

public:
	std::vector<Counter::countData_t> countData;
	counter();
	counter(std::string);
	counter(unsigned long kindFlag);
	counter(unsigned long kindFlag, std::string filePath);
	~counter();

	void setFilePath(std::string filePath);
	void setKindFlag(unsigned long kindFlag);
	std::string getFilePath();
	unsigned long getKindFlag();

	void onlineAnalysisOnly();
	void offlineAnalysisOnly();
	void setCountDevice(int deviceNum);
	void setCountParams(int deviceNum, std::vector<std::string> metricsNames);

	void clearData();
	void startProfiling();
	void stopProfiling();
};