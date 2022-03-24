#include <vector>
#include <string>
#include <iostream>

#include "extensions.h"

typedef struct counterControler
{
	// Parameters needed to be set before profiling
	int deviceNum;
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

void startProfiling(counterControler *controler);
void stopProfiling();

void printValues(counterControler *controler);