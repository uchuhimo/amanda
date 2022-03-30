#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "extensions.h"
#include "counter.h"

static int numRanges = 2;
#define METRIC_NAME "smsp__warps_launched.avg"

#define NVPW_API_CALL(apiFuncCall)                                             \
do {                                                                           \
    NVPA_Status _status = apiFuncCall;                                         \
    if (_status != NVPA_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define CUPTI_API_CALL(apiFuncCall)                                            \
do {                                                                           \
    CUptiResult _status = apiFuncCall;                                         \
    if (_status != CUPTI_SUCCESS) {                                            \
        const char *errstr;                                                    \
        cuptiGetResultString(_status, &errstr);                                \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, errstr);                     \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        const char** errstr;                                                   \
        cuGetErrorString(_status, errstr);                                     \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, *errstr);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

bool CreateCounterDataImage(
    std::vector<uint8_t>& counterDataImage,
    std::vector<uint8_t>& counterDataScratchBuffer,
    std::vector<uint8_t>& counterDataImagePrefix)
{

    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
    counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
    counterDataImageOptions.maxNumRanges = numRanges;
    counterDataImageOptions.maxNumRangeTreeNodes = numRanges;
    counterDataImageOptions.maxRangeNameLength = 64;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};

    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    counterDataImage.resize(calculateSizeParams.counterDataImageSize);
    initializeParams.pCounterDataImage = &counterDataImage[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = initializeParams.pCounterDataImage;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));

    counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = {CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
    initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];

    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

    return true;
}

void validateProfilerEnvironment(int deviceNum) 
{
    int deviceCount;
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    
    CUdevice cuDevice;
    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit(-2);
    }

    DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

    printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor,computeCapabilityMinor);

    if(computeCapabilityMajor < 7) {
      printf("Sample unsupported on Device with compute capability < 7.0\n");
      exit(-2);
    }
}

void profilerInitialization(int deviceNum,
                             std::string& _chipname,
                             std::vector<uint8_t> *counterAvailabilityImage)
{
    CUcontext cuContext;
    DRIVER_API_CALL(cuCtxGetCurrent(&cuContext));    
    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

    /* Get chip name for the cuda  device */
    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = deviceNum;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    std::string chipName(getChipNameParams.pChipName);
    _chipname = chipName;

    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    getCounterAvailabilityParams.ctx = cuContext;
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));
    
    (*counterAvailabilityImage).clear();
    (*counterAvailabilityImage).resize(getCounterAvailabilityParams.counterAvailabilityImageSize);
    getCounterAvailabilityParams.pCounterAvailabilityImage = (*counterAvailabilityImage).data();
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    /* Generate configuration for metrics, this can also be done offline*/
    NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
    NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));
}

bool setupProfiling(std::vector<uint8_t>& configImage,
                    std::vector<uint8_t>& counterDataScratchBuffer,
                    std::vector<uint8_t>& counterDataImage,
                    CUpti_ProfilerRange profilerRange)
{
    CUcontext cuContext;
    DRIVER_API_CALL(cuCtxGetCurrent(&cuContext));

    CUpti_Profiler_BeginSession_Params beginSessionParams = {CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    CUpti_Profiler_SetConfig_Params setConfigParams = {CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};

    beginSessionParams.ctx = NULL;
    beginSessionParams.counterDataImageSize = counterDataImage.size();
    beginSessionParams.pCounterDataImage = &counterDataImage[0];
    beginSessionParams.counterDataScratchBufferSize = counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
    beginSessionParams.range = profilerRange;
    // KernelReplay
    beginSessionParams.replayMode = CUPTI_KernelReplay;
    beginSessionParams.maxRangesPerPass = numRanges;
    beginSessionParams.maxLaunchesPerPass = numRanges;

    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

    setConfigParams.pConfig = &configImage[0];
    setConfigParams.configSize = configImage.size();

    printf("CUPTI_KernelReplay\r\n");
    setConfigParams.passIndex = 0;
    CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
    return true;
}

void counter::startProfiling()
{
    CUpti_ProfilerRange profilerRange = CUPTI_AutoRange;
    counterControler* controler = this->getControler();

    DRIVER_API_CALL(cuInit(0));
    validateProfilerEnvironment(controler->deviceNum);
    
    CUdevice cuDevice;
    CUcontext cuContext;
    DRIVER_API_CALL(cuDeviceGet(&cuDevice, controler->deviceNum));
    DRIVER_API_CALL(cuCtxCreate(&cuContext, 0, cuDevice));
    profilerInitialization(controler->deviceNum, controler->chipName, &controler->counterAvailabilityImage);

    if (controler->metricNames.size()) {
        if(!NV::Metric::Config::GetConfigImage(controler->chipName, controler->metricNames, controler->configImage, controler->counterAvailabilityImage.data()))
        {
            std::cout << "Failed to create configImage" << std::endl;
            exit(-1);
        }
        if(!NV::Metric::Config::GetCounterDataPrefixImage(controler->chipName, controler->metricNames, controler->counterDataImagePrefix))
        {
            std::cout << "Failed to create counterDataImagePrefix" << std::endl;
            exit(-1);
        }
    }
    else
    {
        std::cout << "No metrics provided to profile" << std::endl;
        exit(-1);
    }

    if(!CreateCounterDataImage(controler->counterDataImage, controler->counterDataScratchBuffer, controler->counterDataImagePrefix))
    {
        std::cout << "Failed to create counterDataImage" << std::endl;
        exit(-1);
    }

    if(!setupProfiling(controler->configImage, controler->counterDataScratchBuffer, controler->counterDataImage, profilerRange))
    {
        std::cout << "Failed to setup profiling" << std::endl;
        exit(-1);
    }
}

void counter::stopProfiling() 
{
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
    CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));

    CUcontext cuContext;
    DRIVER_API_CALL(cuCtxGetCurrent(&cuContext));
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
    DRIVER_API_CALL(cuCtxDestroy(cuContext));
}

void counter::printValues()
{
    counterControler* controler = this->getControler();
    /* Evaluation of metrics collected in counterDataImage, this can also be done offline*/
    NV::Metric::Eval::PrintMetricValues(controler->chipName, controler->counterDataImage, controler->metricNames);
}
