#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <pthread.h>

#include "extensions.h"
#include "counter.h"

static int numRanges = 1024;
static pthread_mutex_t opCount_mutex = PTHREAD_MUTEX_INITIALIZER;
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
                    CUpti_ProfilerRange profilerRange,
                    std::string opName)
{
    CUcontext cuContext;
    DRIVER_API_CALL(cuCtxGetCurrent(&cuContext));

    CUpti_Profiler_BeginSession_Params beginSessionParams = {CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    CUpti_Profiler_SetConfig_Params setConfigParams = {CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
    CUpti_Profiler_PushRange_Params pushRangeParams = {CUpti_Profiler_PushRange_Params_STRUCT_SIZE};

    beginSessionParams.ctx = NULL;
    beginSessionParams.counterDataImageSize = counterDataImage.size();
    beginSessionParams.pCounterDataImage = &counterDataImage[0];
    beginSessionParams.counterDataScratchBufferSize = counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
    beginSessionParams.range = profilerRange;
    if (profilerRange == CUPTI_UserRange) {
        beginSessionParams.replayMode = CUPTI_UserReplay;
    }
    else {
       beginSessionParams.replayMode = CUPTI_KernelReplay; 
    }
    beginSessionParams.maxRangesPerPass = numRanges;
    beginSessionParams.maxLaunchesPerPass = numRanges;

    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

    setConfigParams.pConfig = &configImage[0];
    setConfigParams.configSize = configImage.size();

    // printf("CUPTI_KernelReplay\r\n");
    if (profilerRange == CUPTI_AutoRange) {
        setConfigParams.passIndex = 0;
        CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
        CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
    }
    else if (profilerRange == CUPTI_UserRange) {
        setConfigParams.passIndex = 0;
        setConfigParams.minNestingLevel = 1;
        setConfigParams.numNestingLevels = 1;
        CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));

        CUpti_Profiler_BeginPass_Params beginPassParams = {CUpti_Profiler_BeginPass_Params_STRUCT_SIZE};
        CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));
        CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
        pushRangeParams.pRangeName = opName.c_str();
        CUPTI_API_CALL(cuptiProfilerPushRange(&pushRangeParams));
    }

    return true;
}

void startProfiling(counterControler* controler, bool userRange, std::string opName)
{
    std::cout << "START OP: " << opName << std::endl;

    CUpti_ProfilerRange profilerRange = CUPTI_AutoRange;
    if (userRange) {
        profilerRange = CUPTI_UserRange;
    }

    DRIVER_API_CALL(cuInit(0));
    validateProfilerEnvironment(controler->deviceNum);
    
    CUdevice cuDevice;
    DRIVER_API_CALL(cuDeviceGet(&cuDevice, controler->deviceNum));
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

    if(!setupProfiling(controler->configImage, controler->counterDataScratchBuffer, controler->counterDataImage, profilerRange, opName))
    {
        std::cout << "Failed to setup profiling" << std::endl;
        exit(-1);
    }
}

void counter::startProfilingKernel(std::string opName) {
    this->countRange = Counter::AutoRange;
    counterControler* controler = this->getControler();
    
    if (this->countMode == Counter::OFFLINE_AND_ONLINE || this->countMode == Counter::ONLINE_ONLY) {
        Counter::countData_t newOp;
        newOp.rangeName = "NEW OP";
        newOp.metricName = opName;
        this->countData.push_back(newOp);
    }
    if (this->countMode == Counter::OFFLINE_AND_ONLINE || this->countMode == Counter::ONLINE_ONLY) {
        std::fstream countFile;
        countFile.open(this->filePath, std::ios::app);
        countFile << "New Op: " << opName << std::endl;
        countFile.close();
    }

    int rc = pthread_mutex_lock(&opCount_mutex);
    this->opCount++;
    if (this->opCount == 1) {
        startProfiling(controler, false, opName);
    }
    rc = pthread_mutex_unlock(&opCount_mutex);
    std::cout << "Start Profiling!" << std::endl;
}

void counter::startProfilingOp(std::string opName) {
    this->countRange = Counter::UserRange;
    counterControler* controler = this->getControler();

    int rc = pthread_mutex_lock(&opCount_mutex);
    this->opCount++;
    if (this->opCount == 1) {
        startProfiling(controler, false, opName);
    }
    rc = pthread_mutex_unlock(&opCount_mutex);
    std::cout << "Start Profiling!" << std::endl;
}

void counter::stopProfiling() 
{
    int rc = pthread_mutex_lock(&opCount_mutex);
    this->opCount--;
    if (this->opCount == 0) {
       std::cout << "Stop Profiling!" << std::endl;
        if(this->countRange == Counter::UserRange) {
            CUpti_Profiler_PopRange_Params popRangeParams = {CUpti_Profiler_PopRange_Params_STRUCT_SIZE};
            CUPTI_API_CALL(cuptiProfilerPopRange(&popRangeParams));
            CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
            CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams)); 
            CUpti_Profiler_EndPass_Params endPassParams = {CUpti_Profiler_EndPass_Params_STRUCT_SIZE};
            CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));
            CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
            CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));
        }
        else {
            CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
            CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
            CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
            CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));         
        }

        CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
        CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
        CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
        CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));

        CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
        CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));

        bool fileFlag = true, dataFlag = true;
        if (this->countMode == Counter::OFFLINE_ONLY) { dataFlag = false; }
        if (this->countMode == Counter::ONLINE_ONLY) { fileFlag = false; }

        counterControler* controler = this->getControler();
        /* Evaluation of metrics collected in counterDataImage, this can also be done offline*/
        NV::Metric::Eval::GetMetricValues(controler->chipName, controler->counterDataImage, controler->metricNames, fileFlag, this->filePath, dataFlag, this->countData); 
    }
    rc = pthread_mutex_unlock(&opCount_mutex);
}
