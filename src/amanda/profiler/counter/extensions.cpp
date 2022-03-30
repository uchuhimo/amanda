#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>
#include <iostream>
#include <fstream>

#include "extensions.h"
// #include "counter.h"

namespace NV {
    namespace Metric {
        namespace Config {

            bool GetRawMetricRequests(NVPA_MetricsContext* pMetricsContext,
                                      std::vector<std::string> metricNames,
                                      std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
                                      std::vector<std::string>& temp) {
                std::string reqName;
                bool isolated = true;
                bool keepInstances = true;

                for (auto& metricName : metricNames)
                {
                    NV::Metric::Parser::ParseMetricNameString(metricName, &reqName, &isolated, &keepInstances);
                    /* Bug in collection with collection of metrics without instances, keep it to true*/
                    keepInstances = true;
                    NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = { NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE };
                    getMetricPropertiesBeginParams.pMetricsContext = pMetricsContext;
                    getMetricPropertiesBeginParams.pMetricName = reqName.c_str();

                    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetMetricProperties_Begin(&getMetricPropertiesBeginParams));

                    for (const char** ppMetricDependencies = getMetricPropertiesBeginParams.ppRawMetricDependencies; *ppMetricDependencies; ++ppMetricDependencies)
                    {
                        temp.push_back(*ppMetricDependencies);
                    }
                    NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = { NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE };
                    getMetricPropertiesEndParams.pMetricsContext = pMetricsContext;
                    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetMetricProperties_End(&getMetricPropertiesEndParams));
                }

                for (auto& rawMetricName : temp)
                {
                    NVPA_RawMetricRequest metricRequest = { NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE };
                    metricRequest.pMetricName = rawMetricName.c_str();
                    metricRequest.isolated = isolated;
                    metricRequest.keepInstances = keepInstances;
                    rawMetricRequests.push_back(metricRequest);
                }

                return true;
            }

            bool GetConfigImage(std::string chipName, std::vector<std::string> metricNames, std::vector<uint8_t>& configImage, const uint8_t* pCounterAvailabilityImage) 
            {
                NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
                metricsContextCreateParams.pChipName = chipName.c_str();
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

                NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
                metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });
                
                std::vector<NVPA_RawMetricRequest> rawMetricRequests;
                std::vector<std::string> temp;
                GetRawMetricRequests(metricsContextCreateParams.pMetricsContext, metricNames, rawMetricRequests, temp);

                NVPA_RawMetricsConfigOptions metricsConfigOptions = { NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE };
                metricsConfigOptions.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
                metricsConfigOptions.pChipName = chipName.c_str();
                NVPA_RawMetricsConfig* pRawMetricsConfig;
                RETURN_IF_NVPW_ERROR(false, NVPA_RawMetricsConfig_Create(&metricsConfigOptions, &pRawMetricsConfig));

                if(pCounterAvailabilityImage)
                {
                    NVPW_RawMetricsConfig_SetCounterAvailability_Params setCounterAvailabilityParams = {NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
                    setCounterAvailabilityParams.pRawMetricsConfig = pRawMetricsConfig;
                    setCounterAvailabilityParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
                    RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_SetCounterAvailability(&setCounterAvailabilityParams));
                }

                NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = { NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE };
                rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
                SCOPE_EXIT([&]() { NVPW_RawMetricsConfig_Destroy((NVPW_RawMetricsConfig_Destroy_Params *)&rawMetricsConfigDestroyParams); });

                NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = { NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE };
                beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams));

                NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = { NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE };
                addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
                addMetricsParams.pRawMetricRequests = &rawMetricRequests[0];
                addMetricsParams.numMetricRequests = rawMetricRequests.size();
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams));

                NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE };
                endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams));

                NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = { NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE };
                generateConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GenerateConfigImage(&generateConfigImageParams));

                NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = { NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE };
                getConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
                getConfigImageParams.bytesAllocated = 0;
                getConfigImageParams.pBuffer = NULL;
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

                configImage.resize(getConfigImageParams.bytesCopied);

                getConfigImageParams.bytesAllocated = configImage.size();
                getConfigImageParams.pBuffer = &configImage[0];
                RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

                return true;
            }

            bool GetCounterDataPrefixImage(std::string chipName, std::vector<std::string> metricNames, std::vector<uint8_t>& counterDataImagePrefix) 
            {
                NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
                metricsContextCreateParams.pChipName = chipName.c_str();
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

                NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
                metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });

                std::vector<NVPA_RawMetricRequest> rawMetricRequests;
                std::vector<std::string> temp;
                GetRawMetricRequests(metricsContextCreateParams.pMetricsContext, metricNames, rawMetricRequests, temp);

                NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = { NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE };
                counterDataBuilderCreateParams.pChipName = chipName.c_str();
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_Create(&counterDataBuilderCreateParams));

                NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = { NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE };
                counterDataBuilderDestroyParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
                SCOPE_EXIT([&]() { NVPW_CounterDataBuilder_Destroy((NVPW_CounterDataBuilder_Destroy_Params *)&counterDataBuilderDestroyParams); });

                NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = { NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE };
                addMetricsParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
                addMetricsParams.pRawMetricRequests = &rawMetricRequests[0];
                addMetricsParams.numMetricRequests = rawMetricRequests.size();
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_AddMetrics(&addMetricsParams));

                size_t counterDataPrefixSize = 0;
                NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = { NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE };
                getCounterDataPrefixParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
                getCounterDataPrefixParams.bytesAllocated = 0;
                getCounterDataPrefixParams.pBuffer = NULL;
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

                counterDataImagePrefix.resize(getCounterDataPrefixParams.bytesCopied);

                getCounterDataPrefixParams.bytesAllocated = counterDataImagePrefix.size();
                getCounterDataPrefixParams.pBuffer = &counterDataImagePrefix[0];
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

                return true;
            }
        }
    }
}





namespace NV {
    namespace Metric {
        namespace Eval {
            std::string GetHwUnit(const std::string& metricName)
            {
                return metricName.substr(0, metricName.find("__", 0));
            }

            bool GetMetricGpuValue(std::string chipName, std::vector<uint8_t> counterDataImage, std::vector<std::string> metricNames, std::vector<MetricNameValue>& metricNameValueMap) {
                if (!counterDataImage.size()) {
                    std::cout << "Counter Data Image is empty!\n";
                    return false;
                }

                NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
                metricsContextCreateParams.pChipName = chipName.c_str();
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

                NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
                metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });

                NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
                getNumRangesParams.pCounterDataImage = &counterDataImage[0];
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterData_GetNumRanges(&getNumRangesParams));

                std::vector<std::string> reqName;
                reqName.resize(metricNames.size());

                bool isolated = true;
                bool keepInstances = true;
                std::vector<const char*> metricNamePtrs;
                metricNameValueMap.resize(metricNames.size());

                for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex) {
                    NV::Metric::Parser::ParseMetricNameString(metricNames[metricIndex], &reqName[metricIndex], &isolated, &keepInstances);
                    metricNamePtrs.push_back(reqName[metricIndex].c_str());
                    metricNameValueMap[metricIndex].metricName = metricNames[metricIndex];
                    metricNameValueMap[metricIndex].numRanges = getNumRangesParams.numRanges;
                }

                for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex) {
                    std::vector<const char*> descriptionPtrs;

                    NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = { NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
                    getRangeDescParams.pCounterDataImage = &counterDataImage[0];
                    getRangeDescParams.rangeIndex = rangeIndex;
                    RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));
                    descriptionPtrs.resize(getRangeDescParams.numDescriptions);

                    getRangeDescParams.ppDescriptions = &descriptionPtrs[0];
                    RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

                    std::string rangeName;
                    for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
                    {
                        if (descriptionIndex)
                        {
                            rangeName += "/";
                        }
                        rangeName += descriptionPtrs[descriptionIndex];
                    }

                    std::vector<double> gpuValues;
                    gpuValues.resize(metricNames.size());
                    NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = { NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE };
                    setCounterDataParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                    setCounterDataParams.pCounterDataImage = &counterDataImage[0];
                    setCounterDataParams.isolated = true;
                    setCounterDataParams.rangeIndex = rangeIndex;
                    NVPW_MetricsContext_SetCounterData(&setCounterDataParams);

                    NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = { NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE };
                    evalToGpuParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                    evalToGpuParams.numMetrics = metricNamePtrs.size();
                    evalToGpuParams.ppMetricNames = &metricNamePtrs[0];
                    evalToGpuParams.pMetricValues = &gpuValues[0];
                    NVPW_MetricsContext_EvaluateToGpuValues(&evalToGpuParams);
                    for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex) {
                        metricNameValueMap[metricIndex].rangeNameMetricValueMap.push_back(std::make_pair(rangeName, gpuValues[metricIndex]));
                    }
                }

                return true;
            }

            bool GetMetricValues(std::string chipName, std::vector<uint8_t> counterDataImage, std::vector<std::string> metricNames, bool fileFlag, std::string filePath, bool dataFlag, std::vector<Counter::countData_t>& countDataValues) {
                if (!counterDataImage.size()) {
                    std::cout << "Counter Data Image is empty!\n";
                    return false;
                }

                std::fstream countFile;
                countFile.open(filePath, std::ios::app);
                NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
                metricsContextCreateParams.pChipName = chipName.c_str();
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

                NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
                metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });

                NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
                getNumRangesParams.pCounterDataImage = &counterDataImage[0];
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterData_GetNumRanges(&getNumRangesParams));

                std::vector<std::string> reqName;
                reqName.resize(metricNames.size());
                bool isolated = true;
                bool keepInstances = true;
                std::vector<const char*> metricNamePtrs;
                for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex) {
                    NV::Metric::Parser::ParseMetricNameString(metricNames[metricIndex], &reqName[metricIndex], &isolated, &keepInstances);
                    metricNamePtrs.push_back(reqName[metricIndex].c_str());
                }

                for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex) {
                    std::vector<const char*> descriptionPtrs;

                    NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = { NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
                    getRangeDescParams.pCounterDataImage = &counterDataImage[0];
                    getRangeDescParams.rangeIndex = rangeIndex;
                    RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));
                    
                    descriptionPtrs.resize(getRangeDescParams.numDescriptions);
                    
                    getRangeDescParams.ppDescriptions = &descriptionPtrs[0];
                    RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

                    std::string rangeName;
                    for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
                    {
                        if (descriptionIndex)
                        {
                            rangeName += "/";
                        }
                        rangeName += descriptionPtrs[descriptionIndex];
                    }

                    const bool isolated = true;
                    std::vector<double> gpuValues;
                    gpuValues.resize(metricNames.size());

                    NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = { NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE };
                    setCounterDataParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                    setCounterDataParams.pCounterDataImage = &counterDataImage[0];
                    setCounterDataParams.isolated = true;
                    setCounterDataParams.rangeIndex = rangeIndex;
                    NVPW_MetricsContext_SetCounterData(&setCounterDataParams);

                    NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = { NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE };
                    evalToGpuParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
                    evalToGpuParams.numMetrics = metricNamePtrs.size();
                    evalToGpuParams.ppMetricNames = &metricNamePtrs[0];
                    evalToGpuParams.pMetricValues = &gpuValues[0];
                    NVPW_MetricsContext_EvaluateToGpuValues(&evalToGpuParams);
 
                    for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex) {
                        std::cout << "rangeName: " << rangeName << "\tmetricName: " << metricNames[metricIndex] << "\tgpuValue: "  << gpuValues[metricIndex] << std::endl;
                        if (fileFlag) {
                            countFile << "rangeName: " << rangeName << "\tmetricName: " << metricNames[metricIndex] << "\tgpuValue: "  << gpuValues[metricIndex] << std::endl;
                        }
                        if (dataFlag) {
                            Counter::countData_t countData;
                            countData.rangeName = rangeName;
                            countData.metricName = metricNames[metricIndex];
                            countData.gpuValue = gpuValues[metricIndex];
                            countDataValues.push_back(countData); 
                        }
                    }
                }
                countFile.close();
                return true;
            }
        }
    }
}
