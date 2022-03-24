#pragma once

#include <string>
#include <vector>


template <typename T>

class ScopeExit
{
public:
    ScopeExit(T t) : t(t) {}
    ~ScopeExit() { t(); }
    T t;
};

template <typename T>
ScopeExit<T> MoveScopeExit(T t) {
    return ScopeExit<T>(t);
};

#define NV_ANONYMOUS_VARIABLE_DIRECT(name, line) name##line
#define NV_ANONYMOUS_VARIABLE_INDIRECT(name, line) NV_ANONYMOUS_VARIABLE_DIRECT(name, line)

#define SCOPE_EXIT(func) const auto NV_ANONYMOUS_VARIABLE_INDIRECT(EXIT, __LINE__) = MoveScopeExit([=](){func;})

#include <nvperf_host.h>


#define RETURN_IF_NVPW_ERROR(retval, actual)                                        \
do {                                                                                \
    NVPA_Status status = actual;                                                    \
    if (NVPA_STATUS_SUCCESS != status) {                                            \
        fprintf(stderr, "FAILED: %s with error %s\n", #actual, NV::Metric::Utils::GetNVPWResultString(status)); \
        return retval;                                                              \
    }                                                                               \
} while (0)

namespace NV {
    namespace Metric {
        namespace Utils {

            static const char* GetNVPWResultString(NVPA_Status status) {
                const char* errorMsg = NULL;
                switch (status)
                {
                case NVPA_STATUS_ERROR:
                    errorMsg = "NVPA_STATUS_ERROR";
                    break;
                case NVPA_STATUS_INTERNAL_ERROR:
                    errorMsg = "NVPA_STATUS_INTERNAL_ERROR";
                    break;
                case NVPA_STATUS_NOT_INITIALIZED:
                    errorMsg = "NVPA_STATUS_NOT_INITIALIZED";
                    break;
                case NVPA_STATUS_NOT_LOADED:
                    errorMsg = "NVPA_STATUS_NOT_LOADED";
                    break;
                case NVPA_STATUS_FUNCTION_NOT_FOUND:
                    errorMsg = "NVPA_STATUS_FUNCTION_NOT_FOUND";
                    break;
                case NVPA_STATUS_NOT_SUPPORTED:
                    errorMsg = "NVPA_STATUS_NOT_SUPPORTED";
                    break;
                case NVPA_STATUS_NOT_IMPLEMENTED:
                    errorMsg = "NVPA_STATUS_NOT_IMPLEMENTED";
                    break;
                case NVPA_STATUS_INVALID_ARGUMENT:
                    errorMsg = "NVPA_STATUS_INVALID_ARGUMENT";
                    break;
                case NVPA_STATUS_INVALID_METRIC_ID:
                    errorMsg = "NVPA_STATUS_INVALID_METRIC_ID";
                    break;
                case NVPA_STATUS_DRIVER_NOT_LOADED:
                    errorMsg = "NVPA_STATUS_DRIVER_NOT_LOADED";
                    break;
                case NVPA_STATUS_OUT_OF_MEMORY:
                    errorMsg = "NVPA_STATUS_OUT_OF_MEMORY";
                    break;
                case NVPA_STATUS_INVALID_THREAD_STATE:
                    errorMsg = "NVPA_STATUS_INVALID_THREAD_STATE";
                    break;
                case NVPA_STATUS_FAILED_CONTEXT_ALLOC:
                    errorMsg = "NVPA_STATUS_FAILED_CONTEXT_ALLOC";
                    break;
                case NVPA_STATUS_UNSUPPORTED_GPU:
                    errorMsg = "NVPA_STATUS_UNSUPPORTED_GPU";
                    break;
                case NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION:
                    errorMsg = "NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION";
                    break;
                case NVPA_STATUS_OBJECT_NOT_REGISTERED:
                    errorMsg = "NVPA_STATUS_OBJECT_NOT_REGISTERED";
                    break;
                case NVPA_STATUS_INSUFFICIENT_PRIVILEGE:
                    errorMsg = "NVPA_STATUS_INSUFFICIENT_PRIVILEGE";
                    break;
                case NVPA_STATUS_INVALID_CONTEXT_STATE:
                    errorMsg = "NVPA_STATUS_INVALID_CONTEXT_STATE";
                    break;
                case NVPA_STATUS_INVALID_OBJECT_STATE:
                    errorMsg = "NVPA_STATUS_INVALID_OBJECT_STATE";
                    break;
                case NVPA_STATUS_RESOURCE_UNAVAILABLE:
                    errorMsg = "NVPA_STATUS_RESOURCE_UNAVAILABLE";
                    break;
                case NVPA_STATUS_DRIVER_LOADED_TOO_LATE:
                    errorMsg = "NVPA_STATUS_DRIVER_LOADED_TOO_LATE";
                    break;
                case NVPA_STATUS_INSUFFICIENT_SPACE:
                    errorMsg = "NVPA_STATUS_INSUFFICIENT_SPACE";
                    break;
                case NVPA_STATUS_OBJECT_MISMATCH:
                    errorMsg = "NVPA_STATUS_OBJECT_MISMATCH";
                    break;
                case NVPA_STATUS_VIRTUALIZED_DEVICE_NOT_SUPPORTED:
                    errorMsg = "NVPA_STATUS_VIRTUALIZED_DEVICE_NOT_SUPPORTED";
                    break;
                default:
                    break;
                }

                return errorMsg;
            }
        }
    }
}

namespace NV {
    namespace Metric {
        namespace Parser {
            inline bool ParseMetricNameString(const std::string& metricName, std::string* reqName, bool* isolated, bool* keepInstances)
            {
                std::string& name = *reqName;
                name = metricName;
                if (name.empty())
                {
                    return false;
                }

                // boost program_options sometimes inserts a \n between the metric name and a '&' at the end
                size_t pos = name.find('\n');
                if (pos != std::string::npos)
                {
                    name.erase(pos, 1);
                }

                // trim whitespace
                while (name.back() == ' ')
                {
                    name.pop_back();
                    if (name.empty())
                    {
                        return false;
                    }
                }

                *keepInstances = false;
                if (name.back() == '+')
                {
                    *keepInstances = true;
                    name.pop_back();
                    if (name.empty())
                    {
                        return false;
                    }
                }

                *isolated = true;
                if (name.back() == '$')
                {
                    name.pop_back();
                    if (name.empty())
                    {
                        return false;
                    }
                }
                else if (name.back() == '&')
                {
                    *isolated = false;
                    name.pop_back();
                    if (name.empty())
                    {
                        return false;
                    }
                }

                return true;
            }
        }
    }
}

namespace NV {
    namespace Metric {
        namespace Config {
            /* Function to get Config image
            * @param[in]  chipName                          Chip name for which configImage is to be generated
            * @param[in]  metricNames                       List of metrics for which configImage is to be generated
            * @param[out] configImage                       Generated configImage
            * @param[in]  pCounterAvailabilityImage         Pointer to counter availability image queried on target device, can be used to filter unavailable metrics
            */
            bool GetConfigImage(std::string chipName, std::vector<std::string> metricNames, std::vector<uint8_t>& configImage, const uint8_t* pCounterAvailabilityImage = NULL);

            /* Function to get CounterDataPrefix image
            * @param[in]  chipName                  Chip name for which counterDataImagePrefix is to be generated
            * @param[in]  metricNames               List of metrics for which counterDataImagePrefix is to be generated
            * @param[out] counterDataImagePrefix    Generated counterDataImagePrefix
            */
            bool GetCounterDataPrefixImage(std::string chipName, std::vector<std::string> metricNames, std::vector<uint8_t>& counterDataImagePrefix);
        }
    }
}

namespace NV {
    namespace Metric {
        namespace Eval {
            struct MetricNameValue {
                std::string metricName;
                int numRanges;
                // <rangeName , metricValue> pair
                std::vector < std::pair<std::string, double> > rangeNameMetricValueMap;
            };


            /* Function to get aggregate metric value
             * @param[in]  chipName                 Chip name for which to get metric values
             * @param[in]  counterDataImage         Counter data image
             * @param[in]  metricNames              List of metrics to read from counter data image
             * @param[out] metricNameValueMap       Metric name value map
             */
            bool GetMetricGpuValue(std::string chipName, std::vector<uint8_t> counterDataImage, std::vector<std::string> metricNames, std::vector<MetricNameValue>& metricNameValueMap);

            bool PrintMetricValues(std::string chipName, std::vector<uint8_t> counterDataImage, std::vector<std::string> metricNames);

            }
    }
}