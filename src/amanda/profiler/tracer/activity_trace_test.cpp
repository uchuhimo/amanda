#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#include "tracer.h"
#include "kernel.cuh"

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
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

int
main(int argc, char *argv[])
{
  CUdevice device;  
  char deviceName[32];
  int deviceNum = 0, devCount = 0;

  // initialize the activity trace
  // make sure activity is enabled before any CUDA API
  tracer tracer(0);
  tracer.initTrace();

  DRIVER_API_CALL(cuInit(0));
  
  RUNTIME_API_CALL(cudaGetDeviceCount(&devCount));
  for (deviceNum=0; deviceNum<devCount; deviceNum++) {
      DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
      DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, device));
      printf("Device Name: %s\n", deviceName);

      RUNTIME_API_CALL(cudaSetDevice(deviceNum));
      // do pass default stream
      do_pass(0);

      // do pass with user stream
      cudaStream_t stream0;
      RUNTIME_API_CALL(cudaStreamCreate(&stream0));
      do_pass(stream0);

      RUNTIME_API_CALL(cudaDeviceSynchronize());

      // Flush all remaining CUPTI buffers before resetting the device.
      // This can also be called in the cudaDeviceReset callback.
      tracer.activityFlushAll();
      RUNTIME_API_CALL(cudaDeviceReset());
  }
  
  tracer.finishTrace();
  return 0;
}