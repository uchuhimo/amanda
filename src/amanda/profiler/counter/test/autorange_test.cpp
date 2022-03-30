#include <cuda.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <ostream>
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>


#include "../counter.h"
#include "kernel.cuh"

static int numRanges = 2;
#define METRIC_NAME "smsp__warps_launched.avg"

static void initVec(int *vec, int n)
{
  for (int i=0; i< n; i++)
    vec[i] = i;
}

static void cleanUp(int *h_A, int *h_B, int *h_C, int *h_D, int *d_A, int *d_B, int *d_C, int *d_D)
{
  if (d_A)
    cudaFree(d_A);
  if (d_B)
    cudaFree(d_B);
  if (d_C)
    cudaFree(d_C);
  if (d_D)
    cudaFree(d_D);

  // Free host memory
  if (h_A)
    free(h_A);
  if (h_B)
    free(h_B);
  if (h_C)
    free(h_C);
  if (h_D)
    free(h_D);}

bool runTest(int deviceNum, std::vector<std::string> metricNames)
{

    int N = 1024*1024*1024/4/3;
    size_t size = N * sizeof(int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int *h_A, *h_B, *h_C, *h_D;
    int *d_A, *d_B, *d_C, *d_D;
    int i, sum, diff;

    // Allocate input vectors h_A and h_B in host memory
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);
    h_D = (int*)malloc(size);

    // Initialize input vectors
    initVec(h_A, N);
    initVec(h_B, N);
    memset(h_C, 0, size);
    memset(h_D, 0, size);

    // Allocate vectors in device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMalloc((void**)&d_D, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // create a counter and set the parameters
    counter counter(0);
    counter.setCountParams(deviceNum, metricNames);

    // start profiling...
    counter.startProfiling();
    std::cout << "start profiling ..." << std::endl;

    callVecAdd(d_A, d_B, d_C, N);
    callVecSub(d_A, d_B, d_D, N);


    // stop profiling
    counter.stopProfiling();
    std::cout << "stop profiling ..." << std::endl;
    // counter.printValues();


    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost);

    // Verify result
    for (i = 0; i < N; ++i) {
        sum = h_A[i] + h_B[i];
        diff = h_A[i] - h_B[i];
        if (h_C[i] != sum || h_D[i] != diff) {
        fprintf(stderr, "error: result verification failed\n");
        exit(-1);
        }
    }

    cleanUp(h_A, h_B, h_C, h_D, d_A, d_B, d_C, d_D);

    return true;
}

int main(int argc, char* argv[])
{
    // std::string CounterDataFileName("SimpleCupti.counterdata");
    // std::string CounterDataSBFileName("SimpleCupti.counterdataSB");
    int deviceNum;
    char* metricName;
    std::vector<std::string> metricNames;

    printf("Usage: %s [device_num] [metric_names comma separated]\n", argv[0]);

    if (argc > 1)
        deviceNum = atoi(argv[1]);
    else
        deviceNum = 0;
    printf("CUDA Device Number: %d\n", deviceNum);

    // Get the names of the metrics to collect
    if (argc > 2) {
        metricName = strtok(argv[2], ",");
        while(metricName != NULL)
        {
            metricNames.push_back(metricName);
            metricName = strtok(NULL, ",");
        }
    }
    else {
        metricNames.push_back(METRIC_NAME);
    }

    if(!runTest(deviceNum, metricNames))
    {
        std::cout << "Failed to run sample" << std::endl;
        exit(-1);
    }

    /* Dump counterDataImage in file */
    // WriteBinaryFile(CounterDataFileName.c_str(), counterDataImage);
    // WriteBinaryFile(CounterDataSBFileName.c_str(), counterDataScratchBuffer);


    return 0;
}