/*
 * Copyright 2011-2020 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print a trace of CUDA API and GPU activity
 */ 
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "kernel.cuh"

#define COMPUTE_N 50000

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

// Kernels
__global__ void 
VecAdd(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

__global__ void 
VecSub(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] - B[i];
}

void
do_pass(cudaStream_t stream)
{
  int *h_A, *h_B, *h_C;
  int *d_A, *d_B, *d_C;
  size_t size = COMPUTE_N * sizeof(int);
  int threadsPerBlock = 256;
  int blocksPerGrid = 0;
  
  // Allocate input vectors h_A and h_B in host memory
  // don't bother to initialize
  h_A = (int*)malloc(size);
  h_B = (int*)malloc(size);
  h_C = (int*)malloc(size);
  
  // Allocate vectors in device memory
  RUNTIME_API_CALL(cudaMalloc((void**)&d_A, size));
  RUNTIME_API_CALL(cudaMalloc((void**)&d_B, size));
  RUNTIME_API_CALL(cudaMalloc((void**)&d_C, size));

  RUNTIME_API_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
  RUNTIME_API_CALL(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

  blocksPerGrid = (COMPUTE_N + threadsPerBlock - 1) / threadsPerBlock;
  VecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, COMPUTE_N);
  VecSub<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, COMPUTE_N);
    
  RUNTIME_API_CALL(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));

  if (stream == 0)
    RUNTIME_API_CALL(cudaDeviceSynchronize());
  else
    RUNTIME_API_CALL(cudaStreamSynchronize(stream));

  free(h_A);
  free(h_B);
  free(h_C);
  RUNTIME_API_CALL(cudaFree(d_A));
  RUNTIME_API_CALL(cudaFree(d_B));
  RUNTIME_API_CALL(cudaFree(d_C));
}
