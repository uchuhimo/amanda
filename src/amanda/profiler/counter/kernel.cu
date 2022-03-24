#include "kernel.cuh"

#include <stdio.h>

__global__ 
void VecAdd(const int* A, const int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

__global__ 
void VecSub(const int* A, const int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] - B[i];
}


void callVecAdd(const int* d_A, const int* d_B, int* d_C, int N){

    int threadsPerBlock = threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel: blocks %d, thread/block %d\n",
        blocksPerGrid, threadsPerBlock);

    VecAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);
}


void callVecSub(const int* d_A, const int* d_B, int* d_C, int N){

    int threadsPerBlock = threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel: blocks %d, thread/block %d\n",
        blocksPerGrid, threadsPerBlock);

    VecSub << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);
}