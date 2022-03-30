
__global__ void VecAdd(const int* A, const int* B, int* C, int N);
__global__ void VecSub(const int* A, const int* B, int* C, int N);
void do_pass(cudaStream_t stream);