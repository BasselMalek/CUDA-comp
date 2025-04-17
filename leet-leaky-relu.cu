#include "solve.h"
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
 int my_x = blockDim.x * blockIdx.x + threadIdx.x;
 if(my_x >= N){
    return;
 }
 output[my_x] = input[my_x]>0?input[my_x]:(0.01*input[my_x]);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
