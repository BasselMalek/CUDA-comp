#include "solve.h"
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int my_place = blockDim.x * blockIdx.x + threadIdx.x;
    if(my_place>=N/2){
        return;
    }
    float temp = input[my_place];
    input[my_place] = input[N - my_place - 1];
    input[N - my_place - 1] = temp;

}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
