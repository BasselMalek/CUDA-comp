#include "solve.h"
#include <cuda_runtime.h>
#define BLOCK_SIZE 16

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int my_col = blockDim.x * blockIdx.x + threadIdx.x;
    int my_row = blockDim.y * blockIdx.y + threadIdx.y;

    if (my_row < rows && my_col < cols) {
        output[my_col * rows + my_row] = input[my_row * cols + my_col]+.0f;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
