#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_kernel.cuh"
#include "cuda_threadindex.cuh"
#include <stdio.h>

// Axel ist doof

/**
 * Calculate sum of vectors
 */
__global__ void matrix_mul_gpu_element(float *gpuA, float *gpuB, float *gpuC, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < n) {
        float a, b, sum = 0;
        for (int k = 0; k < n; ++k) {
            a = gpuA[k + row*n];
            b = gpuB[col + k*n];
            sum += a * b;
        }
        gpuC[col + row*n] = sum;
    }
}
