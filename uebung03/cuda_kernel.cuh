#ifndef CUDA_KERNEL_CUH_
#define CUDA_KERNEL_CUH_

__global__ void matrix_mul_gpu_element(float *gpuA, float *gpuB, float *gpuC, int n);

#endif
