#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_kernel.cuh"
#include "cuda_threadindex.cuh"
#include <stdio.h>

/**
 * Calculate sum of vectors
 */
__global__ void kernel(int *a, int *b, int *c) {
	int threadID = calcIndex1DGrid1DThreads();

    c[threadID] = a[threadID] + b[threadID];

}
