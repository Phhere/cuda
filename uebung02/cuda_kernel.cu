#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_kernel.cuh"
#include "cuda_threadindex.cuh"
#include <stdio.h>

/**
 * Calculate sum of vectors
 */
__global__ void kernel(){
	int threadID = calcIndex1DGrid1DThreads();

}
