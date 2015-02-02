#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_kernel.cuh"
#include "cuda_threadindex.cuh"
#include <stdio.h>

__global__ void kernel(){
	int threadID = calcIndex1DGrid1DThreads();
	printf("%d\n",threadID);
}