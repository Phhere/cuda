#include <cuda.h>
#include <cuda_runtime.h>

__device__ int calcIndex1DGrid1DThreads(){
	return blockDim.x*blockIdx.x
			+ threadIdx.x;
}

__device__ int calcIndex1DGrid2DThreads(){
	return blockDim.x*blockIdx.x *blockDim.y 
			+ threadIdx.y * blockDim.x
			+ threadIdx.x;
}

__device__ int calcIndex1DGrid3DThreads(){
	return blockIdx.x*blockDim.x*blockDim.y*blockDim.z 
			+ threadIdx.z *blockDim.y*blockDim.x
			+ threadIdx.y*blockDim.x
			+ threadIdx.x;
}