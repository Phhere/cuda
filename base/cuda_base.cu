#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include "cuda_kernel.cuh"

int solveProblem(const int argc, const char* argv[]){
	cudaError_t return_value;
	if(argc == 2){
		cudaEvent_t start, stop;
		float time;
		int threads = atoi(argv[1]);
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		// Everything runs on stream 0
		cudaEventRecord(start);
		kernel<<<1,threads>>>();
		cudaDeviceSynchronize();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
	    return_value = cudaGetLastError();
	    if(return_value != cudaSuccess){
	    	printf("Error in Kernel\n");
	    	printf("%s\n",cudaGetErrorString(return_value));
	    	return -1;
	    }
	    cudaEventElapsedTime(&time, start, stop);
	    printf ("Time for the kernel: %f ms\n", time);
		return 0;
	}
	else{
		printf("parameter required\n");
		return -1;
	}
}

int main(const int argc, const char* argv[]){
	int devices;
	cudaError_t return_value;
	return_value = cudaGetDeviceCount(&devices);
	if(return_value != cudaSuccess){
		printf("Could not get device count\n");	
		return -1;
	}

	if(devices > 0){
		printf("%d devices found\n",devices);
		for(int device = 0; device < devices; device++){
			cudaDeviceProp device_info;
			cudaGetDeviceProperties(&device_info, device);
			printf("Name: %s\n",device_info.name);
			printf("max. Memory: %.0f MB\n",(double)device_info.totalGlobalMem/(double)(1024*1024));
			printf("max. Threads per Block: %d\n", device_info.maxThreadsPerBlock);
			printf("max. Blocks per Grid: %d,%d,%d\n", device_info.maxGridSize[0], device_info.maxGridSize[1],device_info.maxGridSize[2]);
			printf("\n");
		}
		return solveProblem(argc, argv);
	}
	else{
		printf("No CUDA cards found\n");
		return -1;
	}

}