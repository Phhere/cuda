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
		int vectorlength = atoi(argv[1]);
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		// Everything runs on stream 0
		cudaEventRecord(start);

		/* Start Program */

        size_t size = sizeof(int) * vectorlength;
        int *hosta = (int*) malloc(size);
        int *hostb = (int*) malloc(size);
        int *hostc = (int*) malloc(size);
        // maybe null?

        int *deva, *devb, *devc;

        cudaMalloc((void**) &deva, size);
        cudaMalloc((void**) &devb, size);
        cudaMalloc((void**) &devc, size);
        // maybe null?

        for(int i = 0; i < vectorlength; i++) {
            hosta[i] = i;
            hostb[i] = i*i;
        }

        // Copy vectors to device
        cudaMemcpy(deva, hosta, size, cudaMemcpyHostToDevice);
        cudaMemcpy(devb, hostb, size, cudaMemcpyHostToDevice);

        // Calculate blocksize and threadnumber
        int blocksPerGrid = 1;
        int threadsPerBlock = vectorlength;

        if (vectorlength > 1024) {
            blocksPerGrid = (int) ceil(vectorlength / 1024.0);
            threadsPerBlock = 1024;
        }

        // Kernel time!
		kernel<<<blocksPerGrid, threadsPerBlock>>>(deva, devb, devc);

        // Copy results back to host
        cudaMemcpy(hostc, devc, size, cudaMemcpyDeviceToHost);

        // Print results
        for (int i = 0; i < vectorlength; ++i) {
            printf("C[%d] = %d\n", i, hostc[i]);
        }

        cudaFree(deva);
        cudaFree(devb);
        cudaFree(devc);

        free(hosta);
        free(hostb);
        free(hostc);

        /* End Program */

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
	} else {
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
