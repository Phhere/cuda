#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include "cuda_kernel.cuh"

int solveProblem(const int argc, const char* argv[]){
	cudaError_t return_value;
	if(argc == 2){
		cudaEvent_t start, stop, memcopystart, memcopystop;
		float time, memcopytime;
		int vectorlength = atoi(argv[1]);
		cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&memcopystart);
        cudaEventCreate(&memcopystop);
		// Everything runs on stream 0
		cudaEventRecord(start);

		/* Start Program */

        size_t size = sizeof(int) * vectorlength;
        int *hosta = (int*) malloc(size);
        int *hostb = (int*) malloc(size);
        int *hostc = (int*) malloc(size);
        // maybe null?

        int *deva, *devb, *devc;

        if (cudaMalloc((void**) &deva, size) != cudaSuccess) {
            printf("deva failed\n");
            return -1;
        }
        if (cudaMalloc((void**) &devb, size) != cudaSuccess) {
            printf("devb failed\n");
            return -1;
        }
        if (cudaMalloc((void**) &devc, size) != cudaSuccess) {
            printf("devc failed\n");
            return -1;
        }

        for(int i = 0; i < vectorlength; i++) {
            hosta[i] = i/2;
            hostb[i] = i/2;
        }

        // Copy vectors to device
        cudaEventRecord(memcopystart);
        if (cudaMemcpy(deva, hosta, size, cudaMemcpyHostToDevice) != cudaSuccess) {
            printf("deva to device failed.\n");
            return -1;
        }
        if (cudaMemcpy(devb, hostb, size, cudaMemcpyHostToDevice) != cudaSuccess) {
            printf("devb to device failed.\n");
            return -1;
        }
        cudaEventRecord(memcopystop);
        cudaEventSynchronize(memcopystop);

        // Calculate blocksize and threadnumber
        int blocksPerGrid = 1;
        int threadsPerBlock = vectorlength;

        if (vectorlength > 1024) {
            blocksPerGrid = (int) ceil(vectorlength / 1024.0);
            threadsPerBlock = 1024;
        }

        printf("BlocksPerGrid: %d\n", blocksPerGrid);
        printf("ThreadsPerBlock: %d\n", threadsPerBlock);

        // Kernel time!
		kernel<<<blocksPerGrid, threadsPerBlock>>>(deva, devb, devc);

        // Copy results back to host
        if (cudaMemcpy(hostc, devc, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Device to host failed.\n");
            return -1;
        }

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Print results
        //for (int i = 0; i < vectorlength; ++i) {
        printf("C[%d] = %d\n", vectorlength-1, hostc[vectorlength-1]);
        //}

        cudaFree(deva);
        cudaFree(devb);
        cudaFree(devc);

        free(hosta);
        free(hostb);
        free(hostc);

        /* End Program */

	    return_value = cudaGetLastError();
	    if(return_value != cudaSuccess){
	    	printf("Error in Kernel: %d\n", return_value);
	    	printf("%s\n", cudaGetErrorString(return_value));
	    	return -1;
	    }
	    cudaEventElapsedTime(&time, start, stop);
	    printf("Time for the kernel: %f ms\n", time);
        cudaEventElapsedTime(&memcopytime, memcopystart, memcopystop);
        printf("Time to copy data to device: %f ms\n", memcopytime);
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
