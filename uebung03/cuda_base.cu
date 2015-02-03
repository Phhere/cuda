#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include "cuda_kernel.cuh"

void matrixmul_CPU(float *matA, float *matB, float *matC, int n) {
    float a, b, sum;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            sum = 0;
            for (int k = 0; k < n; ++k) {
                a = matA[k + i*n];
                b = matB[j + k*n];
                sum += a * b;
            }
            matC[j + i*n] = sum;
        }
    }
}

int solveProblem(const int argc, const char* argv[]){
	cudaError_t return_value;
	if(argc == 2){
		cudaEvent_t start, stop, memcopystart, memcopystop;
		float time, memcopytime;
		int matrixSize = atoi(argv[1]);

        size_t size = sizeof(float) * matrixSize * matrixSize;

        float *mata = (float *) malloc(size);
        float *matb = (float *) malloc(size);
        float *matc = (float *) malloc(size);

        for (int i = 0; i < matrixSize * matrixSize; ++i) {
            mata[i] = i;
            matb[i] = i+1;
            matc[i] = 0;
        }

        // Allocate GPU RAM
        float *deva, *devb, *devc;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&memcopystart);
        cudaEventCreate(&memcopystop);
        // Everything runs on stream 0
        cudaEventRecord(start);

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

        // Copy vectors to device
        cudaEventRecord(memcopystart);
        if (cudaMemcpy(deva, mata, size, cudaMemcpyHostToDevice) != cudaSuccess) {
            printf("deva to device failed.\n");
            return -1;
        }
        if (cudaMemcpy(devb, matb, size, cudaMemcpyHostToDevice) != cudaSuccess) {
            printf("devb to device failed.\n");
            return -1;
        }
        cudaEventRecord(memcopystop);
        cudaEventSynchronize(memcopystop);

        // Calculate blocksize and threadnumber
        int blocksizePerDim = (int) ceil(matrixSize / 32.0);
        int threadsPerDim = 32;

        printf("blocksizePerDim: %d\n", blocksizePerDim);
        printf("threadsPerDim: %d\n", threadsPerDim);

        // Kernel time!
		matrix_mul_gpu_element<<<dim3(blocksizePerDim, blocksizePerDim), dim3(threadsPerDim, threadsPerDim)>>>(deva, devb, devc, matrixSize);

        // Copy results back to host
        if (cudaMemcpy(matc, devc, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Device to host failed.\n");
            return -1;
        }

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        printf("GPU:\n");
        printf("matc[0] = %f\n", matc[0]);
        printf("matc[%d] = %f\n", matrixSize*matrixSize-1, matc[matrixSize*matrixSize-1]);

        // CPU time!
        double result_cpu;
        clock_t cputime = clock();

        matrixmul_CPU(mata, matb, matc, matrixSize);

        result_cpu = ((double)clock() - cputime);

        // Print results
        printf("CPU:\n");
        printf("matc[0] = %f\n", matc[0]);
        printf("matc[%d] = %f\n", matrixSize*matrixSize-1, matc[matrixSize*matrixSize-1]);

        cudaFree(deva);
        cudaFree(devb);
        cudaFree(devc);

        free(mata);
        free(matb);
        free(matc);

        /* End Program */

	    return_value = cudaGetLastError();
	    if(return_value != cudaSuccess){
	    	printf("Error in Kernel: %d\n", return_value);
	    	printf("%s\n", cudaGetErrorString(return_value));
	    	return -1;
	    }
	    cudaEventElapsedTime(&time, start, stop);
        printf("Time for CPU: %f s\n", result_cpu);
	    printf("Time for the kernel: %f s\n", time/1000);
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
