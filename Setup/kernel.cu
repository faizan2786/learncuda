#include "cudart_platform.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include <stdio.h>
#include<iostream>
#include<chrono>

#include "../Helper.h"

// helper function to get cuda device properties
void GetCudaDeviceProp() {

    int count;
    cudaDeviceProp prop;

    auto return_code = cudaGetDeviceCount(&count);
    std::cout << "Total CUDA Device count: " << count << "\n" << std::endl;

    for (int i = 0; i < count; i++) {
        auto return_code = cudaGetDeviceProperties(&prop, i);
        std::cout << "Devive-" << i << " Properties:" << std::endl;
        std::cout << "Name: " << prop.name << std::endl;
        std::cout << "Device Memory: " << std::round(((prop.totalGlobalMem / 1024.0) / 1024.0) / 1024.0)  << " GB"  << std::endl;
        std::cout << "Max. Threads per Block: " << prop.maxThreadsPerBlock <<  std::endl;
        std::cout << "\n";
    }  
}

// kernel definition (Device code)
__global__ void kernelVecAdd(float* A, float* B, float* C, int vecSize)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;  // since the vector is one-dimensional data, we can just use the x dimension of the block and thread to derive the necessary thread index
    if (i < vecSize)
        C[i] = A[i] + B[i];
}

// device helper function for vector add kernel
void deviceVectorAdd(float* A, float* B, float* C, int vecSize) {

    // Initialize device pointers.
    float *d_A, *d_B, *d_C;
    int num_bytes = vecSize * sizeof(float);
    
    // Allocate device memory.
    cudaMalloc(&d_A, num_bytes);
    cudaMalloc(&d_B, num_bytes);
    cudaMalloc(&d_C, num_bytes);

    // Transfer arrays a and b to device.
    HANDLE_CUDA_ERROR(cudaMemcpy(d_A, A, num_bytes, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_B, B, num_bytes, cudaMemcpyHostToDevice));
 
    // Calculate blocksize and gridsize.
    int threadsPerBlock = 1024;                        // block size
    int numBlocks = vecSize / threadsPerBlock + 1;     // grid size

    auto t1 = std::chrono::high_resolution_clock::now();
    // Launch CUDA kernel.
    kernelVecAdd <<< numBlocks, threadsPerBlock >>> (d_A, d_B, d_C, vecSize);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "Device time: " << time_span.count() << " micro sec" << std::endl;

    // Copy result array c back to host memory.
    HANDLE_CUDA_ERROR(cudaMemcpy(C, d_C, num_bytes, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}