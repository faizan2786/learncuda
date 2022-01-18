#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<iostream>
#include <stdio.h>
#include <algorithm>

using namespace std;

const int N = 1024 * 60;			// array dimensions
const int threadsPerBlock = 1024;	// should be power of 2 to take advantage of reducing arrays for element wise sum in a block
const int MAX_BLOCKS = 64;

// calculate optimum number of Blocks per Grid
// so that c_partial is not very large for CPU 
// However, it has enough blocks to take advantage of the GPU
const int numBlocks = std::min(MAX_BLOCKS, (N + threadsPerBlock-1)/ threadsPerBlock);


// helper function to have dot product of two vectors a and b 
// store its partial result in a vector c_partial 
// where each element is partial dot product computed by each block
cudaError_t cudaDotProduct(float const *a_h, float const *b_h, float *c_partial_h);

// kernel declaration
__global__ void dotProductKernel(float* a_dev, float* b_dev, float* c_partial_dev);

template<class type>
type sum_of_squares(type n) {
	return (n * (n + 1) * (2 * n + 1) / 6);
}

int main(){

	float a[N], b[N];
	float c_partial[numBlocks];
	float dot_product = 0;

	// populate the input vectors
	for (int i = 0; i < N; ++i) {
		a[i] = i;
		b[i] = 2 * i;
	}

	auto status = cudaDotProduct(a, b, c_partial);
	if (status != cudaSuccess)
		return 0;

	for (int i=0; i< numBlocks; i++)
		dot_product += c_partial[i];

	auto sum_squared = 2*sum_of_squares(float(N - 1));

	if (abs(dot_product - sum_squared) < 1.0e-4f)
		printf("A dot B = %0.2f\n", dot_product);
	else
		cout << "dot product produced by cuda is not correct!\nPlease contact the developer." << endl;

	return 0;
}

__global__ void dotProductKernel(float* a_dev, float* b_dev, float* c_partial_dev) {

	__shared__ float cache[threadsPerBlock];   // shared variable is shared across all threads within a block (and have a separate copy for each different block)
	int totalThreads = threadsPerBlock * numBlocks;

	auto tid = threadIdx.x;  // id of the current thread (in the block)

	// elementwise products for each thread...
	float tempSum = 0;
	auto idx = blockDim.x * blockIdx.x + tid;  // first element id corresponding to the current thread
	while (idx < N) {
		tempSum += a_dev[idx] * b_dev[idx];
		idx += totalThreads;   // increment idx to process more than one elements per thread
	}

	cache[tid] = tempSum;

	__syncthreads();	// synchronize threads in this block (up to this point)

	// summing the work of each threads (by reducing arrays in half at a time) ...

	idx = threadsPerBlock / 2;
	while (idx > 0) {
		if (tid < idx)
			cache[tid] += cache[tid + idx]; // add product element from the other half

		__syncthreads();  // sync all threads before further reducing the array

		idx /= 2;
	}

	// store the final sum in c_partial array
	if (tid == 0)
		c_partial_dev[blockIdx.x] = cache[0];
}


cudaError_t cudaDotProduct(float const *a_h, float const *b_h, float* c_partial_h) {

	cudaError_t status;

	float *a_dev, *b_dev, *c_dev;
	float in_size = N * sizeof(float);
	float out_size = numBlocks * sizeof(float);

	// allocate device memory
	status = cudaMalloc(&a_dev, in_size);
	status = cudaMalloc(&b_dev, in_size);
	status = cudaMalloc(&c_dev, out_size);

	// copy host data to device
	status = cudaMemcpy(a_dev, a_h, in_size, cudaMemcpyHostToDevice);
	status = cudaMemcpy(b_dev, b_h, in_size, cudaMemcpyHostToDevice);

	dotProductKernel << <numBlocks, threadsPerBlock >> > (a_dev, b_dev, c_dev);

	// copy device data to host
	status = cudaMemcpy(c_partial_h, c_dev, out_size, cudaMemcpyDeviceToHost);   // blocks CPU until the kernel is finished

	if (status != cudaSuccess)
		cout << "Error in the cuda kernel!" << endl;

	cudaFree(a_dev);
	cudaFree(b_dev);
	cudaFree(c_dev);

	return status;
}

