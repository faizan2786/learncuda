#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>

#include "fpBitmap.h"
#include "cuComplex.h"

#define DIM 1000

__global__ void kernel(unsigned char* d_ptr);
__host__ __device__  int julia(int x, int y);	// function to determine if pixel position belongs to julia set. returns 1 or 0
void cpu_kernel(unsigned char* ptr);

// CPU Julia test
int main() {

	fpBitmap bitmap(DIM, DIM);
	auto img_ptr = bitmap.get_image_ptr();

	// test on CPU...
	/*auto t1 = std::chrono::high_resolution_clock::now();
	cpu_kernel(img_ptr);
	auto t2 = std::chrono::high_resolution_clock::now();

	auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
	std::cout << "Host time: " << time_span.count()/1000.0 << "ms" << std::endl;

	auto test_val_cpu = bitmap(514, 220)[0];
	std::cout << "Host test value: " << (int)test_val_cpu << std::endl << std::endl;*/

	// test on GPU...
	fpBitmap h_bitmap(DIM, DIM);   // bitmap on host
	auto h_ptr = h_bitmap.get_image_ptr();

	unsigned char* d_bitmap;   // bitmap on device

	auto d_t1 = std::chrono::high_resolution_clock::now();

	auto status = cudaMalloc(&d_bitmap, h_bitmap.get_image_size());  // allocate bitmap memory on device
	dim3 threadsPerBlock = 1; // (1024, 1024);
	dim3 numBlocks(DIM, DIM); // (1, 1);  // have the grid size same as the image size for 1-to-1 indexing

	kernel << <numBlocks, threadsPerBlock >> > (d_bitmap);
	status = cudaMemcpy(h_ptr, d_bitmap, h_bitmap.get_image_size(), cudaMemcpyDeviceToHost); 	// copy bitmap back to the host
	if (status == cudaErrorIllegalAddress)
		std::cout << "Error while accessing data in the kernel!" << std::endl;

	cudaFree(d_bitmap);

	auto d_t2 = std::chrono::high_resolution_clock::now();
	auto d_time_span = std::chrono::duration_cast<std::chrono::microseconds>(d_t2 - d_t1);
	std::cout << "Device time: " << d_time_span.count() / 1000.0 << "ms" << std::endl;

	auto test_val_gpu = h_bitmap(514, 220)[0];
	std::cout << "Device test value: " << (int)test_val_gpu << std::endl;

	return 0;
}

void cpu_kernel(unsigned char* ptr) {

	for (int y = 0; y < DIM; y++) {
		for (int x = 0; x < DIM; x++) {

			int idx = (x + y * DIM) * 3;
			int juliaValue = julia(x, y);
			ptr[idx + 0] = 255 * juliaValue;  // set Red value if the pixel belongs to julia set
			ptr[idx + 1] = 0;
			ptr[idx + 2] = 0;

		}
	}
}

// Device kernel
__global__ void kernel(unsigned char* d_ptr) {

	int x_idx = blockIdx.x; // *blockDim.x + threadIdx.x;    // column index
	int y_idx = blockIdx.y; // *blockDim.y + threadIdx.y;	  // row index

	if (x_idx < DIM && y_idx < DIM) {
		auto offset = y_idx * DIM + x_idx;
		int juliaValue = julia(x_idx, y_idx);  // make sure this function and all the externel functions / objects used inside this function are declared with the __device__ qualifier

		d_ptr[offset * 3 + 0] = 255 * juliaValue;  // set Red value if the pixel belongs to julia set
		d_ptr[offset * 3 + 1] = 0;
		d_ptr[offset * 3 + 2] = 0;
	}
}

// this function is defined for both host and device to be able to use it from both
__host__ __device__ int julia(int x, int y) {

	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(-0.8, 0.156);   // **
	cuComplex a(jx, jy);		// **

	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a * a + c;              // **
		if (a.magnitude2() > 1000)	// **
			return 0;
	}
	return 1;

	//** since this is a device function too,
	// we need to make sure that all custom types (class / struct) used in this function defines their member functions with __host__ and __device__ qualifiers
	// otherwise we won't be able to use it from a device code
	// NOTE: In this case compiler won't even give any error since this function is also declared as a __host__ function 
	//		 instead cuda will return some error status which won't be easily traceable
}

// bitmap class test
int main_bitmap_test() {

	int dim_x = 2, dim_y = 2;
	fpBitmap bitmap(dim_x, dim_y);
	auto ptr = bitmap.get_image_ptr();

	for (int i = 0; i < dim_x; ++i) {
		for (int j = 0; j < dim_y; ++j) {

			auto idx = (j * dim_x + i) * 3;
			bitmap[idx] = j * 50 + i;
			bitmap[idx + 1] = j * 50 + 2 * i;
			bitmap[idx + 2] = j * 50 + 3 * i;
		}
	}

	auto size = bitmap.get_image_size();
	auto val = ptr[1];
	auto pixel = bitmap(1, 1)[2];    // should be 53

	return 0;
}