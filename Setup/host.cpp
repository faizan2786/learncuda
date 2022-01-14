//#include "kernel.cu"
#include<iostream>
#include<chrono>


// kernel helper declarations
void GetCudaDeviceProp();
void deviceVectorAdd(float* A, float* B, float* C, int vecSize);

// cpu only vector addition
void hostVectorAdd(float* A, float* B, float* C, int vecSize);

// Host code (driver code)
int main() {

    // device property check...
    GetCudaDeviceProp();

	std::cout << "Cuda runtime code..." << std::endl;

    const int vecSize = 3;
    // Initialize arrays A, B, and C.
    float h_A[vecSize], h_B[vecSize], h_C[vecSize];

    // Populate arrays A and B.
    h_A[0] = 5; h_A[1] = 8; h_A[2] = 3;
    h_B[0] = 7; h_B[1] = 6; h_B[2] = 4;

    // Sum array elements across ( C[0] = A[0] + B[0] ) into array C using CUDA device
    deviceVectorAdd(h_A, h_B, h_C, vecSize);

    // Print out result.
    std::cout << "C = " << h_C[0] << ", " << h_C[1] << ", " << h_C[2] << std::endl;
	return 0;
}

// time comparison
int main1() {
    std::cout << "Cuda runtime code..." << std::endl;

    const int vecSize = 30000;
    // Initialize arrays A, B, and C.
    float h_A[vecSize], h_B[vecSize], h_C[vecSize];

    // Populate arrays A and B.
    for (int i = 0; i < vecSize; ++i) {
        h_A[i] = i;
        h_B[i] = 10;
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    hostVectorAdd(h_A, h_B, h_C, vecSize);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "Host time: " << time_span.count() << " micro sec\n" << std::endl;

    // Sum array elements across ( C[0] = A[0] + B[0] ) into array C using CUDA device
    deviceVectorAdd(h_A, h_B, h_C, vecSize);               

    // Print out result.
    std::cout << "C = " << h_C[0] << ", " << h_C[1] << ", " << h_C[2] << std::endl;
    return 0;
}


// cpu only vector addition
void hostVectorAdd(float* A, float* B, float* C, int vecSize) {
    for (int i = 0; i < vecSize; ++i)
        C[i] = A[i] + B[i];  // C[i] = A[i] * A[i] * A[i] * A[i] * A[i] * A[i] * A[i] * A[i] * A[i] * A[i] + B[i] * B[i] * B[i];   
}