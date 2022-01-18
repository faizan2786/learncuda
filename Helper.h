#pragma once
#include <stdio.h>

inline void handle_cuda_error(cudaError_t cudaStatus, const char *file, int line, bool abort = true)
{
    if (cudaStatus == cudaErrorIllegalAddress) {

        fprintf(stderr, "Error while accessing data in the kernel. Please make sure you are not accessing host memory in the kernel by accident!\n%s in %s at line no. %d\n", cudaGetErrorString(cudaStatus), file, line);
        if (abort)
            exit(cudaStatus);
    }

    if (cudaStatus != cudaSuccess) {

        fprintf(stderr, "Cuda Error: %s in %s at line no. %d\n", cudaGetErrorString(cudaStatus), file, line);
        if (abort)
            exit(cudaStatus);
    }
}

#define HANDLE_CUDA_ERROR(status) {handle_cuda_error(status, __FILE__, __LINE__);}