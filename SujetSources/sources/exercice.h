#pragma once

#include "common.h"
#include "image.h"

#define BLOCK_SIZE 32

// return the address of the first component of pixel (i,j) 
// in the image of C components with pitch bytes
template <typename T>
__device__ inline T* get_ptr(T *img, int i, int j, int C, size_t pitch) {
	return (T*)((char*)img + pitch * j + i * C * sizeof(T));
}


__global__
void kernel_generate1(int N, int M, int C, int pitch, float* img);

float* generate1(int N, int M, int C);


__global__
void kernel_generate2(int N, int M, int C, int pitch, float* img);

float* generate2(int N, int M, int C);


__global__
void kernel_generate3(int N, int M, int C, int pitch, float* img);

float* generate3(int N, int M, int C);