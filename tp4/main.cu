#include "make_vector.h"

#include <iostream>
#include <stdio.h>

#define STATIC_SIZE 64

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}


std::vector<int> scan_exclusif(std::vector<int> vec)
{
	std::vector<int> res(vec.size());
	res[0] = 0;
	for (int i = 1; i < vec.size(); i++)
	{
		res[i] = res[i-1] + vec[i-1];
	}

	return res;
}

__global__ void scan1(int* x, int N)
{
	int i = threadIdx.x;

	__syncthreads();

	int step = (int) log2f(static_cast<float>(N));

	for (int n = 0; n < step; n++)
	{
		int offset = pow(2, n);

		if ( n == 0 || i <= (8/2)/(2*n) )
		{
			int index = i + (n * 2 - 1) + offset;
			x[index] = x[index] + x[index - offset]; 
		}
		__syncthreads();
	}
}


int main()
{
	std::vector<int> vector1 = {3, 2, 5, 6, 8, 7, 4, 1};
	std::vector<int> res = scan_exclusif(vector1);
	for (int i = 0; i < res.size() ; i++)
	{
		std::cout << res[i] << " ";
	}
	std::cout << std::endl;
	int *d_x;
	int *x = (int*) malloc(sizeof(int) *8);
	CUDA_CHECK(cudaMalloc(&d_x, 8 *sizeof(int)));
	CUDA_CHECK(cudaMemcpy(d_x, vector1.data(), 8 * sizeof(int), cudaMemcpyHostToDevice));

	dim3 nb_block = { 1, 1, 1};
	dim3 nb_thread_per_block = { 4, 1, 1};

	scan1<<<nb_block, nb_thread_per_block>>>(d_x, 8);
	
	CUDA_CHECK(cudaMemcpy(x, d_x, 8 * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < 8 ; i++)
	{
		std::cout << x[i] << " ";
	}
	std::cout << std::endl;
	cudaFree(d_x);
	free(x);

    return 0;
}
