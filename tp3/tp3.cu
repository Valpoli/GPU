#include <vector>
//#include <random>
#include <algorithm>
// #include <random>
#include <iostream>

#include "make_matrix.h"
#include <stdio.h>

#define BLOCK_SIZE 16

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

// create a row-major matrix of size (rows x cols) with random scalar numbers between -100 and +100
std::vector<float> make_matrix(int rows, int cols)
{
    std::vector<float> matrix(rows * cols);
    std::generate(matrix.begin(), matrix.end(), [](){return float(std::rand())/RAND_MAX*200.f-100.f;});
    return matrix;
}

// compute the maximal absolute difference between matrices A and B
float max_abs_diff(const std::vector<float>& A, const std::vector<float>& B)
{
    float res = 0;
    for(size_t i = 0; i < A.size(); ++i)
        res = std::max(res, std::abs(A.at(i) - B.at(i)));
    return res;
}



// return the 1D index of a row-major matrix of size (rows,cols) from indices (i,j)
__host__ __device__ int index1(int i, int j, int rows, int cols)
{
    return (j + i * cols);
}



// perform matrix multiplication C = A * B on the CPU
std::vector<float> matmul_cpu(const std::vector<float>& A, const std::vector<float>& B, int N, int M, int P)
{
    std::vector<float> res = make_matrix(N,P);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
	    int indexRes = index1(i,j,N,P);
	    res[indexRes] = 0;
            for (int k = 0; k < M; ++k) {
                int indexA = index1(i,k,N,M);
                int indexB = index1(k,j,M,P);
                res[indexRes] += A[indexA] * B[indexB];
            }
        }
    }
    return res;
}



// return the 1D index of a row-major matrix of size (rows,cols) from indices (i,j) inside sub-matrix (bi,bj)
__device__ int index2(int i, int j, int bi, int bj, int rows, int cols)
{
    // ...
}

// __global__ matmul cpu(float *d_img, int size, int *d_hist)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//     if (i < N && j < M) {

//     }
// }


int main()
{
    // srand(time(nullptr));

    const int N = 64 * BLOCK_SIZE;
    const int M = 19 * BLOCK_SIZE;
    const int P = 12 * BLOCK_SIZE;

    // int threads_per_block = P;
    // int block_count = M;

    std::vector<float> testA = make_matrix(2,3);
    std::vector<float> testB = make_matrix(3,4);

    float testa[] = {2,1,4,0,1,1};
    float testb[] = {6,3,-1,0,1,1,0,4,-2,5,0,2};

    for (int i = 0; i < 6; i++)
    {
        testA[i] = testa[i];
    }
    for (int i = 0; i < 12; i++)
    {
        testB[i] = testb[i];
    }

    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << testA[index1(i,j,2,3)] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Matrix B:" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << testB[index1(i,j,3,4)] << " ";
        }
        std::cout << std::endl;
    }

    const std::vector<float> res = matmul_cpu(testA,testB,2,3,4);

    std::cout << "res:" << std::endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << res[index1(i,j,2,4)] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
