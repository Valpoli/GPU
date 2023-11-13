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
    int i_res = bi * rows + i;
    int j_res = bj * cols + j;
    return (j_res + i_res * cols);
}

__global__ void matmul1(float *d_A, float *d_B, float *d_C, int N, int M, int P)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < P) {
        float result = 0.0f;

        for (int k = 0; k < M; ++k) {
            float element_A = d_A[index1(i, k, N, M)];
            float element_B = d_B[index1(k, j, M, P)];
            result += element_A * element_B;
        }

        d_C[index1(i, j, N, P)] = result;
    }
}


int main()
{
    // srand(time(nullptr));

    const int N = 64 * BLOCK_SIZE;
    const int M = 19 * BLOCK_SIZE;
    const int P = 12 * BLOCK_SIZE;

    const dim3 threads_per_block(BLOCK_SIZE,BLOCK_SIZE,1);
    const dim3 blocks((N + BLOCK_SIZE -1)/BLOCK_SIZE, (P + BLOCK_SIZE -1)/BLOCK_SIZE, 1);

    std::vector<float> A = make_matrix(N,M);
    std::vector<float> B = make_matrix(M,P);
    std::vector<float> C = matmul_cpu(A,B,N,M,P);


    float *d_A;
    float *d_B;
    float *d_C;
    float *C_GPU = (float*) malloc(sizeof(float) * N * P);

    cudaMalloc(&d_A, N * M * sizeof(float));
    cudaMalloc(&d_B, M * P * sizeof(float));
    cudaMalloc(&d_C, N * P * sizeof(float));
    cudaMemcpy(d_A, A.data(), N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), M * P * sizeof(float), cudaMemcpyHostToDevice);

    matmul1<<<blocks,threads_per_block>>>(d_A,d_B,d_C, N, M, P);

    cudaMemcpy(C_GPU, d_C, N * P * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> res = make_matrix(N,P);

    for (int i = 0; i < N * P; ++i) {
        res[i] = C_GPU[i];
    }

    float test = max_abs_diff(res, C);

    printf("%f\n", test);

    return 0;
}
