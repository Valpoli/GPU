#include <vector>
#include <random>
#include <algorithm>

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
    // ...
}



// perform matrix multiplication C = A * B on the CPU
std::vector<float> matmul_cpu(const std::vector<float>& A, const std::vector<float>& B, int N, int M, int P)
{
    // ...
}



// return the 1D index of a row-major matrix of size (rows,cols) from indices (i,j) inside sub-matrix (bi,bj)
__device__ int index2(int i, int j, int bi, int bj, int rows, int cols)
{
    // ...
}



int main()
{
    // srand(time(nullptr));

    const int N = 64 * BLOCK_SIZE;
    const int M = 19 * BLOCK_SIZE;
    const int P = 12 * BLOCK_SIZE;

    const std::vector<float> A = make_matrix(N,M);
    const std::vector<float> B = make_matrix(M,P);

    // ...

    return 0;
}
