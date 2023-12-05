#include "make_vector.h"
#include <iostream>

#define STATIC_SIZE 64

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

std::vector<int> scan_exclu(std::vector<int> table)
{
    std::size_t size = table.size();
    std::vector<int> res = make_vector((int) size);
    res[0] = 0;
    for (std::size_t i = 1; i < size; ++i)
    {
        res[i] = res[i-1] + table[i-1];
    }
    return res;
}

__global__ void scan1(int* x, int N)
{
    __shared__ int buffers[2*8];
    int in = 0;
    int out = N;
    int i = threadIdx.x;
    buffers[i] = x[i];
    __syncthreads();
    if (i == 0)
    {
        x[i] = 0;
    }
    else
    {
        int step = (int) log2(static_cast<float>(N));
        for (int n = 0; n < step; ++n)
        {
            int offset = pow(2, n) ;
            if (offset <= i)
            {
                buffers[i + out] = buffers[i + in] + buffers[i + in - offset];
            }
            else
            {
                buffers[i + out] = buffers[i + in];
            }
            __syncthreads();
            int temp = in;
            in = out;
            out = in;
        }
        x[i] = buffers[i + out];
    }
}



int main()
{
    // srand(time(nullptr));
    int size_test = 8;
    constexpr int N = STATIC_SIZE;
    // const std::vector<int> x = make_vector(N);
    const dim3 threads_per_block(size_test,1,1);
    const dim3 blocks(1,1,1);

    std::vector<int> test = {3,2,5,6,8,7,4,1};
    std::cout << "Contenu du vecteur :";
    for (int value : test) {
        std::cout << " " << value;
    }
    std::cout << std::endl;
    // std::vector<int> test_exclu = scan_exclu(test);
    int* d_x;
    cudaMalloc(&d_x, size_test * sizeof(int));
    cudaMemcpy(d_x, test.data(), size_test * sizeof(int), cudaMemcpyHostToDevice);
    scan1<<<blocks,threads_per_block>>>(d_x, N);
    int *x = (int*) malloc(size_test * sizeof(int));
    cudaMemcpy(x, d_x, size_test * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Contenu du vecteur :";
    for (int n = 0; n < size_test; ++n){
        std::cout << " " << x[n];
    }
    std::cout << std::endl;
    free(x);
    cudaFree(d_x);
    return 0;
}