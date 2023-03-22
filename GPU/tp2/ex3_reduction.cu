#include <iostream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

constexpr auto block_dim = 256;  // constexpr equivalent to blockDim.x in CUDA kernel
constexpr auto block_count = 256; // constexpr equivalent to gridDim.x in CUDA kernel



__global__ void dot(int n, const float *x, const float *y, float* device_result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float buffer[block_dim];
    for (int j = i; j < n; j += block_dim*block_count) {
        buffer[i] += y[j] * x[j];
    }
    __syncthreads();
    if (i == 0)
    {
        for (int k = 0; k < block_dim; k++){
            device_result[i] += buffer[k];
        }
    }
}

int main(int argc, char const *argv[])
{
    const int N = argc >= 2 ? std::stoi(argv[1]) : 1e6;
    std::cout << "N = " << N << std::endl;

    float *x, *y, *dx, *dy, *device_result, *dres;

    float host_expected_result = 0;
    float device_result = 0;

    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));
    device_result = (float*)malloc((N/(block_dim*block_count)) * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 2 * float(std::rand()) / RAND_MAX - 1; // random float in (-1,+1)
        y[i] = 2 * float(std::rand()) / RAND_MAX - 1; // random float in (-1,+1)
        host_expected_result += x[i] * y[i];
    }
    for (int i = 0; i < (N/(block_dim*block_count)) * sizeof(float); i++) {
        device_result[i] = 0;
    }


    CUDA_CHECK(cudaMalloc(&dx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dres, (N/(block_dim*block_count)) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dres, device_result, (N/(block_dim*block_count)) * sizeof(float), cudaMemcpyHostToDevice));

    dot<<<block_count, block_dim>>>(N,dx,dy,dres);
    
    CUDA_CHECK(cudaMemcpy(device_result, dres, (N/(block_dim*block_count)) * sizeof(float), cudaMemcpyDeviceToHost));


    std::cout << "host_expected_result = " << host_expected_result << std::endl;
    std::cout << "device_result = " << device_result << std::endl;


    cudaFree(dx);
    cudaFree(dy);
    free(x);
    free(y);
    free(device_result);
    
    return 0;
}