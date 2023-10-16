#include <iostream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

constexpr auto block_dim = 256;  // 256 constexpr equivalent to blockDim.x in CUDA kernel
constexpr auto block_count = 256; // 256 constexpr equivalent to gridDim.x in CUDA kernel

__global__ void dot(int n, const float *x, const float *y, float* res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float buffer[block_dim];
    buffer[threadIdx.x] = 0;
    for (int j = i; j < n; j += block_dim*block_count) {
        buffer[threadIdx.x] += y[j] * x[j];
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        for (int k = 0; k < block_dim; k++){
            res[blockIdx.x] += buffer[k];
        }
    }
}

__global__ void dot2(int n, const float *x, const float *y, float* res2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float buffer[block_dim];
    buffer[threadIdx.x] = 0;
    for (int j = i; j < n; j += block_dim*block_count) {
        buffer[threadIdx.x] += y[j] * x[j];
    }
    __syncthreads();
    for (int k = block_dim/2; k >= 1; k/=2)
    {
        if (threadIdx.x < k)
        {
            buffer[threadIdx.x] += buffer[threadIdx.x+k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        res2[blockIdx.x] = buffer[0];
    }
}

int main(int argc, char const *argv[])
{
    const int N = argc >= 2 ? std::stoi(argv[1]) : 1e6;
    std::cout << "N = " << N << std::endl;

    float *x, *y, *dx, *dy, *res, *dres;

    float host_expected_result = 0;
    float device_result = 0;

    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));
    res = (float*)malloc(block_count * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 2 * float(std::rand()) / RAND_MAX - 1; // random float in (-1,+1)
        y[i] = 2 * float(std::rand()) / RAND_MAX - 1; // random float in (-1,+1)
        host_expected_result += x[i] * y[i];
        //printf("on fait la multiplication %f * %f = %f et le total est %f\n",y[i],x[i],y[i] * x[i], host_expected_result);
    }

    CUDA_CHECK(cudaMalloc(&dx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dres, block_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dres, res, block_count * sizeof(float), cudaMemcpyHostToDevice));

    dot<<<block_count, block_dim>>>(N,dx,dy,dres);
    
    CUDA_CHECK(cudaMemcpy(res, dres, block_count * sizeof(float), cudaMemcpyDeviceToHost));

    int m = 0;
    while( m < block_count) {
        device_result += res[m];
        m += 1;
    }

    std::cout << "host_expected_result = " << host_expected_result << std::endl;
    std::cout << "device_result = " << device_result << std::endl;

    // DOT 2
    float device_result_dot2 = 0;
    for (int i = 0; i < block_count; i++) {
        res[i] = 0;
    }
    CUDA_CHECK(cudaMemcpy(dres, res, block_count * sizeof(float), cudaMemcpyHostToDevice));
    dot2<<<block_count, block_dim>>>(N,dx,dy,dres);
    CUDA_CHECK(cudaMemcpy(res, dres, block_count * sizeof(float), cudaMemcpyDeviceToHost));

    m = 0;
    while( m < block_count) {
        device_result_dot2 += res[m];
        m += 1;
    }

    std::cout << "device_result_dot2 = " << device_result_dot2 << std::endl;

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dres);
    free(x);
    free(y);
    free(res);
    
    return 0;
}