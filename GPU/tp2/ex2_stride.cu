#include <iostream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

__global__ void add(int n, float *x, float *y, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = i; j < n; j += stride) {
        y[j] += x[j];
    }
}

int main(int argc, char const *argv[])
{
    const int N = argc >= 2 ? std::stoi(argv[1]) : 1e6;
    std::cout << "N = " << N << std::endl;

    float *x, *y, *dx, *dy;
    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    CUDA_CHECK(cudaMalloc(&dx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice));

    int nbThreadBlock = 512;
    int threadByBlock = 128;
    int stride = 512*128;
    add<<<nbThreadBlock,threadByBlock>>>(N, dx, dy, stride);
    CUDA_CHECK(cudaMemcpy(y, dy, N * sizeof(float), cudaMemcpyDeviceToHost));

    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    free(x);
    free(y);

    cudaFree(dx);
    cudaFree(dy);

    return 0;
}