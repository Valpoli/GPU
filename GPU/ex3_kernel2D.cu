#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10
#define M 20

template<typename T>
__device__ T* get_ptr(T* start, int i, int j, size_t pitch) {
    return reinterpret_cast<T*>(reinterpret_cast<char*>(start) + i * pitch + j * sizeof(T));
}

__global__ void add(int n, int m, int pitch, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < m) {
        float *px = get_ptr(x, i, j, pitch);
        float *py = get_ptr(y, i, j, pitch);
        *px += *py;
    }
}

int main(void) {
    float *h_a, *h_b, *d_a, *d_b;
    size_t pitch;
    
    // allocate host memory
    h_a = (float*) malloc(N * M * sizeof(float));
    h_b = (float*) malloc(N * M * sizeof(float));
    
    // initialize host arrays
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            h_a[i * M + j] = i + j;
            h_b[i * M + j] = i - j;
        }
    }
    
    // allocate device memory with pitch
    cudaMallocPitch(&d_a, &pitch, M * sizeof(float), N);
    cudaMallocPitch(&d_b, &pitch, M * sizeof(float), N);
    
    // copy host memory to device memory
    cudaMemcpy2D(d_a, pitch, h_a, M * sizeof(float), M * sizeof(float), N, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_b, pitch, h_b, M * sizeof(float), M * sizeof(float), N, cudaMemcpyHostToDevice);
    
    // launch kernel
    dim3 block_dim(32, 32);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x, (M + block_dim.y - 1) / block_dim.y);
    add<<<grid_dim, block_dim>>>(N, M, pitch, d_a, d_b);
    
    // copy device memory back to host memory
    cudaMemcpy2D(h_a, M * sizeof(float), d_a, pitch, M * sizeof(float), N, cudaMemcpyDeviceToHost);
    
    // print result
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%f ", h_a[i * M + j]);
        }
        printf("\n");
    }
    
    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    
    return 0;
}
