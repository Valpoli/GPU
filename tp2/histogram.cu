#include "image.h"

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

#define BINS 32

__global__ void complet_hist_cpu(float *d_img, int size, int *d_hist)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        for (int i = 0; i < size ; ++i)
        {

            float pixel = d_img[i];
            if (pixel < 1)
            {
                int idx = pixel * BINS;
                atomicAdd(&d_hist[idx + blockIdx.x * BINS],1);
            }
        }
    }
}

__global__ void add_hist(int *d_hist, int block_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < block_count ; ++j)
    {
        d_hist[i] += d_hist[i + j * BINS];
    }
}

int main()
{
    int N, M, C;
    float* img = image::load("mandelbrot.jpg", &N, &M, &C, 1);
    const int size = N * M * C;
    int threads_per_block = 16;
    int block_count = (size + threads_per_block - 1)/ threads_per_block;

    float *d_img;
    cudaMalloc(&d_img, N * M * C * sizeof(float));
    cudaMemcpy(d_img, img, N * M * C * sizeof(float), cudaMemcpyHostToDevice);

    int *d_hist;
    cudaMalloc(&d_hist, BINS * block_count * sizeof(int));
    cudaMemset(d_hist, 0, BINS * block_count * sizeof(int));

    complet_hist_cpu<<<block_count, threads_per_block>>>(d_img,size,d_hist);
    add_hist<<<1, BINS>>>(d_hist,block_count);

    int hist[32] = {0};

    cudaMemcpy(hist, d_hist, BINS * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < BINS; ++i)
    {
        printf("%d\n",hist[i]);
    }

    cudaFree(d_img);
    cudaFree(d_hist);
    free(img);
    return 0;
}
