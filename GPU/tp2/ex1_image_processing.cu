#include <string>
#include <iostream>
#include "image.h"

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

template <typename T>
__device__ inline T* get_ptr(T *img, int i, int j, int C, size_t pitch) {
    return img + i * pitch / sizeof(float) + j * C;
}

__global__ void process(int N, int M, int C, int pitch, float* img)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        float* pixel = get_ptr(img,i,j,C,pitch);
        float newColor = 0;
        for (int k=0; k<C; k+=1)
        {
            newColor += pixel[k];
        }
        newColor =  newColor/C;
        for (int k=0; k<C; k+=1)
        {
            pixel[k] = newColor;
        }
    }
}

int main(int argc, char const *argv[])
{
    const std::string filename = argc >= 2 ? argv[1] : "image.jpg";
    std::cout << "filename = " << filename << std::endl;
    int M = 0;
    int N = 0;
    int C = 0;
    float* img = image::load(filename, &N, &M, &C);
    std::cout << "N (columns, width) = " << N << std::endl;
    std::cout << "M (rows, height) = " << M << std::endl;
    std::cout << "C (channels, depth) = " << C << std::endl;

    size_t cpyPitch;

    float* cpy;
    //CUDA_CHECK(cudaMallocPitch(&cpy, &pitch, N * sizeof(float), M));
    //CUDA_CHECK(cudaMemcpy2D(cpy, pitch, img, N * sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice));

    cudaMallocPitch(&cpy, &cpyPitch, M * C * sizeof(float), N);

    // Copy the image data from the host to the device
    cudaMemcpy2D(cpy, cpyPitch, img, N * sizeof(float), M * C * sizeof(float), N, cudaMemcpyHostToDevice);
    
    // launch kernel
    dim3 block_dim(32, 32);
    dim3 grid_dim((M + block_dim.x - 1) / block_dim.x, (N + block_dim.y - 1) / block_dim.y);
    process<<<grid_dim, block_dim>>>(N,M,C,cpyPitch,cpy);
    
    // copy device memory back to host memory
    CUDA_CHECK(cudaMemcpy2D(img, N * sizeof(float) , cpy, cpyPitch, M * C * sizeof(float), N, cudaMemcpyDeviceToHost));
    image::save("result.jpg", N, M, C, img);

    cudaFree(cpy);
    free(img);

    return 0;
}
