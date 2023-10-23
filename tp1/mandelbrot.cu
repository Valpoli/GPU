#include "image.h"

const float Xmax = 1
const float Xmin = -2
const float Ymax = 1
const float Ymin = -1

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

template <typename T>
__device__ inline T* get_ptr(T *img, int i, int j, int C, size_t pitch) {
	return (T*)((char*)img + pitch * j + i * C * sizeof(T));
}

__device__ void map(int N, int M, int i, int j, float *a, float *b)
{
    int height = Xmax - Xmin;
    int width = Ymax - Ymin;
    *a = (static_cast<float>i/N-1) * height;
    *b = (static_cast<float>j/M-1) * width;
}


// #initialser la série a 0 et on itere jusqua voir la convergence
__device__ bool is_converging(float a, float b)
{
    float za0 = 0;
    float zb0 = 0;
    float tempZa = 0;
    float tempZb = 0
    float za = a;
    float zb = b;
    int i = 0;
    while (i < 100)
    {
        tempZa = za;
        tempZb = zb;
        za = za0^2 - zb0^2;
        zb = 2*za0*zb0;
        za0 = tempZa;
        zb0 = tempZb
        i += 1;
    }
    absz = sqrt(za^2+zb^2)
    if (absz < 1)
    {
        return true;
    }
    return false;
}



__global__ void kernel (float *img, int N, int M, size_t pitch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        float* pixel = get_ptr(img,i,j,C,pitch);
        float *a;
        float *b;
        map(N,M i,j, a ,b);
        if (is_converging(a,b))
        {   
            pixel[0] = 0;
        }
        else
        {
            pixel[0] = 1;
        }
    }
}

int main(int argc, char const *argv[])
{
    const std::string filename = argc >= 2 ? argv[1] : "image.jpg";
    std::cout << "filename = " << filename << std::endl;
    int M = 960;
    int N = 640;
    int C = 1;
    std::cout << "N (columns, width) = " << N << std::endl;
    std::cout << "M (rows, height) = " << M << std::endl;
    std::cout << "C (channels, depth) = " << C << std::endl;

    size_t pitch;

    float* img;
    // CUDA_CHECK(cudaMallocPitch(&img, &pitch, N * C * sizeof(float), M));
    // dim3 block_dim(32, 32, 1);
    // dim3 grid_dim((M + block_dim.x - 1) / block_dim.x, (N + block_dim.y - 1) / block_dim.y);
    // kernel<<<grid_dim, block_dim>>>(cpy, N,M,pitch);
    // float* res = malloc(N * C * M * sizeof(float))
    // CUDA_CHECK(cudaMemcpy2D(res, C * N * sizeof(float), img, pitch, C * N * sizeof(float), M, cudaMemcpyDeviceToHost));
    // image::save("result.jpg", N, M, C, res);

    // cudaFree(img);
    // free(res);

    return 0;
}
