#include "image.h"

const float Xmax = 1;
const float Xmin = -2;
const float Ymax = 1;
const float Ymin = -1;

const int M = 960;
const int N = 640;
const int C = 1;

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
    *a = Xmin + (float(i) / float(N - 1)) * height;
    *b = Ymax - (float(j) / float(M - 1)) * width;
}


__device__ bool is_converging(float a, float b)
{
    float za0 = 0;
    float zb0 = 0;
    float tempZa = 0;
    float tempZb = 0;
    float za = 0;
    float zb = 0;
    int i = 0;

    while (i < 100)
    {
        tempZa = za;
        tempZb = zb;
        za = za0*za0 - zb0*zb0 + a;
        zb = 2*za0*zb0 + b;
        za0 = tempZa;
        zb0 = tempZb;
        i += 1;
    }
    float absz = sqrt(za*za +zb*zb);
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
    if (i < N && j < M) {
        float *pixel = get_ptr(img,i,j,C,pitch);
        float *a = (float*) malloc(sizeof(float));
        float *b = (float*) malloc(sizeof(float));
        map(N,M,i,j,a,b);
        if (is_converging(*a,*b))
        {   
            pixel[0] = 0;
        }
        else
        {
            pixel[0] = 1;
        }
        free(a);
        free(b);
    }
}

int main(int argc, char const *argv[])
{
    size_t pitch;

    float* img;
    CUDA_CHECK(cudaMallocPitch(&img, &pitch, N * C * sizeof(float), M));
    dim3 block_dim(32, 32, 1);
    dim3 grid_dim((N + 32 - 1) / 32, (M + 32 - 1) / 32, 1);
    kernel<<<grid_dim, block_dim>>>(img, N,M,pitch);
    float* res =(float*) malloc(M * C * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy2D(res, C * N * sizeof(float), img, pitch, C * N * sizeof(float), M, cudaMemcpyDeviceToHost));
    image::save("result.jpg", N, M, C, res);

    cudaFree(img);
    free(res);

    return 0;
}
