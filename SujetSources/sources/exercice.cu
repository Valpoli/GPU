#include "exercice.h"

const float Xmax = 1.5;
const float Xmin = -1.5;
const float Ymax = 1;
const float Ymin = -1;
#define BLOCK_SIZE 32

__device__ void map(int N, int M, int i, int j, float *a, float *b)
{
    int height = Xmax - Xmin;
    int width = Ymax - Ymin;
    *a = Xmin + (float(i) / float(N - 1)) * height;
    *b = Ymax - (float(j) / float(M - 1)) * width;
}

__device__ bool is_converging(float a, float b)
{
    float zc = -0.5;
    float z_imc = 0.6;
    float z = a;
    float z_im = b;
    int i = 0;
    while (i <= 100) {
        float tempz = z * z - z_im * z_im + zc;
        float tempz_im = 2.0 * z * z_im + z_imc;
        z = tempz;
        z_im = tempz_im;
        if (z * z + z_im * z_im >= 4.0) {
            return false;
        }
        i += 1;
    }
    return true;
}

__global__
void kernel_generate1(int N, int M, int C, int pitch, float* img)
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
            pixel[0] = 1;
        }
        else
        {
            pixel[0] = 0;
        }
        free(a);
        free(b);
    }
}

float* generate1(int N, int M, int C)
{
    size_t pitch;
    float* img;
    cudaMallocPitch(&img, &pitch, N * C * sizeof(float), M);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid_dim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    kernel_generate1<<<grid_dim, block_dim>>>(N,M,C,pitch,img);
    float* res =(float*) malloc(M * C * N * sizeof(float));
    cudaMemcpy2D(res, C * N * sizeof(float), img, pitch, C * N * sizeof(float), M, cudaMemcpyDeviceToHost);
    cudaFree(img);
    return res;
}

__global__
void kernel_generate2(int N, int M, int C, int pitch, float* img)
{
}

float* generate2(int N, int M, int C)
{
    return nullptr;
}



__global__
void kernel_generate3(int N, int M, int C, int pitch, float* img)
{
}

float* generate3(int N, int M, int C)
{
    return nullptr;
}
