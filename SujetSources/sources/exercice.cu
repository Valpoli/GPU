#include "exercice.h"

const float Xmax = 1.5;
const float Xmin = -1.5;
const float Ymax = 1;
const float Ymin = -1;
#define BLOCK_SIZE 32

__device__ void map(int N, int M, int i, int j, float *a, float *b)
{
    float height = Ymax - Ymin;
    float width = Xmax - Xmin;
    *a = Xmin + (float(i) / float(N - 1)) * width;
    *b = Ymax - (float(j) / float(M - 1)) * height;
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
        if (sqrt(z*z +z_im*z_im)*sqrt(z*z +z_im*z_im) >= 2.0) {
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
        float a, b;
        map(N, M, i, j, &a, &b);
        float *pixel = get_ptr(img, i, j, C, pitch);
        if (is_converging(a, b)) {
            pixel[0] = 1;
        } else {
            pixel[0] = 0;
        }
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

__device__ float is_converging_greyscale(float a, float b)
{
    float zc = -0.5;
    float z_imc = 0.6;
    float z = a;
    float z_im = b;
    int i = 0;
    int iter = 100;
    while (i <= iter) {
        float tempz = z * z - z_im * z_im + zc;
        float tempz_im = 2.0 * z * z_im + z_imc;
        z = tempz;
        z_im = tempz_im;
        if (sqrt(z*z +z_im*z_im)*sqrt(z*z +z_im*z_im) >= 2.0) {
            break;
        }
        i += 1;
    }
    if (i > 100)
        i = 100;
    return (float)i/iter;
}

__global__
void kernel_generate2(int N, int M, int C, int pitch, float* img)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < M) {
        float a, b;
        map(N, M, i, j, &a, &b);
        float *pixel = get_ptr(img, i, j, C, pitch);
        pixel[0] = is_converging_greyscale(a, b);
    }
}

float* generate2(int N, int M, int C)
{
    size_t pitch;
    float* img;
    cudaMallocPitch(&img, &pitch, N * C * sizeof(float), M);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid_dim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    kernel_generate2<<<grid_dim, block_dim>>>(N,M,C,pitch,img);
    float* res =(float*) malloc(M * C * N * sizeof(float));
    cudaMemcpy2D(res, C * N * sizeof(float), img, pitch, C * N * sizeof(float), M, cudaMemcpyDeviceToHost);
    cudaFree(img);
    return res;
}

__device__ void color_from_ratio(float ratio, float* r, float* g, float* b)
{
    if (ratio >= 1.0f) {
        *r = 1.0f;
        *g = 1.0f;
	*b = 1.0f;
    } else if (ratio <= 0.5f) {
        *r = 0.0f;
        *g = ratio * 2.0f;
        *b = 1.0f - ratio * 2.0f;
    } else {
        *r = (ratio - 0.5f) * 2.0f;
        *g = 1.0f - (ratio - 0.5f) * 2.0f;
        *b = 0.0f;
    }
}

__global__
void kernel_generate3(int N, int M, int C, int pitch, float* img)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < M) {
        float a, b;
        map(N, M, i, j, &a, &b);
        float *pixel = get_ptr(img, i, j, C, pitch);
        float ratio = is_converging_greyscale(a, b);
        color_from_ratio(ratio, &pixel[0], &pixel[1], &pixel[2]);
    }
}

float* generate3(int N, int M, int C)
{
    size_t pitch;
    float* img;
    cudaMallocPitch(&img, &pitch, N * C * sizeof(float), M);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid_dim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    kernel_generate3<<<grid_dim, block_dim>>>(N,M,C,pitch,img);
    float* res =(float*) malloc(M * C * N * sizeof(float));
    cudaMemcpy2D(res, C * N * sizeof(float), img, pitch, C * N * sizeof(float), M, cudaMemcpyDeviceToHost);
    cudaFree(img);
    return res;
}

