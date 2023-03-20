#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 16

// Device function to get the pointer to the pixel (i,j)
__device__ float* get_ptr(float* img, int i, int j, int C, size_t pitch)
{
    return (float*)((char*)img + i * pitch) + j * C;
}

// CUDA kernel to modify a pixel
__global__ void process(int N, int M, int C, size_t pitch, float* img)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < M) {
        float* pixel = get_ptr(img, i, j, C, pitch);
        float k = (*pixel + *(pixel + 1) + *(pixel + 2)) / 3.0f;
        *pixel = k;
        *(pixel + 1) = k;
        *(pixel + 2) = k;
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Load the image into a float array on the host
    int N, M, C;
    float* h_img = NULL;
    size_t h_pitch;
    FILE* fp = fopen(argv[1], "rb");
    if (fp == NULL) {
        printf("Error: could not open file %s\n", argv[1]);
        exit(EXIT_FAILURE);
    }
    fread(&N, sizeof(int), 1, fp);
    fread(&M, sizeof(int), 1, fp);
    fread(&C, sizeof(int), 1, fp);
    h_pitch = M * C * sizeof(float);
    h_img = (float*)malloc(N * h_pitch);
    for (int i = 0; i < N; i++) {
        float* row = (float*)((char*)h_img + i * h_pitch);
        fread(row, sizeof(float), M * C, fp);
    }
    fclose(fp);

    // Allocate memory for the image on the device
    float* d_img = NULL;
    size_t d_pitch;
    cudaMallocPitch(&d_img, &d_pitch, M * C * sizeof(float), N);

    // Copy the image data from the host to the device
    cudaMemcpy2D(d_img, d_pitch, h_img, h_pitch, M * C * sizeof(float), N, cudaMemcpyHostToDevice);

    // Define the grid and block dimensions for the kernel launch
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((M + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM);

    // Launch the kernel
    process<<<gridDim, blockDim>>>(N, M, C, d_pitch, d_img);

    // Copy the modified image data from the device to the host
    cudaMemcpy2D(h_img, h_pitch, d_img, d_pitch, M * C * sizeof(float), N, cudaMemcpyDeviceToHost);

    // Save the modified image on the host
    fp = fopen(argv[2], "wb");
    if (fp == NULL) {
        printf("Error: could not open file %s\n", argv[2]);
        exit(EXIT_FAILURE);
    }
    fwrite(&N, sizeof(int), 1, fp);
    fwrite(&M, sizeof(int), 1, fp);
fwrite(&C, sizeof(int), 1, fp);
for (int i = 0; i < N; i++) {
    float* row = (float*)((char*)h_img + i * h_pitch);
    fwrite(row, sizeof(float), M * C, fp);
}
fclose(fp);

// Free the memory
free(h_img);
cudaFree(d_img);

return 0;

