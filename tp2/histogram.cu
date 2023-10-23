#include "image.h"

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

#define BINS 32



int main()
{
    int N, M, C;
    float* img = image::load("mandelbrot.jpg", &N, &M, &C, 1);
    const int size = N * M * C;

    // ...

    return 0;
}
