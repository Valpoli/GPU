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
    // ...
	return 0;
}


__global__
void process(int N, int M, int C, int pitch, float* img)
{
    // ...
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

    // ...

    image::save("result.jpg", N, M, C, img);

    free(img);

    return 0;
}
