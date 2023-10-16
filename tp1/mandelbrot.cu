#include "image.h"

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

int main()
{


    return 0;
}
