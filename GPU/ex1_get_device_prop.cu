#include <iostream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        std::cout << file << ':' << line << ": [CUDA ERROR] " << cudaGetErrorString(code) << std::endl; 
        std::abort();
    }
}

int main()
{
    int device_count;

    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::cout << "device count = " << device_count << std::endl;

    for(auto i = 0; i < device_count; ++i)
    {
        // ...
    }

    return 0;
}