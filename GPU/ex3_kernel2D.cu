#include <string>
#include <iostream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

int main(int argc, char const *argv[])
{
    const int N = argc >= 2 ? std::stoi(argv[1]) : 1e3;
    const int M = argc >= 3 ? std::stoi(argv[2]) : 1e3;
    std::cout << "N (columns) = " << N << std::endl;
    std::cout << "M (rows) = " << M << std::endl;

    // ...

    size_t pitch = 0;

    // ...

    return 0;
}
