#pragma once

#include <iostream>
#include <vector>

#define CUDA_CHECK(code) { __cuda_check((code), __FILE__, __LINE__); }
inline void __cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << file << ":" << line << ": " << cudaGetErrorString(code) << std::endl;
    }
}
