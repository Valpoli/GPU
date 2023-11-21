#include "make_vector.h"
#include <iostream>

#define STATIC_SIZE 64

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

std::vector<int> scan_exclu(std::vector<int> table)
{
    std::size_t size = table.size();
    std::vector<int> res = make_vector((int) size);
    res[0] = 0;
    for (std::size_t i = 1; i < size; i++)
    {
        res[i] = res[i-1] + table[i-1];
    }
    return res;
}





int main()
{
    // srand(time(nullptr));

    constexpr int N = STATIC_SIZE;
    const std::vector<int> x = make_vector(N);

    int size_test = 8;
    std::vector<int> test = {3,2,5,6,8,7,4,1};
    std::vector<int> test_exclu = scan_exclu(test);
    std::cout << "Contenu du vecteur :";
    for (int value : test_exclu) {
        std::cout << " " << value;
    }
    std::cout << std::endl;
    return 0;
}
