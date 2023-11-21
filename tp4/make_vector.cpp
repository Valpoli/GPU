#include "make_vector.h"

#include <random>
#include <algorithm>

std::vector<int> make_vector(int size)
{
    std::vector<int> matrix(size);
    std::generate(matrix.begin(), matrix.end(), [](){return float(std::rand())/RAND_MAX*200.f-100.f;});
    return matrix;
}