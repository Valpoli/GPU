#include "exercice.h"

int main()
{
    const int N = 3 * 320;
    const int M = 2 * 320;
    const int C = 3;
    
    float* img = generate3(N, M, C);
    
    image::save("image3.jpg", N, M, C, img);
    free(img);

    return 0;
}
