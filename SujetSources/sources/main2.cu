#include "exercice.h"

int main()
{
    const int N = 3 * 320;
    const int M = 2 * 320;
    const int C = 1;
    
    float* img = generate2(N, M, C);
    
    image::save("image2.jpg", N, M, C, img);
    free(img);

    return 0;
}
