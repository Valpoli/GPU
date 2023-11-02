#include "image.h"

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

#define BINS 32

void complet_hist_cpu(float *img, int N, int M, int C, int *hist)
{
    for (int i = 0; i < N * M * C; ++i)
    {
        float pixel = img[i];
        if (pixel < 1)
        {
            int idx = pixel * BINS
            ++ hist[idx]
        }
    }
}

int main()
{
    int N, M, C;
    float* img = image::load("mandelbrot.jpg", &N, &M, &C, 1);
    const int size = N * M * C;

    int *hist = (int) malloc(sizeof(int) * BINS);
    complet_hist_cpu(img, N, M, C, hist);
    for (int i = 0; i < N * M * C; ++i)
    {
        cout << hist[i] << endl;
    }
    free(hist)

    return 0;
}
