#include <iostream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

constexpr auto block_dim = 2;  // 256 constexpr equivalent to blockDim.x in CUDA kernel
constexpr auto block_count = 2; // 256 constexpr equivalent to gridDim.x in CUDA kernel



__global__ void dot(int n, const float *x, const float *y, float* res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float buffer[block_dim];
    buffer[blockDim.x] = 0;
    for (int j = i; j < n; j += block_dim*block_count) {
        buffer[blockDim.x] += y[j] * x[j];
        //printf("on fait la multiplication %f * %f = %f et le buffer egal a %f dans le bloc %d\n",y[j],x[j],y[j] * x[j], buffer[blockIdx.x], blockIdx.x);
    }
    //printf("this is the block %d and the total for it is %f\n",blockIdx.x * blockDim.x + threadIdx.x,buffer[blockIdx.x * blockDim.x + threadIdx.x]);
    __syncthreads();
    if (i == 0)
    {
        for (int k = 0; k < block_dim; k++){
            printf("%d\n", buffer[k]);
            //res[0] += buffer[k];
        }
        /*for (int k = 0; k < 4; k++){
            printf("%f,",res[k]);
        }
        printf("\n");*/
    }

    //printf("LE RESULTAT POUR CE BLOC %d EST : %f, PAS MAL, N'EST CE PAS ?????\n",blockIdx.x * blockDim.x + threadIdx.x,res[i]);
}

int main(int argc, char const *argv[])
{
    const int N = argc >= 2 ? std::stoi(argv[1]) : /*1e6*/ 8;
    std::cout << "N = " << N << std::endl;

    float *x, *y, *dx, *dy, *res, *dres;

    float host_expected_result = 0;
    float device_result = 0;

    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));
    res = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 2 * float(std::rand()) / RAND_MAX - 1; // random float in (-1,+1)
        y[i] = 2 * float(std::rand()) / RAND_MAX - 1; // random float in (-1,+1)
        host_expected_result += x[i] * y[i];
        //printf("on fait la multiplication %f * %f = %f et le total est %f\n",y[i],x[i],y[i] * x[i], host_expected_result);
    }

    CUDA_CHECK(cudaMalloc(&dx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dres, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dres, res, N * sizeof(float), cudaMemcpyHostToDevice));

    dot<<<block_count, block_dim>>>(N,dx,dy,dres);
    
    CUDA_CHECK(cudaMemcpy(res, dres, N * sizeof(float), cudaMemcpyDeviceToHost));

    /*int m = 0;
    while( m<2) {
        printf("%f\n", res[m]);
        device_result += res[m];
        m += 1;
    }*/
    device_result = res[0];
    std::cout << "host_expected_result = " << host_expected_result << std::endl;
    std::cout << "device_result = " << device_result << std::endl;


    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dres);
    free(x);
    free(y);
    free(res);
    
    return 0;
}