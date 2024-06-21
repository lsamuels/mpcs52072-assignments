#include <stdio.h> 
#include <cuda_runtime.h>
#include <stdlib.h> 
#include <string.h> 
#include <time.h>
#include <sys/time.h>

#define N (1<<24)
#define THREADS_PER_BLOCK 1024

#define CUDA_CHECK(call)                                                    \
{                                                                           \
   const cudaError_t error = call;                                          \
   if (error != cudaSuccess)                                                \
   {                                                                        \
       printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
       printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));   \
       exit(1);                                                             \
   }                                                                        \
}                                                                           \

double get_time_secs() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void add(float *a, float *b, float *c, const int num_elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elems) { 
        c[idx] = a[idx] + b[idx];
    }
}

void random_floats(float *ip,int size) {
    // generate different seed for random number 
    time_t t;
    srand((unsigned int) time(&t));
    for (int i=0; i<size; i++) {
        ip[i] =  (rand() & 0xFF )/10.0f;
    } 
}

int main(int argc, char *argv[]) {
    printf("%s Starting...\n", argv[0]);

    float *h_a, *h_b, *h_c;   // host copies of h_a, h_b, h+c
    float *d_a, *d_b, *d_c;	// device copies of h_a, h_b, h_c
    int size = N * sizeof(float);
    
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CUDA_CHECK(cudaSetDevice(dev));

    // Alloc space for device copies of h_a, h_b, h_c
    CUDA_CHECK(cudaMalloc((float **)&d_a, size));
    CUDA_CHECK(cudaMalloc((float **)&d_b, size));
    CUDA_CHECK(cudaMalloc((float **)&d_c, size));

    // Alloc space for host copies of a, b, c and setup input values
    h_a = (float *)malloc(size); 
    random_floats(h_a, N);
    h_b = (float *)malloc(size); 
    random_floats(h_b, N);
    h_c = (float *)malloc(size);
    memset(h_c, 0, size);

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch add() kernel on GPU with N blocks
    dim3 block (THREADS_PER_BLOCK);
    dim3 grid ((N+block.x-1)/block.x);
    /***** Start the timer *****/ 
    double start_time = get_time_secs();
    add<<<grid,block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    /***** End the timer *****/
    double elapsed_time = get_time_secs() - start_time;

    printf("add <<<%d,%d>>> Time elapsed %f" 
           "sec\n", grid.x, block.x, elapsed_time);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}