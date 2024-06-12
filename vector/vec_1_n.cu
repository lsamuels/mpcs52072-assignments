#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <time.h>

#define N 512

__global__ void add(int *a, int *b, int *c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void random_ints(int *ip,int size) {
    // generate different seed for random number 
    time_t t;
    srand((unsigned int) time(&t));
    for (int i=0; i<size; i++) {
        ip[i] = (int)( rand() & 0xFF )/10.0f;
    } 
}

int main(void) {
    
    int *h_a, *h_b, *h_c;   // host copies of h_a, h_b, h+c
    int *d_a, *d_b, *d_c;	// device copies of h_a, h_b, h_c
    int size = N * sizeof(int);
    
    // Alloc space for device copies of h_a, h_b, h_c
    cudaMalloc((int **)&d_a, size);
    cudaMalloc((int **)&d_b, size);
    cudaMalloc((int **)&d_c, size);

    // Alloc space for host copies of a, b, c and setup input values
    h_a = (int *)malloc(size); 
    random_ints(h_a, N);
    h_b = (int *)malloc(size); 
    random_ints(h_b, N);
    h_c = (int *)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with N blocks
    add<<<1,N>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    //Print out results 
    for(int i =0; i < N; i++) {
        printf("a[%d]=%d, b[%d]=%d, c=[%d]=%d\n",i,h_a[i],i,h_b[i],i,h_c[i]); 
    }

    return 0;
}