#include <stdio.h> 

__global__ void mykernel(void) {
	printf("Hello");

}

int main(void) {
    mykernel<<<1,1>>>();
    cudaDeviceSynchronize(); 
	printf(" World!\n");
	return 0;
}
