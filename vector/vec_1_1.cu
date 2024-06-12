#include <stdio.h> 

__global__ void add(int *a, int *b, int *c) {
		*c = *a + *b;
}
    
int main(void) {
		int h_a, h_b, h_c;	         // host copies of h_a, h_b, h_c
		int *d_a, *d_b, *d_c;	     // device copies of a, b, c
		int size = sizeof(int);
		
		// Allocate space for device copies of h_a, h_b, h_c
		cudaMalloc((int **)&d_a, size);
		cudaMalloc((int **)&d_b, size);
		cudaMalloc((int **)&d_c, size);

		// Setup input values
		h_a = 2;
		h_b = 7;		
        
        // Copy inputs to device
		cudaMemcpy(d_a, &h_a, size, cudaMemcpyHostToDevice); // This is a blocking call 
		cudaMemcpy(d_b, &h_b, size, cudaMemcpyHostToDevice); // This is a blocking call 

		// Launch add() kernel on GPU
		add<<<1,1>>>(d_a, d_b, d_c); // This is not a blocking call 

		// Copy result back to host
		cudaMemcpy(&h_c, d_c, size, cudaMemcpyDeviceToHost); // This is a blocking call 

		// Cleanup
		cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

        // Print out the result 
        printf("a = %d, b = %d, c = %d\n",h_a,h_b,h_c); 
		return 0;
}