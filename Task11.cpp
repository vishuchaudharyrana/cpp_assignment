### *CUDA Parallel Vector Addition*
// Step 1: Include necessary headers
#include <iostream>
#include <cuda_runtime.h>

// Step 2: Define the CUDA kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global thread index
    if (i < N) { // Ensure index is within bounds
        C[i] = A[i] + B[i]; // Perform element-wise addition
    }
}

// Step 3: Main function to set up and execute the kernel
int main() {
    int N = 1000000; // Define vector size
    size_t size = N * sizeof(float); // Calculate memory size

    // Step 4: Allocate memory on the host (CPU)
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Step 5: Initialize vectors with sample data
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i); // Assign values to vector A
        h_B[i] = static_cast<float>(i * 2); // Assign values to vector B
    }

    // Step 6: Allocate memory on the device (GPU)
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Step 7: Copy vectors from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Step 8: Define block and grid sizes
    int blockSize = 256; // Number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize; // Calculate number of blocks

    // Step 9: Launch the CUDA kernel
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // Step 10: Copy result vector from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Step 11: Verify the result
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            std::cerr << "Error at index " << i << ": " << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            return -1;
        }
    }
    std::cout << "Vector addition successful!" << std::endl;

    // Step 12: Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Step 13: Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}


