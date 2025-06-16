// *CUDA Parallel Vector Addition*
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// CUDA Kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main() {
    std::cout << "... CUDA Vector Addition...\n";

    const int N = 1000000;
    size_t size = N * sizeof(float);

    // Host memory allocation
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];
    float* h_C_CPU = new float[N];

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch CUDA kernel and time it
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // CPU addition and timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
        h_C_CPU[i] = h_A[i] + h_B[i];
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // Verify result (first few elements)
    std::cout << "\n Results:\n";
    for (int i = 0; i < 5; ++i)
        std::cout << "A[" << i << "] + B[" << i << "] = " << h_C[i] << "\n";

    // Show timings
    std::cout << "\n Vector addition completed!\n";
    std::cout << " GPU Time: " << gpuTime << " ms\n";
    std::cout << " CPU Time: " << cpuTime << " ms\n";

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_CPU;

    return 0;
}

//------output------
Running NVIDIA GTX TITAN X in FUNCTIONAL mode...
Compiling...
Executing...
... CUDA Vector Addition...

 Results:
A[0] + B[0] = 0
A[1] + B[1] = 3
A[2] + B[2] = 6
A[3] + B[3] = 9
A[4] + B[4] = 12

 Vector addition completed!
 GPU Time: 190.704 ms
 CPU Time: 3.83186 ms
Exit status: 0

