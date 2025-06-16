//*CUDA Parallel Matrix Transpose*
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define SIZE 4 // Matrix size (4x4)

// CUDA kernel for matrix transpose
__global__ void transpose(float* input, float* output, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        int inIdx  = row * width + col;
        int outIdx = col * width + row;
        output[outIdx] = input[inIdx];
    }
}

// CPU-based transpose
void transposeCPU(float* input, float* output, int width) {
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < width; ++j)
            output[j * width + i] = input[i * width + j];
}

int main() {
    std::cout << " CUDA Matrix Transpose (Size: " << SIZE << "x" << SIZE << ")\n";

    const int N = SIZE * SIZE;
    size_t bytes = N * sizeof(float);
    float h_input[N], h_output[N], h_ref[N];

    // Initialize matrix: 0 to N-1
    for (int i = 0; i < N; ++i)
        h_input[i] = i;

    // Allocate GPU memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((SIZE + block.x - 1) / block.x, (SIZE + block.y - 1) / block.y);
    transpose<<<grid, block>>>(d_input, d_output, SIZE);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // Copy result to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // CPU transpose timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    transposeCPU(h_input, h_ref, SIZE);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // Print input matrix
    std::cout << "\n Original Matrix:\n";
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j)
            std::cout << h_input[i * SIZE + j] << "\t";
        std::cout << "\n";
    }

    // Print transposed matrix (from GPU)
    std::cout << "\n Transposed Matrix (GPU):\n";
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j)
            std::cout << h_output[i * SIZE + j] << "\t";
        std::cout << "\n";
    }

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_output[i] != h_ref[i]) {
            correct = false;
            break;
        }
    }

    std::cout << "\n Result: Matrix transpose is " << (correct ? "CORRECT!" : "INCORRECT!") << "\n";
    std::cout << "\n Timings:\n";
    std::cout << "GPU Time: " << gpuTime << " ms\n";
    std::cout << "CPU Time: " << cpuTime << " ms\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}



//-----output-----
Executing...
 CUDA Matrix Transpose (Size: 4x4)

 Original Matrix:
0       1       2       3
4       5       6       7
8       9       10      11
12      13      14      15

 Transposed Matrix (GPU):
0       4       8       12
1       5       9       13
2       6       10      14
3       7       11      15

 Result: Matrix transpose is CORRECT!

 Timings:
GPU Time: 0.536414 ms
CPU Time: 0.000149 ms
Exit status: 0
