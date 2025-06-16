//*CUDA Parallel Matrix Transpose*
// Step 1: Include necessary headers
#include <iostream>
#include <cuda_runtime.h>

#define MATRIX_SIZE 1024 // Define the size of the matrix

// Step 2: CUDA kernel for matrix transpose
__global__ void matrixTransposeKernel(float* d_inputMatrix, float* d_transposedMatrix, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Compute row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Compute column index

    if (row < width && col < width) { // Ensure indices are within bounds
        int inputIndex = row * width + col; // Compute input matrix index
        int transposedIndex = col * width + row; // Compute transposed matrix index
        d_transposedMatrix[transposedIndex] = d_inputMatrix[inputIndex]; // Perform transpose
    }
}

// Step 3: CPU-based matrix transpose for validation
void matrixTransposeCPU(float* h_inputMatrix, float* h_transposedMatrix, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            h_transposedMatrix[j * width + i] = h_inputMatrix[i * width + j]; // Swap row and column
        }
    }
}

// Step 4: Main function to set up and execute the kernel
int main() {
    int matrixSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(float); // Compute memory size

    // Step 5: Allocate memory on the host (CPU)
    float* h_inputMatrix = (float*)malloc(matrixSize);
    float* h_transposedMatrix = (float*)malloc(matrixSize);
    float* h_transposedMatrixCPU = (float*)malloc(matrixSize);

    // Step 6: Initialize input matrix with sample values
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            h_inputMatrix[i * MATRIX_SIZE + j] = static_cast<float>(i * MATRIX_SIZE + j);
        }
    }

    // Step 7: Allocate memory on the device (GPU)
    float* d_inputMatrix;
    float* d_transposedMatrix;
    cudaMalloc(&d_inputMatrix, matrixSize);
    cudaMalloc(&d_transposedMatrix, matrixSize);

    // Step 8: Copy input matrix from host to device
    cudaMemcpy(d_inputMatrix, h_inputMatrix, matrixSize, cudaMemcpyHostToDevice);

    // Step 9: Define block and grid sizes
    dim3 blockSize(16, 16); // Define block size (16x16 threads)
    dim3 gridSize((MATRIX_SIZE + blockSize.x - 1) / blockSize.x, (MATRIX_SIZE + blockSize.y - 1) / blockSize.y); // Compute grid size

    // Step 10: Launch the CUDA kernel
    matrixTransposeKernel<<<gridSize, blockSize>>>(d_inputMatrix, d_transposedMatrix, MATRIX_SIZE);

    // Step 11: Copy transposed matrix from device to host
    cudaMemcpy(h_transposedMatrix, d_transposedMatrix, matrixSize, cudaMemcpyDeviceToHost);

    // Step 12: Validate results against CPU-based matrix transpose
    matrixTransposeCPU(h_inputMatrix, h_transposedMatrixCPU, MATRIX_SIZE);

    bool isValid = true;
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        if (h_transposedMatrix[i] != h_transposedMatrixCPU[i]) {
            isValid = false;
            break;
        }
    }

    // Step 13: Display validation result
    if (isValid) {
        std::cout << "Matrix transpose is correct!" << std::endl;
    } else {
        std::cout << "Matrix transpose is incorrect!" << std::endl;
    }

    // Step 14: Free device memory
    cudaFree(d_inputMatrix);
    cudaFree(d_transposedMatrix);

    // Step 15: Free host memory
    free(h_inputMatrix);
    free(h_transposedMatrix);
    free(h_transposedMatrixCPU);

    return 0;
}

//-----output-----
Running NVIDIA GTX TITAN X in FUNCTIONAL mode...
Compiling...
Executing...
Matrix transpose is correct!
Exit status: 0
