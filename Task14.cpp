//*CUDA Multithreaded Histogram Computation*
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel to compute histogram
__global__ void computeHistogram(int* d_data, int* d_histogram, int dataSize, int numBins) {
    int index = threadIdx.x + blockIdx.x * blockDim.x; // Compute global thread index
    if (index < dataSize) {
        atomicAdd(&d_histogram[d_data[index]], 1); // Atomic operation to update histogram bin
    }
}

// Function to compute histogram using CUDA
void histogramCUDA(const std::vector<int>& data, std::vector<int>& histogram, int numBins) {
    int* d_data;
    int* d_histogram;
    int dataSize = data.size();

    // Allocate memory on GPU
    cudaMalloc(&d_data, dataSize * sizeof(int));
    cudaMalloc(&d_histogram, numBins * sizeof(int));

    // Copy data to GPU and initialize histogram bins
    cudaMemcpy(d_data, data.data(), dataSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, numBins * sizeof(int));

    // Define block and grid sizes
    int blockSize = 256;
    int numBlocks = (dataSize + blockSize - 1) / blockSize;

    // Launch CUDA kernel
    computeHistogram<<<numBlocks, blockSize>>>(d_data, d_histogram, dataSize, numBins);
    cudaDeviceSynchronize();

    // Copy histogram back to CPU
    cudaMemcpy(histogram.data(), d_histogram, numBins * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_data);
    cudaFree(d_histogram);
}

// Function to compute histogram using a single-threaded CPU approach
void histogramCPU(const std::vector<int>& data, std::vector<int>& histogram, int numBins) {
    for (int value : data) {
        histogram[value]++;
    }
}

int main() {
    std::vector<int> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // Sample data
    int numBins = 10;
    std::vector<int> histogramCUDAResult(numBins, 0);
    std::vector<int> histogramCPUResult(numBins, 0);

    // Compute histogram using CUDA
    histogramCUDA(data, histogramCUDAResult, numBins);

    // Compute histogram using single-threaded CPU approach
    histogramCPU(data, histogramCPUResult, numBins);

    // Compare results
    bool isCorrect = (histogramCUDAResult == histogramCPUResult);
    std::cout << "Histogram computation correctness: " << (isCorrect ? "Valid" : "Invalid") << std::endl;

    return 0;
}

//-----output-------
Running NVIDIA GTX TITAN X in FUNCTIONAL mode...
Compiling...
Executing...
Histogram computation correctness: Valid
Exit status: 0

