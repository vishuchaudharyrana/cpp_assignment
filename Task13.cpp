### *CUDA Multi-Threaded Prime Number Finder*
cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// CUDA kernel to mark non-prime numbers
__global__ void markNonPrimes(bool* isPrime, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // Compute global thread index
    if (index < 2) return; // 0 and 1 are not prime numbers
    if (index * index > n) return; // No need to mark beyond sqrt(n)
    
    if (isPrime[index]) { // If index is prime, mark its multiples as non-prime
        for (int j = index * index; j <= n; j += index) {
            isPrime[j] = false;
        }
    }
}

// Function to find prime numbers up to n using CUDA
std::vector<int> findPrimesCUDA(int n) {
    bool* d_isPrime;
    size_t size = (n + 1) * sizeof(bool);
    cudaMalloc(&d_isPrime, size);
    cudaMemset(d_isPrime, true, size);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    markNonPrimes<<<numBlocks, blockSize>>>(d_isPrime, n);
    cudaDeviceSynchronize();

    std::vector<bool> h_isPrime(n + 1);
    cudaMemcpy(h_isPrime.data(), d_isPrime, size, cudaMemcpyDeviceToHost);
    cudaFree(d_isPrime);

    std::vector<int> primes;
    for (int i = 2; i <= n; ++i) {
        if (h_isPrime[i]) {
            primes.push_back(i);
        }
    }
    return primes;
}

// Function to find prime numbers up to n using a sequential approach
std::vector<int> findPrimesSequential(int n) {
    std::vector<bool> isPrime(n + 1, true);
    isPrime[0] = isPrime[1] = false;

    for (int i = 2; i * i <= n; ++i) {
        if (isPrime[i]) {
            for (int j = i * i; j <= n; j += i) {
                isPrime[j] = false;
            }
        }
    }

    std::vector<int> primes;
    for (int i = 2; i <= n; ++i) {
        if (isPrime[i]) {
            primes.push_back(i);
        }
    }
    return primes;
}

int main() {
    int n = 1000000;

    // Find primes using CUDA
    std::vector<int> primesCUDA = findPrimesCUDA(n);

    // Find primes using sequential approach
    std::vector<int> primesSequential = findPrimesSequential(n);

    // Compare results
    if (primesCUDA == primesSequential) {
        std::cout << "Both methods found the same prime numbers up to " << n << std::endl;
    } else {
        std::cout << "Mismatch in prime numbers found by CUDA and sequential methods" << std::endl;
    }

    return 0;
}


