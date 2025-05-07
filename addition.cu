#include <iostream>              // Standard C++ library for input/output operations
#include <cuda_runtime.h>       // CUDA runtime API header for managing GPU resources

#define N 1000000               // Defining size of the vectors to be 1 million

// __global__ marks this function as a CUDA kernel, which runs on the device (GPU)
__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Calculate global index of the thread
    if (idx < n) {
        C[idx] = A[idx] + B[idx];                     // Perform addition only if index is within bounds
    }
}

int main() {
    int *A, *B, *C;               // Host pointers (CPU memory)
    int *d_A, *d_B, *d_C;         // Device pointers (GPU memory)

    size_t size = N * sizeof(int); // Total memory needed for each vector (in bytes)

    // Allocate memory on the host (CPU)
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);

    // Allocate memory on the device (GPU)
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Initialize host vectors A and B with random values from 0 to 99
    for (int i = 0; i < N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Copy data from host to device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define number of threads per block
    int blockSize = 256;

    // Compute number of blocks needed (ceil division to handle leftover elements)
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch CUDA kernel with computed grid and block sizes
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Copy the result from device back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the first 10 elements of each vector to verify the result
    std::cout << "First 10 elements of Vector A (Input):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "A[" << i << "] = " << A[i] << std::endl;
    }

    std::cout << "\nFirst 10 elements of Vector B (Input):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "B[" << i << "] = " << B[i] << std::endl;
    }

    std::cout << "\nFirst 10 elements of Vector C (Output):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "C[" << i << "] = " << C[i] << std::endl;
    }

    // Free allocated memory on both host and device
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
