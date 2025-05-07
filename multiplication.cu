#include <iostream>               // Used for input and output functions (e.g., std::cout)
#include <cuda_runtime.h>        // Includes CUDA runtime API functions and definitions

#define N 1024                   // Defines the size of the matrix (N x N)

// CUDA kernel function for matrix multiplication
__global__ void matrixMul(int *A, int *B, int *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Compute the row index for this thread
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Compute the column index for this thread

    if (row < n && col < n) {    // Check if indices are within bounds
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col]; // Dot product of the row of A and column of B
        }
        C[row * n + col] = sum;  // Store the result in matrix C
    }
}

int main() {
    int *A, *B, *C;              // Pointers for host matrices
    int *d_A, *d_B, *d_C;        // Pointers for device (GPU) matrices

    size_t size = N * N * sizeof(int); // Total size in bytes for an N x N integer matrix

    // Allocate memory on the host (CPU)
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);

    // Allocate memory on the device (GPU)
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Initialize host matrices A and B with random values from 0 to 9
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = rand() % 10;
            B[i * N + j] = rand() % 10;
        }
    }

    // Copy input matrices from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define the number of threads per block in both x and y directions
    dim3 threadsPerBlock(16, 16);

    // Calculate number of blocks in each dimension required to cover the matrix
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the CUDA kernel on the GPU
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Display first 5x5 elements of input matrix A
    std::cout << "First 5x5 elements of Matrix A (Input):" << std::endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Display first 5x5 elements of input matrix B
    std::cout << "\nFirst 5x5 elements of Matrix B (Input):" << std::endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << B[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Display first 5x5 elements of output matrix C
    std::cout << "\nFirst 5x5 elements of Matrix C (Output):" << std::endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free the allocated memory on host
    free(A);
    free(B);
    free(C);

    // Free the allocated memory on device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0; // Indicate successful execution
}
