#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

__global__ void rowWiseAdd(double* A, double* B, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        A[row * cols + col] += B[col];
    }
}

void verifyResult(double* A_h, int rows, int cols) {
    double soln[3][3] = {{351, 159, 272},
                         {445, 170, 344},
                         {275, 170, 277}};
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (A_h[i * cols + j] != soln[i][j])
                throw std::runtime_error(
                    "Mismatch at (" + std::to_string(i) + "," +
                    std::to_string(j) + "): got " +
                    std::to_string(A_h[i * cols + j]) +
                    " expected " + std::to_string(soln[i][j]));
}

int main() {
    const int rows = 3, cols = 3;

    double init_A[3][3] = {{130, 147, 115},
                           {224, 158, 187},
                           { 54, 158, 120}};
    double init_B[3] = {221, 12, 157};

    // Allocate and copy to device
    double *A_d, *B_d;
    cudaMalloc(&A_d, rows * cols * sizeof(double));
    cudaMalloc(&B_d, cols * sizeof(double));
    cudaMemcpy(A_d, init_A, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, init_B, cols * sizeof(double),        cudaMemcpyHostToDevice);

    dim3 blockSize(2, 2);
    dim3 gridSize((rows + blockSize.x - 1) / blockSize.x,
                  (cols + blockSize.y - 1) / blockSize.y);
    rowWiseAdd<<<gridSize, blockSize>>>(A_d, B_d, rows, cols);
    cudaDeviceSynchronize();

    double A_h[3][3];
    cudaMemcpy(A_h, A_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    try {
        verifyResult(&A_h[0][0], rows, cols);
        std::cout << "Result matrix A after row-wise addition:\n";
        for (int i = 0; i < rows; i++) {
            std::cout << "[ ";
            for (int j = 0; j < cols; j++) std::cout << A_h[i][j] << " ";
            std::cout << "]\n";
        }
        std::cout << "Verification passed: result matches expected solution.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    cudaFree(A_d);
    cudaFree(B_d);
    return 0;
}
