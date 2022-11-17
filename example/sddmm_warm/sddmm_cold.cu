#include "../../src/sddmm/sddmm.h"
#include "../util/sp_util.hpp" // read_mtx
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>

#define VALIDATE

/* This example need CUDA Version >=11.3 */
// correspond with benchmark

double calc_std(std::vector<float> vec){
    if ((float)vec.size() > 0){
        float mean = 0;
        for (auto val: vec){
            mean += val;
        }
        mean /= (float)vec.size();

        float std = 0;
        std::for_each(std::begin(vec), std::end(vec), [&](const float d) {
            std += (d - mean) * (d - mean);
        });
        return sqrt(std / ((float)vec.size() - 1));
    }
    return 0;
}

int main(int argc, char *argv[]) {
    // check command-line argument

    if (argc < 2) {
        printf("Require command-line argument: name of the sparse matrix file in "
               ".mtx format.\n");
        return EXIT_FAILURE;
    }

    //
    // Load sparse matrix
    //

    int M;                              // number of S-rows
    int N;                              // number of S-columns
    int nnz;                            // number of non-zeros in S
    std::vector<int> csr_indptr_buffer; // buffer for indptr array in CSR format
    std::vector<int>
            csr_indices_buffer; // buffer for indices (column-ids) array in CSR format
    // load sparse matrix from mtx file
    read_mtx_file(argv[1], M, N, nnz, csr_indptr_buffer, csr_indices_buffer);
    printf("Finish reading matrix %d rows, %d columns, %d nnz. \nIgnore original "
           "values and use randomly generated values.\n",
           M, N, nnz);

    // Create GPU arrays
    int K = 128; // number of A-columns
    if (argc > 2) {
        K = atoi(argv[2]);
    }
    assert(
            K > 0 &&
            "second command-line argument is number of B columns, should be >0.\n");

    int iters = 1000;

    float kernel_time, d2h_time, h2d_time, event_time;
    std::vector<float> event_times;
    kernel_time = 0;
    d2h_time = 0;
    h2d_time = 0;
    event_time = 0;

    float *A_h = NULL, *B_h = NULL, *C_h = NULL, *csr_values_h = NULL,
            *C_ref = NULL;
    float *A_d = NULL, *B_d = NULL, *csr_values_d = NULL;
    int *csr_indptr_d = NULL, *csr_indices_d = NULL;
    A_h = (float *) malloc(sizeof(float) * M * K);
    B_h = (float *) malloc(sizeof(float) * N * K);
    C_h = (float *) malloc(sizeof(float) * nnz);
    C_ref = (float *) malloc(sizeof(float) * nnz);
    csr_values_h = (float *) malloc(sizeof(float) * nnz);
    if (!A_h || !B_h || !C_h || !C_ref || !csr_values_h) {
        printf("Host allocation failed.\n");
        return EXIT_FAILURE;
    }
    fill_random(A_h, M * K);
    fill_random(B_h, N * K);

    for (int x = 0; x < iters; x++) {
        fill_zero(csr_values_h, nnz);
        cudaDeviceReset();
        cudaSetDevice(0);

        cudaEvent_t start_k, stop_k, start_d2h, stop_d2h, start_h2d, stop_h2d, start_e, stop_e;
        cudaEventCreate(&start_k);
        cudaEventCreate(&stop_k);
        cudaEventCreate(&start_e);
        cudaEventCreate(&stop_e);
        cudaEventCreate(&start_d2h);
        cudaEventCreate(&stop_d2h);
        cudaEventCreate(&start_h2d);
        cudaEventCreate(&stop_h2d);

        // allocate device memory
        cudaEventRecord(start_e);
        CUDA_CHECK(cudaMalloc((void **) &A_d, sizeof(float) * M * K));
        CUDA_CHECK(cudaMalloc((void **) &B_d, sizeof(float) * N * K));
        CUDA_CHECK(cudaMalloc((void **) &csr_values_d, sizeof(float) * nnz));
        CUDA_CHECK(cudaMalloc((void **) &csr_indptr_d, sizeof(int) * (M + 1)));
        CUDA_CHECK(cudaMalloc((void **) &csr_indices_d, sizeof(int) * nnz));

        cudaEventRecord(start_h2d);
        CUDA_CHECK(
                cudaMemcpy(A_d, A_h, sizeof(float) * M * K, cudaMemcpyHostToDevice));
        CUDA_CHECK(
                cudaMemcpy(B_d, B_h, sizeof(float) * N * K, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(csr_indptr_d, csr_indptr_buffer.data(),
                              sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(csr_indices_d, csr_indices_buffer.data(),
                              sizeof(int) * nnz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(csr_values_d, csr_values_h, sizeof(float) * nnz,
                              cudaMemcpyHostToDevice));
        cudaEventRecord(stop_h2d);

        cudaEventRecord(start_k);
        sddmm_cuda_csr(M, K, nnz, csr_indptr_d, csr_indices_d, A_d, B_d, csr_values_d);
        cudaEventRecord(stop_k);

        cudaEventRecord(start_d2h);
        CUDA_CHECK(cudaMemcpy(csr_values_h, csr_values_d, nnz * sizeof(float),
                              cudaMemcpyDeviceToHost));
        cudaEventRecord(stop_d2h);
        cudaEventRecord(stop_e);

        if (A_d) CUDA_CHECK(cudaFree(A_d));
        if (B_d) CUDA_CHECK(cudaFree(B_d));
        if (csr_values_d) CUDA_CHECK(cudaFree(csr_values_d));
        if (csr_indptr_d) CUDA_CHECK(cudaFree(csr_indptr_d));
        if (csr_indices_d) CUDA_CHECK(cudaFree(csr_indices_d));

        cudaEventSynchronize(start_e);
        cudaEventSynchronize(stop_e);
        cudaEventSynchronize(stop_h2d);
        cudaEventSynchronize(stop_d2h);
        cudaEventSynchronize(start_k);
        cudaEventSynchronize(stop_k);

        float local_kernel = 0;
        cudaEventElapsedTime(&local_kernel, start_k, stop_k);
        kernel_time += local_kernel / 1000;

        float local_d2h = 0;
        cudaEventElapsedTime(&local_d2h, start_d2h, stop_d2h);
        d2h_time += local_d2h / 1000;

        float local_h2d = 0;
        cudaEventElapsedTime(&local_h2d, start_h2d, stop_h2d);
        h2d_time += local_h2d / 1000;

        float local_event = 0;
        cudaEventElapsedTime(&local_event, start_e, stop_e);
        event_time += local_event / 1000;
        event_times.push_back(local_event / 1000);
    }

    if (A_h)
        free(A_h);
    if (B_h)
        free(B_h);
    if (C_h)
        free(C_h);
    if (C_ref)
        free(C_ref);
    if (csr_values_h)
        free(csr_values_h);

    std::cout << h2d_time / (iters) << '\t'
              << kernel_time / (iters) << '\t'
              << d2h_time / (iters) << '\t'
              << event_time / (iters) << '\t'
              << calc_std(event_times) << std::endl;

    return 0;
}