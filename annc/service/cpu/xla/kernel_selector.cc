/* Copyright 2025 Huawei. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "kernel_selector.h"

namespace xla {
namespace cpu {

// TODO: Need to test handling trA, trB
void __xla_cpu_runtime_KernelSelectorGEMM(bool trA, bool trB, const float* A,
                                          const float* B, int M, int N, int K,
                                          float alpha, float beta, float* C) {
  CBLAS_LAYOUT Order = CblasRowMajor;
  CBLAS_TRANSPOSE TransA = (trA) ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE TransB = (trB) ? CblasTrans : CblasNoTrans;
  int lda = trA ? M : K;
  int ldb = trB ? K : N;
  int ldc = N;

  cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

void __xla_cpu_runtime_KernelSelectorBatch3D(bool trA, bool trB, const float* A,
                                             const float* B, int P, int M,
                                             int N, int K, float* C) {
  CBLAS_LAYOUT Order = CblasRowMajor;
  CBLAS_TRANSPOSE TransA = (trA) ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE TransB = (trB) ? CblasTrans : CblasNoTrans;
  int lda = trA ? M : K;
  int ldb = trB ? K : N;
  int ldc = N;

  float alpha = 1.0;
  float beta = 0.0;

  for (int i = 0; i < P; ++i) {
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, &A[i * M * K], lda,
                &B[i * K * N], ldb, beta, &C[i * M * N], ldc);
  }
}

void __xla_cpu_runtime_KernelSelectorGEMV(bool trA, const float* A,
                                          const float* X, int M, int N,
                                          float alpha, float beta, float* Y) {
  int lda = trA ? M : N;
  int incX = 1;
  int incY = 1;
  CBLAS_LAYOUT Order = CblasRowMajor;
  CBLAS_TRANSPOSE TransA = (trA) ? CblasTrans : CblasNoTrans;
  cblas_sgemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

void __xla_cpu_runtime_KernelSelectorGEMMMLIR(bool trA, bool trB,
                                              const float* A, const float* B,
                                              int M, int N, int K, float alpha,
                                              float beta, float* C) {
  CBLAS_LAYOUT Order = CblasRowMajor;
  CBLAS_TRANSPOSE TransA = (trA) ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE TransB = (trB) ? CblasTrans : CblasNoTrans;
  int lda = trA ? M : K;
  int ldb = trB ? K : N;
  int ldc = N;

  cblas_sgemm_mlir(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta,
                   C, ldc);
}

void __xla_cpu_runtime_KernelSelectorBatch3DMLIR(bool trA, bool trB,
                                                 const float* A, const float* B,
                                                 int P, int M, int N, int K,
                                                 float* C) {
  CBLAS_LAYOUT Order = CblasRowMajor;
  CBLAS_TRANSPOSE TransA = (trA) ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE TransB = (trB) ? CblasTrans : CblasNoTrans;
  int lda = trA ? M : K;
  int ldb = trB ? K : N;
  int ldc = N;

  cblas_sbatch_matmul_mlir(Order, TransA, TransB, P, M, N, K, A, lda, B, ldb, C,
                           ldc);
}

void __xla_cpu_runtime_KernelSelectorBatch4DMLIR(bool trA, bool trB,
                                                 const float* A, const float* B,
                                                 int Q, int P, int M, int N,
                                                 int K, float* C) {
  CBLAS_LAYOUT Order = CblasRowMajor;
  CBLAS_TRANSPOSE TransA = (trA) ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE TransB = (trB) ? CblasTrans : CblasNoTrans;
  int lda = trA ? M : K;
  int ldb = trB ? K : N;
  int ldc = N;

  cblas_sbatch_matmul_4d_mlir(Order, TransA, TransB, Q, P, M, N, K, A, lda, B,
                              ldb, C, ldc);
}

void __xla_cpu_runtime_KernelSelectorGEMVMLIR(bool trA, const float* A,
                                              const float* X, int M, int N,
                                              float alpha, float beta,
                                              float* Y) {
  int lda = trA ? M : N;
  int incX = 1;
  int incY = 1;
  CBLAS_LAYOUT Order = CblasRowMajor;
  CBLAS_TRANSPOSE TransA = (trA) ? CblasTrans : CblasNoTrans;

  cblas_sgemv_mlir(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

}  // namespace cpu
}  // namespace xla
