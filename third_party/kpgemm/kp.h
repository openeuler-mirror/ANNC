#ifndef KP_H
#define KP_H

#include <stddef.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
	/* Assume C declarations for C++ */
#endif  /* __cplusplus */

/*Set the number of threads on runtime.*/
void openblas_set_num_threads(int num_threads);
void goto_set_num_threads(int num_threads);
int openblas_set_num_threads_local(int num_threads);

/*Get the number of threads on runtime.*/
int openblas_get_num_threads(void);

/*Get the number of physical processors (cores).*/
int openblas_get_num_procs(void);

/*Get the build configure on runtime.*/
char* openblas_get_config(void);

/*Get the CPU corename on runtime.*/
char* openblas_get_corename(void);

/*Set the threading backend to a custom callback.*/
typedef void (*openblas_dojob_callback)(int thread_num, void *jobdata, int dojob_data);
typedef void (*openblas_threads_callback)(int sync, openblas_dojob_callback dojob, int numjobs, size_t jobdata_elsize, void *jobdata, int dojob_data);
void openblas_set_threads_callback_function(openblas_threads_callback callback);

#ifdef OPENBLAS_OS_LINUX
/* Sets thread affinity for OpenBLAS threads. `thread_idx` is in [0, openblas_get_num_threads()-1]. */
int openblas_setaffinity(int thread_idx, size_t cpusetsize, cpu_set_t* cpu_set);
/* Queries thread affinity for OpenBLAS threads. `thread_idx` is in [0, openblas_get_num_threads()-1]. */
int openblas_getaffinity(int thread_idx, size_t cpusetsize, cpu_set_t* cpu_set);
#endif

/* Get the parallelization type which is used by OpenBLAS */
int openblas_get_parallel(void);
/* OpenBLAS is compiled for sequential use  */
#define OPENBLAS_SEQUENTIAL  0
/* OpenBLAS is compiled using normal threading model */
#define OPENBLAS_THREAD  1
/* OpenBLAS is compiled using OpenMP threading model */
#define OPENBLAS_OPENMP 2


/*
 * Since all of GotoBlas was written without const,
 * we disable it at build time.
 */
#ifndef OPENBLAS_CONST
# define OPENBLAS_CONST const
#endif


#define KP_INDEX size_t

typedef enum KP_ORDER     {KpRowMajor=101, KpColMajor=102} KP_ORDER;
typedef enum KP_TRANSPOSE {KpNoTrans=111, KpTrans=112, KpConjTrans=113, KpConjNoTrans=114} KP_TRANSPOSE;
typedef enum KP_UPLO      {KpUpper=121, KpLower=122} KP_UPLO;
typedef enum KP_DIAG      {KpNonUnit=131, KpUnit=132} KP_DIAG;
typedef enum KP_SIDE      {KpLeft=141, KpRight=142} KP_SIDE;
typedef KP_ORDER KP_LAYOUT;


void kp_sgemv(OPENBLAS_CONST enum KP_ORDER order,  OPENBLAS_CONST enum KP_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
		 OPENBLAS_CONST float alpha, OPENBLAS_CONST float  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST float beta,  float  *y, OPENBLAS_CONST blasint incy);
void kp_dgemv(OPENBLAS_CONST enum KP_ORDER order,  OPENBLAS_CONST enum KP_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
		 OPENBLAS_CONST double alpha, OPENBLAS_CONST double  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST double  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST double beta,  double  *y, OPENBLAS_CONST blasint incy);
void kp_cgemv(OPENBLAS_CONST enum KP_ORDER order,  OPENBLAS_CONST enum KP_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST void *beta,  void  *y, OPENBLAS_CONST blasint incy);
void kp_zgemv(OPENBLAS_CONST enum KP_ORDER order,  OPENBLAS_CONST enum KP_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST void *beta,  void  *y, OPENBLAS_CONST blasint incy);


void kp_sgemm(OPENBLAS_CONST enum KP_ORDER Order, OPENBLAS_CONST enum KP_TRANSPOSE TransA, OPENBLAS_CONST enum KP_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
void kp_dgemm(OPENBLAS_CONST enum KP_ORDER Order, OPENBLAS_CONST enum KP_TRANSPOSE TransA, OPENBLAS_CONST enum KP_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);
void kp_cgemm(OPENBLAS_CONST enum KP_ORDER Order, OPENBLAS_CONST enum KP_TRANSPOSE TransA, OPENBLAS_CONST enum KP_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
void kp_cgemm3m(OPENBLAS_CONST enum KP_ORDER Order, OPENBLAS_CONST enum KP_TRANSPOSE TransA, OPENBLAS_CONST enum KP_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
void kp_zgemm(OPENBLAS_CONST enum KP_ORDER Order, OPENBLAS_CONST enum KP_TRANSPOSE TransA, OPENBLAS_CONST enum KP_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
void kp_zgemm3m(OPENBLAS_CONST enum KP_ORDER Order, OPENBLAS_CONST enum KP_TRANSPOSE TransA, OPENBLAS_CONST enum KP_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);


void kp_sgemm_batch(OPENBLAS_CONST enum KP_ORDER Order, OPENBLAS_CONST enum KP_TRANSPOSE * TransA_array, OPENBLAS_CONST enum KP_TRANSPOSE * TransB_array, OPENBLAS_CONST blasint * M_array, OPENBLAS_CONST blasint * N_array, OPENBLAS_CONST blasint * K_array,
		       OPENBLAS_CONST float * alpha_array, OPENBLAS_CONST float ** A_array, OPENBLAS_CONST blasint * lda_array, OPENBLAS_CONST float ** B_array, OPENBLAS_CONST blasint * ldb_array, OPENBLAS_CONST float * beta_array, float ** C_array, OPENBLAS_CONST blasint * ldc_array, OPENBLAS_CONST blasint group_count, OPENBLAS_CONST blasint * group_size);

void kp_dgemm_batch(OPENBLAS_CONST enum KP_ORDER Order, OPENBLAS_CONST enum KP_TRANSPOSE * TransA_array, OPENBLAS_CONST enum KP_TRANSPOSE * TransB_array, OPENBLAS_CONST blasint * M_array, OPENBLAS_CONST blasint * N_array, OPENBLAS_CONST blasint * K_array,
		       OPENBLAS_CONST double * alpha_array, OPENBLAS_CONST double ** A_array, OPENBLAS_CONST blasint * lda_array, OPENBLAS_CONST double ** B_array, OPENBLAS_CONST blasint * ldb_array, OPENBLAS_CONST double * beta_array, double ** C_array, OPENBLAS_CONST blasint * ldc_array, OPENBLAS_CONST blasint group_count, OPENBLAS_CONST blasint * group_size);

void kp_cgemm_batch(OPENBLAS_CONST enum KP_ORDER Order, OPENBLAS_CONST enum KP_TRANSPOSE * TransA_array, OPENBLAS_CONST enum KP_TRANSPOSE * TransB_array, OPENBLAS_CONST blasint * M_array, OPENBLAS_CONST blasint * N_array, OPENBLAS_CONST blasint * K_array,
		       OPENBLAS_CONST void * alpha_array, OPENBLAS_CONST void ** A_array, OPENBLAS_CONST blasint * lda_array, OPENBLAS_CONST void ** B_array, OPENBLAS_CONST blasint * ldb_array, OPENBLAS_CONST void * beta_array, void ** C_array, OPENBLAS_CONST blasint * ldc_array, OPENBLAS_CONST blasint group_count, OPENBLAS_CONST blasint * group_size);

void kp_zgemm_batch(OPENBLAS_CONST enum KP_ORDER Order, OPENBLAS_CONST enum KP_TRANSPOSE * TransA_array, OPENBLAS_CONST enum KP_TRANSPOSE * TransB_array, OPENBLAS_CONST blasint * M_array, OPENBLAS_CONST blasint * N_array, OPENBLAS_CONST blasint * K_array,
		       OPENBLAS_CONST void * alpha_array, OPENBLAS_CONST void ** A_array, OPENBLAS_CONST blasint * lda_array, OPENBLAS_CONST void ** B_array, OPENBLAS_CONST blasint * ldb_array, OPENBLAS_CONST void * beta_array, void ** C_array, OPENBLAS_CONST blasint * ldc_array, OPENBLAS_CONST blasint group_count, OPENBLAS_CONST blasint * group_size);

/*** BFLOAT16 and INT8 extensions ***/
float  kp_sbdot(OPENBLAS_CONST blasint n, OPENBLAS_CONST bfloat16 *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST bfloat16 *y, OPENBLAS_CONST blasint incy);
void   kp_sbgemv(OPENBLAS_CONST enum KP_ORDER order,  OPENBLAS_CONST enum KP_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST bfloat16 *a, OPENBLAS_CONST blasint lda, OPENBLAS_CONST bfloat16 *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST float beta, float *y, OPENBLAS_CONST blasint incy);

void   kp_sbgemm(OPENBLAS_CONST enum KP_ORDER Order, OPENBLAS_CONST enum KP_TRANSPOSE TransA, OPENBLAS_CONST enum KP_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		    OPENBLAS_CONST float alpha, OPENBLAS_CONST bfloat16 *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST bfloat16 *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
void kp_sbgemm_batch(OPENBLAS_CONST enum KP_ORDER Order, OPENBLAS_CONST enum KP_TRANSPOSE * TransA_array, OPENBLAS_CONST enum KP_TRANSPOSE * TransB_array, OPENBLAS_CONST blasint * M_array, OPENBLAS_CONST blasint * N_array, OPENBLAS_CONST blasint * K_array,
		       OPENBLAS_CONST float * alpha_array, OPENBLAS_CONST bfloat16 ** A_array, OPENBLAS_CONST blasint * lda_array, OPENBLAS_CONST bfloat16 ** B_array, OPENBLAS_CONST blasint * ldb_array, OPENBLAS_CONST float * beta_array, float ** C_array, OPENBLAS_CONST blasint * ldc_array, OPENBLAS_CONST blasint group_count, OPENBLAS_CONST blasint * group_size);

#ifdef __cplusplus
}
#endif  /* __cplusplus */

#endif
