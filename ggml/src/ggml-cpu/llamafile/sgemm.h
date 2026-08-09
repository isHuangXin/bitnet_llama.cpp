#pragma once
#include <stdint.h>
#include <stdbool.h>

#if defined(__VXE__) || defined(__VXE2__)
#include <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
extern "C" {
#endif

bool llamafile_sgemm(const struct ggml_compute_params * params, int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     int, int, int);

bool llamafile_sgemm_i2s(const struct ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                         const void *A, int64_t lda, const void *B, int64_t ldb, void *C, int64_t ldc,
                         const float *act_scales, const int32_t *act_sums, float weight_scale);

#ifdef __cplusplus
}
#endif
