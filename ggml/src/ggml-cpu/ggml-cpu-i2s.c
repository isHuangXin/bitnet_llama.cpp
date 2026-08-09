// BitNet I2_S GEMV/GEMM optimized kernels
// Extracted from ggml-aarch64.c

#include "ggml-quants.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <string.h>
#include <assert.h>
#include <math.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#define QK_I2_S 128
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define QK_I2_S 64
#else
#define QK_I2_S 128
#endif

#include "gemm-config.h"

void ggml_gemv_i2_i8_s(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(ACT_PARALLEL)
    for (int64_t iir0 = 0; iir0 < nc; iir0 += 1) {
        ggml_vec_dot_i2_i8_s(n, s + iir0, 1, (const uint8_t *)vx + iir0 * n / 4, n, vy, 0, 1);
    }
#else
    const int64_t blck_0 = 16;
    for (int64_t iir0 = 0; iir0 < nc; iir0 += blck_0) {
        ggml_vec_dot_i2_i8_s(n, s + iir0, 1, (const uint8_t *)vx + iir0 * n / 4, n, vy, 0, blck_0);
    }
#endif
}

void ggml_gemm_i2_i8_s(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    // nr = number of activation columns (B side)
    // nc = number of weight rows (A side)
    // n  = inner dimension (elements per row)
    // vx = weight data (I2_S, 2-bit packed, nc rows, each n/4 bytes)
    // vy = activation data (I8_S, nr columns, each n bytes)
    // s  = output, s[col * bs + row], where col indexes B columns, row indexes A rows

#if defined(__AVX2__)
    const uint8_t * x = (const uint8_t *)vx;
    const int8_t  * y = (const int8_t *)vy;
    const int64_t a_row_bytes = n / 4;
    const int64_t nb = n / 128;  // number of 128-element blocks

    const __m256i mask_2bit = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);

    // Process 4 weight rows (A) × 4 activation columns (B) at a time
    int64_t r = 0;
    for (; r + 4 <= nc; r += 4) {
        int64_t c = 0;
        for (; c + 4 <= nr; c += 4) {
            // 4×4 int32 accumulators
            __m256i acc[4][4];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    acc[i][j] = _mm256_setzero_si256();

            for (int64_t bl = 0; bl < nb; bl++) {
                // Load and unpack 4 rows of A (32 packed bytes → 4×32 uint8 values each)
                __m256i a_unpacked[4][4];
                for (int i = 0; i < 4; i++) {
                    const uint8_t * a_ptr = x + (r + i) * a_row_bytes + bl * 32;
                    __m256i packed = _mm256_loadu_si256((const __m256i *)a_ptr);
                    a_unpacked[i][0] = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask_2bit);
                    a_unpacked[i][1] = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_2bit);
                    a_unpacked[i][2] = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask_2bit);
                    a_unpacked[i][3] = _mm256_and_si256(packed, mask_2bit);
                }

                // Load 4 columns of B (128 bytes each) and compute dot products
                for (int j = 0; j < 4; j++) {
                    const int8_t * b_ptr = y + (c + j) * n + bl * 128;
                    __m256i b0 = _mm256_loadu_si256((const __m256i *)(b_ptr));
                    __m256i b1 = _mm256_loadu_si256((const __m256i *)(b_ptr + 32));
                    __m256i b2 = _mm256_loadu_si256((const __m256i *)(b_ptr + 64));
                    __m256i b3 = _mm256_loadu_si256((const __m256i *)(b_ptr + 96));

                    // Each of 4 A rows multiplied with this B column
                    for (int i = 0; i < 4; i++) {
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][0], b0);
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][1], b1);
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][2], b2);
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][3], b3);
#else
                        __m256i d0 = _mm256_maddubs_epi16(a_unpacked[i][0], b0);
                        __m256i d1 = _mm256_maddubs_epi16(a_unpacked[i][1], b1);
                        __m256i d2 = _mm256_maddubs_epi16(a_unpacked[i][2], b2);
                        __m256i d3 = _mm256_maddubs_epi16(a_unpacked[i][3], b3);
                        __m256i sum16 = _mm256_add_epi16(_mm256_add_epi16(d0, d1),
                                                         _mm256_add_epi16(d2, d3));
                        acc[i][j] = _mm256_add_epi32(acc[i][j],
                                                      _mm256_madd_epi16(sum16, one16));
#endif
                    }
                }
            }

            // Horizontal sum and store 4×4 results
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    __m128i sum128 = _mm_add_epi32(
                        _mm256_castsi256_si128(acc[i][j]),
                        _mm256_extractf128_si256(acc[i][j], 1));
                    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
                    __m128i sum64 = _mm_add_epi32(hi64, sum128);
                    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
                    s[(c + j) * bs + (r + i)] = (float)_mm_cvtsi128_si32(
                        _mm_add_epi32(sum64, hi32));
                }
            }
        }

        // Handle remaining B columns (< 4)
        for (; c < nr; c++) {
            __m256i acc_rem[4];
            for (int i = 0; i < 4; i++)
                acc_rem[i] = _mm256_setzero_si256();

            for (int64_t bl = 0; bl < nb; bl++) {
                const int8_t * b_ptr = y + c * n + bl * 128;
                __m256i b0 = _mm256_loadu_si256((const __m256i *)(b_ptr));
                __m256i b1 = _mm256_loadu_si256((const __m256i *)(b_ptr + 32));
                __m256i b2 = _mm256_loadu_si256((const __m256i *)(b_ptr + 64));
                __m256i b3 = _mm256_loadu_si256((const __m256i *)(b_ptr + 96));

                for (int i = 0; i < 4; i++) {
                    const uint8_t * a_ptr = x + (r + i) * a_row_bytes + bl * 32;
                    __m256i packed = _mm256_loadu_si256((const __m256i *)a_ptr);
                    __m256i a0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask_2bit);
                    __m256i a1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_2bit);
                    __m256i a2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask_2bit);
                    __m256i a3 = _mm256_and_si256(packed, mask_2bit);

#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a0, b0);
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a1, b1);
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a2, b2);
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a3, b3);
#else
                    __m256i d0 = _mm256_maddubs_epi16(a0, b0);
                    __m256i d1 = _mm256_maddubs_epi16(a1, b1);
                    __m256i d2 = _mm256_maddubs_epi16(a2, b2);
                    __m256i d3 = _mm256_maddubs_epi16(a3, b3);
                    __m256i sum16 = _mm256_add_epi16(_mm256_add_epi16(d0, d1),
                                                     _mm256_add_epi16(d2, d3));
                    acc_rem[i] = _mm256_add_epi32(acc_rem[i],
                                                   _mm256_madd_epi16(sum16, one16));
#endif
                }
            }

            for (int i = 0; i < 4; i++) {
                __m128i sum128 = _mm_add_epi32(
                    _mm256_castsi256_si128(acc_rem[i]),
                    _mm256_extractf128_si256(acc_rem[i], 1));
                __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
                __m128i sum64 = _mm_add_epi32(hi64, sum128);
                __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
                s[c * bs + (r + i)] = (float)_mm_cvtsi128_si32(
                    _mm_add_epi32(sum64, hi32));
            }
        }
    }

    // Handle remaining A rows (< 4)
    for (; r < nc; r++) {
        for (int64_t c = 0; c < nr; c++) {
            ggml_vec_dot_i2_i8_s(n, s + c * bs + r, 1,
                (const uint8_t *)vx + r * a_row_bytes, n,
                (const int8_t *)vy + c * n, 0, 1);
        }
    }
#else
    // Scalar / non-AVX2 fallback: use vec_dot row by row
    const int64_t row_block = ROW_BLOCK_SIZE;
    const int64_t col_block = COL_BLOCK_SIZE;

#if defined(ACT_PARALLEL)
    for (int64_t c0 = 0; c0 < nc; c0 += col_block) {
        int64_t cur_c = (c0 + col_block <= nc) ? col_block : (nc - c0);
        for (int64_t r0 = 0; r0 < nr; r0 += row_block) {
            int64_t cur_r = (r0 + row_block <= nr) ? row_block : (nr - r0);
            const void * vy_r = (const uint8_t *)vy + r0 * n;
            for (int64_t c = 0; c < cur_c; ++c) {
                const int64_t col = c0 + c;
                float * s_col = s + col;
                const void * vx_col = (const uint8_t *)vx + col * n / 4;
                ggml_vec_dot_i2_i8_s(n, s_col + r0 * bs, bs, vx_col, n, vy_r, n, cur_r);
            }
        }
    }
#else
    for (int64_t r0 = 0; r0 < nr; r0 += row_block) {
        int64_t cur_r = (r0 + row_block <= nr) ? row_block : (nr - r0);
        const void * vy_row = (const uint8_t *)vy + r0 * n;
        for (int64_t c0 = 0; c0 < nc; c0 += col_block) {
            int64_t cur_c = (c0 + col_block <= nc) ? col_block : (nc - c0);
            for (int64_t r = 0; r < cur_r; ++r) {
                const int64_t row = r0 + r;
                float * s_row = s + row * bs;
                const void * vy_cur = (const uint8_t *)vy_row + r * n;
                const void * vx_c = (const uint8_t *)vx + c0 * n / 4;
                ggml_vec_dot_i2_i8_s(n, s_row + c0, bs, vx_c, n, vy_cur, n, cur_c);
            }
        }
    }
#endif
#endif // __AVX2__
}

// =============================================================
// Interleaved (8-row repacked) GEMM/GEMV for I2_S
//
// Weight layout after repack: groups of 8 rows, within each group
// bytes are interleaved in 8-byte blocks:
//   [row0_b0..b7][row1_b0..b7]...[row7_b0..b7][row0_b8..b15]...
//
// vx_base points to the start of ALL weight data (interleaved)
// row_start is the first output row to process
// nc = number of output rows to process (must be multiple of 4)
// nr = number of B columns to process
// =============================================================

void ggml_gemm_i2_i8_s_interleaved(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx_base, int64_t row_start, int64_t a_row_bytes,
    const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__)
    // For the interleaved layout, the data for row `r` at byte offset `off` is at:
    //   group = r / 8
    //   group_base = group * 8 * a_row_bytes
    //   within_group = (r % 8) * INTERLEAVE + (off / INTERLEAVE) * 8 * INTERLEAVE + (off % INTERLEAVE)
    // With INTERLEAVE=8:
    //   within_group = (r % 8) * 8 + (off / 8) * 64 + (off % 8)
    //
    // But actually, since we process 4 rows that are consecutive (within a group of 8),
    // we can load them efficiently. For 4 consecutive rows r, r+1, r+2, r+3 within the
    // same group, at block `bl`:
    //   row_r:   group_base + bl * 64 + (r%8) * 8
    //   row_r+1: group_base + bl * 64 + ((r+1)%8) * 8
    //   etc.
    // These are 4 * 8 = 32 contiguous bytes if r%8 is 0 or 4!
    // For r%8 = 0: offsets 0,8,16,24 → we can load 32 bytes and deinterleave
    // For r%8 = 4: offsets 32,40,48,56 → same, load 32 bytes

    const uint8_t * x = (const uint8_t *)vx_base;
    const int8_t  * y = (const int8_t *)vy;
    const int64_t nb_128 = n / 128; // number of 128-element blocks
    const int64_t INTERLEAVE = 8;
    const int64_t NROWS_GROUP = 8;
    const int64_t group_stride = NROWS_GROUP * a_row_bytes; // bytes per group of 8 rows

    const __m256i mask_2bit = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);

    int64_t r = 0;
    for (; r + 4 <= nc; r += 4) {
        const int64_t abs_row = row_start + r;
        const int64_t group = abs_row / NROWS_GROUP;
        const int64_t row_in_group = abs_row % NROWS_GROUP;
        const uint8_t * group_base = x + group * group_stride;

        int64_t c = 0;
        for (; c + 4 <= nr; c += 4) {
            __m256i acc[4][4];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    acc[i][j] = _mm256_setzero_si256();

            for (int64_t bl = 0; bl < nb_128; bl++) {
                // Each 128-element block spans 32 packed bytes per row
                // In interleaved layout: 32 bytes = 4 sub-blocks of 8 bytes
                // Load 4 rows × 4 sub-blocks
                __m256i a_unpacked[4][4];
                for (int i = 0; i < 4; i++) {
                    // Load 32 bytes for row (row_in_group+i) from interleaved data
                    // These are at: group_base + (sub_block * 64 + (row_in_group+i) * 8) for each sub_block
                    const int64_t row_off = row_in_group + i;
                    uint8_t tmp_packed[32];
                    for (int sb = 0; sb < 4; sb++) {
                        const uint8_t * src = group_base + (bl * 4 + sb) * NROWS_GROUP * INTERLEAVE + row_off * INTERLEAVE;
                        memcpy(tmp_packed + sb * 8, src, 8);
                    }
                    __m256i packed = _mm256_loadu_si256((const __m256i *)tmp_packed);
                    a_unpacked[i][0] = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask_2bit);
                    a_unpacked[i][1] = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_2bit);
                    a_unpacked[i][2] = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask_2bit);
                    a_unpacked[i][3] = _mm256_and_si256(packed, mask_2bit);
                }

                for (int j = 0; j < 4; j++) {
                    const int8_t * b_ptr = y + (c + j) * n + bl * 128;
                    __m256i b0 = _mm256_loadu_si256((const __m256i *)(b_ptr));
                    __m256i b1 = _mm256_loadu_si256((const __m256i *)(b_ptr + 32));
                    __m256i b2 = _mm256_loadu_si256((const __m256i *)(b_ptr + 64));
                    __m256i b3 = _mm256_loadu_si256((const __m256i *)(b_ptr + 96));

                    for (int i = 0; i < 4; i++) {
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][0], b0);
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][1], b1);
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][2], b2);
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][3], b3);
#else
                        __m256i d0 = _mm256_maddubs_epi16(a_unpacked[i][0], b0);
                        __m256i d1 = _mm256_maddubs_epi16(a_unpacked[i][1], b1);
                        __m256i d2 = _mm256_maddubs_epi16(a_unpacked[i][2], b2);
                        __m256i d3 = _mm256_maddubs_epi16(a_unpacked[i][3], b3);
                        __m256i sum16 = _mm256_add_epi16(_mm256_add_epi16(d0, d1),
                                                         _mm256_add_epi16(d2, d3));
                        acc[i][j] = _mm256_add_epi32(acc[i][j], _mm256_madd_epi16(sum16, one16));
#endif
                    }
                }
            }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    __m128i sum128 = _mm_add_epi32(
                        _mm256_castsi256_si128(acc[i][j]),
                        _mm256_extractf128_si256(acc[i][j], 1));
                    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
                    __m128i sum64 = _mm_add_epi32(hi64, sum128);
                    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
                    s[(c + j) * bs + (r + i)] = (float)_mm_cvtsi128_si32(
                        _mm_add_epi32(sum64, hi32));
                }
            }
        }

        // Remaining B columns
        for (; c < nr; c++) {
            __m256i acc_rem[4];
            for (int i = 0; i < 4; i++)
                acc_rem[i] = _mm256_setzero_si256();

            for (int64_t bl = 0; bl < nb_128; bl++) {
                const int8_t * b_ptr = y + c * n + bl * 128;
                __m256i b0 = _mm256_loadu_si256((const __m256i *)(b_ptr));
                __m256i b1 = _mm256_loadu_si256((const __m256i *)(b_ptr + 32));
                __m256i b2 = _mm256_loadu_si256((const __m256i *)(b_ptr + 64));
                __m256i b3 = _mm256_loadu_si256((const __m256i *)(b_ptr + 96));

                for (int i = 0; i < 4; i++) {
                    const int64_t row_off = row_in_group + i;
                    uint8_t tmp_packed[32];
                    for (int sb = 0; sb < 4; sb++) {
                        const uint8_t * src = group_base + (bl * 4 + sb) * NROWS_GROUP * INTERLEAVE + row_off * INTERLEAVE;
                        memcpy(tmp_packed + sb * 8, src, 8);
                    }
                    __m256i packed = _mm256_loadu_si256((const __m256i *)tmp_packed);
                    __m256i a0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask_2bit);
                    __m256i a1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_2bit);
                    __m256i a2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask_2bit);
                    __m256i a3 = _mm256_and_si256(packed, mask_2bit);
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a0, b0);
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a1, b1);
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a2, b2);
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a3, b3);
#else
                    __m256i d0 = _mm256_maddubs_epi16(a0, b0);
                    __m256i d1 = _mm256_maddubs_epi16(a1, b1);
                    __m256i d2 = _mm256_maddubs_epi16(a2, b2);
                    __m256i d3 = _mm256_maddubs_epi16(a3, b3);
                    __m256i sum16 = _mm256_add_epi16(_mm256_add_epi16(d0, d1),
                                                     _mm256_add_epi16(d2, d3));
                    acc_rem[i] = _mm256_add_epi32(acc_rem[i], _mm256_madd_epi16(sum16, one16));
#endif
                }
            }

            for (int i = 0; i < 4; i++) {
                __m128i sum128 = _mm_add_epi32(
                    _mm256_castsi256_si128(acc_rem[i]),
                    _mm256_extractf128_si256(acc_rem[i], 1));
                __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
                __m128i sum64 = _mm_add_epi32(hi64, sum128);
                __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
                s[c * bs + (r + i)] = (float)_mm_cvtsi128_si32(
                    _mm_add_epi32(sum64, hi32));
            }
        }
    }

    // Remaining A rows (< 4) — fallback to non-interleaved single-row dot
    for (; r < nc; r++) {
        const int64_t abs_row = row_start + r;
        const int64_t group = abs_row / NROWS_GROUP;
        const int64_t row_in_group = abs_row % NROWS_GROUP;
        const uint8_t * group_base = x + group * group_stride;

        for (int64_t c2 = 0; c2 < nr; c2++) {
            int32_t sum = 0;
            for (int64_t bl = 0; bl < nb_128; bl++) {
                for (int sb = 0; sb < 4; sb++) {
                    const uint8_t * a_ptr = group_base + (bl * 4 + sb) * NROWS_GROUP * INTERLEAVE + row_in_group * INTERLEAVE;
                    const int8_t * b_ptr = y + c2 * n + bl * 128 + sb * 32;
                    for (int i = 0; i < 8; i++) {
                        uint8_t byte = a_ptr[i];
                        sum += ((byte >> 6) & 3) * b_ptr[i*4+0];
                        sum += ((byte >> 4) & 3) * b_ptr[i*4+1];
                        sum += ((byte >> 2) & 3) * b_ptr[i*4+2];
                        sum += ((byte >> 0) & 3) * b_ptr[i*4+3];
                    }
                }
            }
            s[c2 * bs + r] = (float)sum;
        }
    }
#else
    // Scalar fallback — not implemented for interleaved layout
    GGML_ASSERT(false && "interleaved I2_S GEMM requires AVX2");
    GGML_UNUSED(n); GGML_UNUSED(s); GGML_UNUSED(bs);
    GGML_UNUSED(vx_base); GGML_UNUSED(row_start); GGML_UNUSED(a_row_bytes);
    GGML_UNUSED(vy); GGML_UNUSED(nr); GGML_UNUSED(nc);
#endif
}

void ggml_gemv_i2_i8_s_interleaved(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx_base, int64_t row_start, int64_t a_row_bytes,
    const void * GGML_RESTRICT vy, int nr, int nc) {
    // GEMV is just GEMM with nr=1
    ggml_gemm_i2_i8_s_interleaved(n, s, bs, vx_base, row_start, a_row_bytes, vy, nr, nc);
}
