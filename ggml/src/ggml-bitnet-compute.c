/*
 * BitNet computation functions
 * 
 * Extracted from merge-dev branch's ggml/src/ggml.c
 * These functions implement BitNet-specific computation paths:
 * - I2_S quantization/dequantization helpers
 * - T-MAC (Table-lookup Multiply-ACcumulate) ARM TL1 path
 * - T-MAC x86 TL2 path
 * - I2_S mul_mat special handling
 * - I2_S get_rows support
 *
 * This file is meant to be #included from ggml.c when 
 * GGML_BITNET_ARM_TL1 or GGML_BITNET_X86_TL2 is defined.
 */

// ====================== BitNet Helper Functions ======================

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

void float_act_quant(const int K, float* B, int32_t* dst, float* act_scale) {
    double min = 0.00001;
    double max = min;
    for (int i = 0; i < K; ++i) {
        max = MAX(max, (double)fabs((double)B[i]));
    }
    float s = 127 / max;
    act_scale[0] = s;
    for (int i = 0; i < K; ++i) {
        int v = nearest_int(B[i] * s);
        if (v >  127) v = 127;
        if (v < -128) v = -128;
        dst[i] = (int32_t)v;
    }
}

void weight_quant_f32(const int M, const int K, float* A, int32_t* dst, float* i2_scale) {
    double max = 0.00001;

    for (int i = 0; i < M * K; i++) {
        if (fabs(A[i]) > max){
            i2_scale[0] = fabs(A[i]);
            break;
        }
    }
    
    for (int i = 0; i < M * K; ++i) {
        if (fabs((double)(A[i])) < 1e-6) {
            dst[i] = 0;
            continue;
        } else {
            dst[i] = (double)A[i] * i2_scale[0] > 0 ? 1 : -1;
        }
    }
}

void weight_quant_f16(const int M, const int K, uint16_t* A, int32_t* dst, float* i2_scale) {
    double max = 0.00001;
    for (int i = 0; i < M * K; i++) {
        float temp_A = GGML_FP16_TO_FP32(A[i]);
        if (fabs(temp_A) > max){
            i2_scale[0] = fabs(temp_A);
            break;
        }
    }
    
    for (int i = 0; i < M * K; ++i) {
        float temp_A = GGML_FP16_TO_FP32(A[i]);
        if (fabs((double)(temp_A)) < 1e-6) {
            dst[i] = 0;
            continue;
        } else {
            dst[i] = (double)temp_A * i2_scale[0] > 0 ? 1 : -1;
        }
    }
}

void matrixMultiply_int(const int M, const int N, const int K, const int32_t* A, const int32_t* B, int32_t* C) {


// ====================== I2_S mul_mat one_chunk path ======================

// This code belongs inside ggml_compute_forward_mul_mat_one_chunk()

// after the normal vec_dot loop, guarded by (src0->type == GGML_TYPE_I2_S)


    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    const float * scale      = (float * )((uint8_t*) (src0->data) + (ne00 * ne01 / 4));
    const float * act_scales = (const float*) ((const char *) wdata + (ne11 * ne10));
    const int32_t * act_sums   = (const int32_t*) ((const char *) act_scales + (ne11) * sizeof(float));

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                // hack
                const char * src1_col_de = (const char*)wdata + (i11 * nb11 / 4);

                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                // for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                //     if (src0->type == GGML_TYPE_I2_S) {
                //         vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01 / 4, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col_de, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                //         tmp[ir0 - iir0] = (tmp[ir0 - iir0]  - act_sums[i1]) / (act_scales[i1]) * (*scale);
                //     } else {
                //         vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                //     }
                // }

                if (src0->type == GGML_TYPE_I2_S && iir0 + blck_0 - 1 < ir0_end) {
                    // 16 rows per vector dot product, so we can process 16 rows at a time blck == 16
                    // this is a bit of a hack, we should probably have a better way to handle this
                    vec_dot(ne00, &tmp[0], 1, 
                        src0_row + iir0 * nb01 / 4, nb01, 
                        src1_col_de, 0, 16);
                    
                    // post compute activation scaling
                    for (int row = 0; row < 16; row++) {
                        tmp[row] = (tmp[row] - act_sums[i1]) / (act_scales[i1]) * (*scale);
                    }
                }
                else
                {
                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                        if (src0->type == GGML_TYPE_I2_S) {
                            vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_row + ir0 * nb01 / 4, 0, src1_col_de, 0, 1);
                            tmp[ir0 - iir0] = (tmp[ir0 - iir0]  - act_sums[i1]) / (act_scales[i1]) * (*scale);
                        } else {
                            vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                        }
                    }
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}



// ====================== T-MAC / BitNet mul_mat paths ======================

// This code belongs inside ggml_compute_forward_mul_mat()

// These are alternative mul_mat implementations using T-MAC lookup tables

// #ifndef GGML_BITNET_X86_TL2
//     if (src1->ne[1] <= 1 && src0->type != GGML_TYPE_TL1 && src0->type != GGML_TYPE_I2_S && src0->type != GGML_TYPE_TQ1_0 && src0->type != GGML_TYPE_TQ2_0 && src0->ne[1] != 32002 && src0->ne[1] != 96 && src0->ne[0] != 96) {
//         int32_t* int_C = (int32_t*)malloc(1 * src0->ne[1] * sizeof(int32_t));
//         for (int i = 0; i < src0->ne[1] * 1; i++) {
//             int_C[i] = 0;
//         }
//         float* act_scale = (float*)malloc(sizeof(float));
//         float* i2_scale = (float*)malloc(sizeof(float));
//         int32_t* int_B = (int32_t*)malloc(1 * src0->ne[0] * sizeof(int32_t));
//         int32_t* int_A = (int32_t*)malloc(src0->ne[0] * src0->ne[1] * sizeof(int32_t));
//         float_act_quant(src1->ne[0], (float*)src1->data, int_B, act_scale);
//         if (src0->type == 0) {
//             weight_quant_f32(src0->ne[1], src0->ne[0], src0->data, int_A, i2_scale);
//         } else if (src0->type == 1) {
//             weight_quant_f16(src0->ne[1], src0->ne[0], src0->data, int_A, i2_scale);
//         }   
//         matrixMultiply_int(src0->ne[1], src1->ne[1], src0->ne[0], int_A, int_B, int_C);
//         for (int i=0; i < src0->ne[1] * 1; i++) {
//             ((float*)(dst->data))[i] = int_C[i] / act_scale[0] * i2_scale[0];
//         }
//         free(int_A);
//         free(int_B);
//         free(int_C);
//         free(act_scale);
//         free(i2_scale);
//         return;
//     }
// #endif
    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows
#if defined(GGML_BITNET_ARM_TL1)
    if (ggml_bitnet_can_mul_mat(src0, src1, dst)) {

        const int bits = ggml_bitnet_get_type_bits(type);
        // src0: weight,     ne00 = k, ne01 = n
        // src1: activation, ne10 = k, ne11 = m
        char * wdata = params->wdata;

        struct bitnet_tensor_extra * wt = src0->extra;
        char * cur_wdata = wdata;
        bitnet_float_type * bitnet_f_ptr = wdata;
        if (sizeof(bitnet_float_type) == 2) {
            cur_wdata = wdata + MAX(ne10, ne01) * ne11 * sizeof(bitnet_float_type);
        };
        int8_t * qlut = cur_wdata;
        bitnet_float_type * lut_scales = (bitnet_float_type *) (qlut + ne10 * ne11 * 16);
        bitnet_float_type * lut_biases = (bitnet_float_type *) (lut_scales + wt->lut_scales_size * ne11);

        // g = 4
        if (ith == 0) {
            // Transform tensor if not already transformed
            // Although we have done this in file `llama.cpp`,
            // we still need to do it here for non-model inference, e.g., test-backend-ops.cpp.
            // It's better to do this in ggml-backend.c,
            // but llama.cpp directly manipulates tensor.data for cbe in a lot of space.
            ggml_bitnet_transform_tensor(src0);
            GGML_ASSERT(src1->type == GGML_TYPE_F32);
            bitnet_float_type * act_input;
            if (sizeof(bitnet_float_type) == 2) {
                ggml_fp32_to_fp16_row(src1->data, bitnet_f_ptr, ne10 * ne11);
                act_input = bitnet_f_ptr;
            } else {
                act_input = src1->data;
            }
            ggml_preprocessor(ne01, ne10, act_input, lut_scales, qlut);
        }

        ggml_barrier(params->threadpool);

        bitnet_float_type * act_output;
        if (sizeof(bitnet_float_type) == 2) {
            act_output = bitnet_f_ptr;
        } else {
            act_output = dst->data;
        }
        const int n_tile_num = wt->n_tile_num;
        GGML_ASSERT(ne0 % n_tile_num == 0);
        const int w_size           = ne00 * ne01 / 4;
        const int w_tile_size      = w_size / n_tile_num;
        const int c_size           = ne01 * ne11;
        const int c_tile_size      = c_size / n_tile_num;
        const int lut_size         = ne11 * 16 * (ne10 / 2) * 2; // int8
        const int lut_tile_size    = lut_size / n_tile_num;

        const int th_tile_num = (n_tile_num + nth - 1) / nth;
        const int th_tile_beg = ith * th_tile_num;
        const int th_tile_end = MIN((ith + 1) * th_tile_num, n_tile_num);

        for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
            const int w_offset          = i_tile * w_tile_size;
            const int scales_offset     = 0;

            const int qlut_offset       = 0;
            const int lut_scales_offset = 0;
            const int dst_offset        = i_tile * c_tile_size;

            ggml_qgemm_lut( ne01, ne00, ((uint8_t *)(wt->qweights) + w_offset), 
                            qlut, 
                            wt->scales + scales_offset, 
                            lut_scales + lut_scales_offset, 
                            act_output + dst_offset);
            if (sizeof(bitnet_float_type) == 2) {
                ggml_fp16_to_fp32_row(act_output + dst_offset, (float *) dst->data + dst_offset, ne01 / n_tile_num);
            }
        }
        return;
    }
#endif
#if defined(GGML_BITNET_X86_TL2)
    if (ggml_bitnet_can_mul_mat(src0, src1, dst)) {
        // src0: weight,     ne00 = k, ne01 = n
        // src1: activation, ne10 = k, ne11 = m
        char * wdata = params->wdata;

        struct bitnet_tensor_extra * wt = src0->extra;
        char * cur_wdata = wdata;
        bitnet_float_type * bitnet_f_ptr = wdata;
        if (sizeof(bitnet_float_type) == 2) {
            cur_wdata = wdata + MAX(ne10, ne01) * ne11 * sizeof(bitnet_float_type);
        };
        int8_t * three_qlut = cur_wdata;
        bitnet_float_type * lut_scales;
        int8_t * two_qlut;
        const int total_k = ne10;
        const int three_k = (int)(total_k / wt->BK) * wt->BK;
        const int two_k = total_k - three_k;

        lut_scales = (bitnet_float_type *) (three_qlut + three_k / 3 * 16 * 2 * ne11);
        two_qlut = (int8_t *) (lut_scales + ne11);

        // g = 4
        if (ith == 0) {
            // Transform tensor if not already transformed
            // Although we have done this in file `llama.cpp`,
            // we still need to do it here for non-model inference, e.g., test-backend-ops.cpp.
            // It's better to do this in ggml-backend.c,
            // but llama.cpp directly manipulates tensor.data for cbe in a lot of space.
            ggml_bitnet_transform_tensor(src0);
            GGML_ASSERT(src1->type == GGML_TYPE_F32);
            bitnet_float_type * act_input;
            if (sizeof(bitnet_float_type) == 2) {
                ggml_fp32_to_fp16_row(src1->data, bitnet_f_ptr, ne10 * ne11);
                act_input = bitnet_f_ptr;
            } else {
                act_input = src1->data;
            }
            ggml_preprocessor(ne11, ne01, three_k, two_k, act_input, lut_scales, three_qlut, two_qlut);
        }

        ggml_barrier(params->threadpool);

        bitnet_float_type * act_output;
        if (sizeof(bitnet_float_type) == 2) {
            act_output = bitnet_f_ptr;
        } else {
            act_output = dst->data;
        }

        const int n_tile_num = wt->n_tile_num;
        GGML_ASSERT(ne0 % n_tile_num == 0);
        const int w_size           = three_k * ne01 / (2 * 3);
        const int w_tile_size      = w_size / n_tile_num;
        const int c_size           = ne01;
        const int c_tile_size      = c_size / n_tile_num;
        const int sign_size        = three_k * ne01 / 24;
        const int sign_tile_size   = sign_size / n_tile_num;

        const int th_tile_num = (n_tile_num + nth - 1) / nth;
        const int th_tile_beg = ith * th_tile_num;
        const int th_tile_end = MIN((ith + 1) * th_tile_num, n_tile_num);

        uint8_t* sign = ((uint8_t *)(wt->qweights)) + three_k * ne01 / 3 / 2;

        if (ne11 > 1) {
            // printf("ne11:%d\n", ne11);
            int iter = ne11;
            int bs512_num = iter / 512;
            iter = iter - 512 * bs512_num;
            int bs256_num = iter / 256;
            iter = iter - 256 * bs256_num;
            int bs128_num = iter / 128;
            iter = iter - 128 * bs128_num;
            int bs32_num = iter / 32;
            iter = iter - 32 * bs32_num;
            int bs8_num = iter / 8;
            iter = iter - 8 * bs8_num;
            int bs1_num = iter / 1;

        // bs 512
        for (int i = 0; i < bs512_num; i++) {

        for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
            const int w_offset          = i_tile * w_tile_size;
            const int sign_offset       = i_tile * sign_tile_size;
            const int dst_offset        = i_tile * c_tile_size;

            ggml_qgemm_lut( 512, ne01, ne00, three_k, ((uint8_t *)(wt->qweights) + w_offset),
                            sign + sign_offset,
                            three_qlut + i * 512 * three_k / 3 * 32,
                            wt->scales,
                            lut_scales + i * 512,
                            act_output + dst_offset + i * 512 * ne01);
        }

        const int two_w_size           = ne01 * two_k / (2 * 2); // int8 
        const int two_w_tile_size      = two_w_size / n_tile_num;
        uint8_t* two_A = ((uint8_t *)(wt->qweights)) + three_k * ne01 / 3 / 2 + three_k * ne01 / 3 / 8;
        // auto gemm_start = std::chrono::high_resolution_clock::now();
        for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
            const int two_w_offset          = i_tile * two_w_tile_size;
            const int two_dst_offset        = i_tile * c_tile_size;

            ggml_qgemm_lut( 512, ne01, ne00, two_k, two_A + two_w_offset, 
                            NULL,
                            two_qlut + i * 512 * two_k / 2 * 32,
                            wt->scales,
                            lut_scales + i * 512,
                            act_output + two_dst_offset + i * 512 * ne01);
        }   
         
        }

        // bs 256
        for (int i = 0; i < bs256_num; i++) {

        for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
            const int w_offset          = i_tile * w_tile_size;
            const int sign_offset       = i_tile * sign_tile_size;
            const int dst_offset        = i_tile * c_tile_size;

            ggml_qgemm_lut( 256, ne01, ne00, three_k, ((uint8_t *)(wt->qweights) + w_offset),
                            sign + sign_offset,
                            three_qlut + bs512_num * 512 * three_k / 3 * 32 + i * 256 * three_k / 3 * 32,
                            wt->scales,
                            lut_scales + bs512_num * 512 + i * 256,
                            act_output + dst_offset + bs512_num * 512 * ne01 + i * 256 * ne01);
        }

        const int two_w_size           = ne01 * two_k / (2 * 2); // int8 
        const int two_w_tile_size      = two_w_size / n_tile_num;
        uint8_t* two_A = ((uint8_t *)(wt->qweights)) + three_k * ne01 / 3 / 2 + three_k * ne01 / 3 / 8;
        // auto gemm_start = std::chrono::high_resolution_clock::now();
        for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
            const int two_w_offset          = i_tile * two_w_tile_size;
            const int two_dst_offset        = i_tile * c_tile_size;

            ggml_qgemm_lut( 256, ne01, ne00, two_k, two_A + two_w_offset, 
                            NULL,
                            two_qlut + bs512_num * 512 * two_k / 2 * 32 + i * 256 * two_k / 2 * 32,
                            wt->scales,
                            lut_scales + bs512_num * 512 + i * 256,
                            act_output + two_dst_offset + bs512_num * 512 * ne01 + i * 256 * ne01);
        }   
         
        }

        // bs 128
        // printf("128:%d\n", bs128_num);
        for (int i = 0; i < bs128_num; i++) {

        for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
            const int w_offset          = i_tile * w_tile_size;
            const int sign_offset       = i_tile * sign_tile_size;
            const int dst_offset        = i_tile * c_tile_size;

            ggml_qgemm_lut( 128, ne01, ne00, three_k, ((uint8_t *)(wt->qweights) + w_offset),
                            sign + sign_offset,
                            three_qlut + bs512_num * 512 * three_k / 3 * 32 + bs256_num * 256 * three_k / 3 * 32 + i * 128 * three_k / 3 * 32,
                            wt->scales,
                            lut_scales + bs512_num * 512 + bs256_num * 256 + i * 128,
                            act_output + dst_offset + bs512_num * 512 * ne01 + bs256_num * 256 * ne01 + i * 128 * ne01);
        }

        const int two_w_size           = ne01 * two_k / (2 * 2); // int8 
        const int two_w_tile_size      = two_w_size / n_tile_num;
        uint8_t* two_A = ((uint8_t *)(wt->qweights)) + three_k * ne01 / 3 / 2 + three_k * ne01 / 3 / 8;
        // auto gemm_start = std::chrono::high_resolution_clock::now();
        for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
            const int two_w_offset          = i_tile * two_w_tile_size;
            const int two_dst_offset        = i_tile * c_tile_size;

            ggml_qgemm_lut( 128, ne01, ne00, two_k, two_A + two_w_offset, 
                            NULL,
                            two_qlut + bs512_num * 512 * two_k / 2 * 32 + bs256_num * 256 * two_k / 2 * 32 + i * 128 * two_k / 2 * 32,
                            wt->scales,
                            lut_scales + bs512_num * 512 + bs256_num * 256 + i * 128,
                            act_output + two_dst_offset + bs512_num * 512 * ne01 + bs256_num * 256 * ne01 + i * 128 * ne01);
        }   
         
        }
        // printf("128end\n");

        // bs 32
        // printf("32:%d\n", bs32_num);
        for (int i = 0; i < bs32_num; i++) {

        for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
            const int w_offset          = i_tile * w_tile_size;
            const int sign_offset       = i_tile * sign_tile_size;
            const int dst_offset        = i_tile * c_tile_size;

            ggml_qgemm_lut( 32, ne01, ne00, three_k, ((uint8_t *)(wt->qweights) + w_offset),
                            sign + sign_offset,
                            three_qlut + bs512_num * 512 * three_k / 3 * 32 + bs256_num * 256 * three_k / 3 * 32\
                             + bs128_num * 128 * three_k / 3 * 32 + i * 32 * three_k / 3 * 32,
                            wt->scales,
                            lut_scales + bs512_num * 512 + bs256_num * 256 + bs128_num * 128 + i * 32,
                            act_output + dst_offset + bs512_num * 512 * ne01 + bs256_num * 256 * ne01 + bs128_num * 128 * ne01 + i * 32 * ne01);
        }

        const int two_w_size           = ne01 * two_k / (2 * 2); // int8 
        const int two_w_tile_size      = two_w_size / n_tile_num;
        uint8_t* two_A = ((uint8_t *)(wt->qweights)) + three_k * ne01 / 3 / 2 + three_k * ne01 / 3 / 8;
        // auto gemm_start = std::chrono::high_resolution_clock::now();
        for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
            const int two_w_offset          = i_tile * two_w_tile_size;
            const int two_dst_offset        = i_tile * c_tile_size;

            ggml_qgemm_lut( 32, ne01, ne00, two_k, two_A + two_w_offset, 
                            NULL,
                            two_qlut + bs512_num * 512 * two_k / 3 * 32 + bs256_num * 256 * two_k / 3 * 32\
                             + bs128_num * 128 * two_k / 2 * 32 + i * 32 * two_k / 2 * 32,
                            wt->scales,
                            lut_scales + bs512_num * 512 + bs256_num * 256 + bs128_num * 128 + i * 32,
                            act_output + two_dst_offset + bs512_num * 512 * ne01 + bs256_num * 256 * ne01 + bs128_num * 128 * ne01 + i * 32 * ne01);
        }   
         
        }
        // printf("32end\n");
   
        // bs 8
        // printf("8:%d\n", bs8_num);


// ====================== get_rows I2_S ======================

static void ggml_compute_forward_get_rows_i2_s(
            struct ggml_compute_params * params,
            struct ggml_tensor * dst) {

    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = ggml_nelements(src1);

    const enum ggml_type type = src0->type;

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == ggml_type_size(type));
    assert(ggml_nrows(dst) == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);



// ====================== I2_S activation quantize (mul_mat prep) ======================

// This code belongs in the preparation section of ggml_compute_forward_mul_mat()

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        float* act_scales = (float*) ((char *) wdata + (ne11 * ne10));
        int32_t* act_sums = (int32_t*) ((char *) act_scales + (ne11) * sizeof(float));

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                int64_t i11_processed = 0;
                if ((ggml_n_dims(src1) == 2) && from_float_to_mat && gemm) {
                    for (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
                        from_float_to_mat((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                                          (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                          4, ne10, blck_size_interleave);
                    }
                    i11_processed = ne11 - ne11 % 4;
                }
                for (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
                    if (src0->type == GGML_TYPE_I2_S) {
                        quantize_row_i8_s((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11), (void *) (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1), ne10, act_scales + i11, act_sums + i11);
                    } else {
                        from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                        (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                        ne10);
                    }
                }
            }
        }
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE
    if (src1->type != vec_dot_type) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     ith, nth,
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
    int64_t num_rows_per_vec_dot = vec_dot_num_rows;
    // TODO: currently the mmla kernels support only even numbered rows/cols.
    // this check can be removed once they are extended to support odd numbered rows/cols too
    if ((nr0 % 2 != 0) || (ne11 % 2 != 0)) {
        num_rows_per_vec_dot = 1;
    }

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggerganov/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    // if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    // }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    if ((ggml_n_dims(src0) == 2) && gemv) {
        const void * src1_wdata      = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t src1_col_stride = ggml_is_contiguous(src1) || src1->type != vec_dot_type ? ggml_row_size(vec_dot_type, ne10) : nb11;
        int64_t src0_start = (ith * ne01) / nth;
        int64_t src0_end   = ((ith + 1) * ne01) / nth;
        src0_start = (src0_start % matmul_num_cols) ? src0_start + matmul_num_cols - (src0_start % matmul_num_cols): src0_start;
        src0_end   = (src0_end   % matmul_num_cols) ? src0_end   + matmul_num_cols - (src0_end   % matmul_num_cols): src0_end;
        if (src0_start >= src0_end) return;

        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        if (gemm && (ne11 > 3)) {
            if (src0->type == GGML_TYPE_I2_S) {
                float tmp[(src0_end - src0_start)*(ne11 - ne11 % 4)];
                const float * scale      = (float * )((uint8_t*) (src0->data) + (ne00 * ne01 / 4));
                const float * act_scales = (const float*) ((const char *) src1_wdata + (ne11 * ne10));
                const int32_t * act_sums   = (const int32_t*) ((const char *) act_scales + (ne11) * sizeof(float));
                gemm(ne00, &tmp[0], src0_end - src0_start, (const char *) src0->data + src0_start * nb01 / 4,
                    (const char *) src1_wdata, ne11 - ne11 % 4, src0_end - src0_start);
                for (int col = 0; col < ne11 - ne11 % 4; col++) {
                    for (int row = 0; row < src0_end - src0_start; row++) {
                        tmp[col * (src0_end - src0_start) + row] = (tmp[col * (src0_end - src0_start) + row] - act_sums[col]) / (act_scales[col]) * (*scale);
                    }
                    memcpy((float *)((char *) dst->data + (col * nb1)) + src0_start, tmp + col * (src0_end - src0_start), (src0_end - src0_start) * sizeof(float));
                }
            }
            else {
                gemm(ne00, (float *)((char *) dst->data) + src0_start, ne01, (const char *) src0->data + src0_start * nb01,
                    (const char *) src1_wdata, ne11 - ne11 % 4, src0_end - src0_start);
            }
        }
        for (int iter = gemm ? ne11 - ne11 % 4 : 0; iter < ne11; iter++) {
            if (src0->type == GGML_TYPE_I2_S) {
                float tmp[src0_end - src0_start];
                const float * scale      = (float * )((uint8_t*) (src0->data) + (ne00 * ne01 / 4));
                const float * act_scales = (const float*) ((const char *) src1_wdata + (ne11 * ne10));
                const int32_t * act_sums   = (const int32_t*) ((const char *) act_scales + (ne11) * sizeof(float));
                gemv(ne00, &tmp[0], ne01,
                    (const char *) src0->data + src0_start * nb01 / 4,
                    (const char *) src1_wdata + (src1_col_stride * iter),
                    1, src0_end - src0_start);
                for (int row = 0; row < src0_end - src0_start; row++) {
                    tmp[row] = (tmp[row] - act_sums[iter]) / (act_scales[iter]) * (*scale);
                }
                memcpy((float *)((char *) dst->data + (iter * nb1)) + src0_start, tmp, (src0_end - src0_start) * sizeof(float));
            }
            else {
                gemv(ne00, (float *)((char *) dst->data + (iter * nb1)) + src0_start, ne01,
                (const char *) src0->data + src0_start * nb01, (const char *) src1_wdata + (src1_col_stride * iter), 1,
                src0_end - src0_start);
            }
        }
        return;


// ====================== BitNet graph compute wsize ======================

// This code belongs in ggml_graph_compute_thread() work size calculation

                } break;
            case GGML_OP_MUL_MAT:
                {
                    const enum ggml_type vec_dot_type = type_traits[node->src[0]->type].vec_dot_type;

#if defined(GGML_BITNET_ARM_TL1) || defined(GGML_BITNET_X86_TL2)
                    if (ggml_bitnet_can_mul_mat(node->src[0], node->src[1], node)) {
                        cur = ggml_bitnet_mul_mat_get_wsize(node->src[0], node->src[1], node);
                    } else
#endif
                    if (node->src[1]->type != vec_dot_type) {
                        if (vec_dot_type == GGML_TYPE_I8_S) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1])) + node->src[1]->ne[1] * sizeof(float) + node->src[1]->ne[1] * sizeof(int32_t);

