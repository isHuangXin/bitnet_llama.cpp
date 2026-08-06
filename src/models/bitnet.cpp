#include "models.h"

void llama_model_bitnet::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // YOCO-U: read self-decoder iteration count (T), default 1
    ml.get_key(LLM_KV_YOCO_U_ITERS, hparams.yoco_u_iters, false);

    if (hparams.yoco_u_iters > 1) {
        // YOCO-U: self-decoder layers use SWA via sliding_window parameter
        // Cross-decoder layers use shared global KV (no per-layer cache)
        ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
        if (hparams.n_swa == 0) {
            hparams.n_swa = 512;
        }

        const uint32_t n_layer_total = hparams.n_layer();
        const uint32_t n_cross = n_layer_total / (hparams.yoco_u_iters + 1);
        const uint32_t n_self_unrolled = n_cross * hparams.yoco_u_iters;

        // Only self-decoder layers (0-20) get KV cache
        // Cross-decoder layers (21-27) use no-cache shared KV attention
        hparams.n_layer_kv_from_start = (int32_t)n_self_unrolled;

        // Note: swa_type remains NONE — we use regular KV cache with sliding_window
        // The sliding_window parameter applies window-based masking to all KV cache layers
        // This is functionally equivalent to SWA for self-decoder layers
    }

    switch (hparams.n_layer()) {
        case 14: type = LLM_TYPE_1_5B; break;
        case 26: type = LLM_TYPE_3B; break;
        case 28: type = LLM_TYPE_3B; break; // YOCO-U: 14 layers x T=3 self-decoder iters = 28 logical layers
        case 30: type = LLM_TYPE_2B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_bitnet::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head_i    = hparams.n_head(i);
        const int64_t n_head_kv_i = hparams.n_head_kv(i);
        const int64_t q_dim_i     = n_head_i * n_embd_head_k;
        const int64_t kv_dim_i    = n_head_kv_i * n_embd_head_k;

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sub_norm = create_tensor(tn(LLM_TENSOR_ATTN_SUB_NORM, "weight", i), {q_dim_i}, 0);

        layer.wq       = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, q_dim_i}, 0);
        layer.wq_s     = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
        layer.wk       = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, kv_dim_i}, 0);
        layer.wk_s     = create_tensor(tn(LLM_TENSOR_ATTN_K,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
        layer.wv       = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, kv_dim_i}, 0);
        layer.wv_s     = create_tensor(tn(LLM_TENSOR_ATTN_V,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
        layer.wo       = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {q_dim_i, n_embd}, 0);
        layer.wo_s     = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale",  i), {1}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm     = create_tensor(tn(LLM_TENSOR_FFN_NORM,     "weight", i), {n_embd}, 0);
        layer.ffn_sub_norm = create_tensor(tn(LLM_TENSOR_FFN_SUB_NORM, "weight", i), {n_ff}, 0);

        layer.ffn_gate       = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_gate_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "scale",  i), {1}, TENSOR_NOT_REQUIRED);
        layer.ffn_down       = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale",  i), {1}, TENSOR_NOT_REQUIRED);
        layer.ffn_up         = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_s   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
    }

    // YOCO shared cross KV tensors (model-level, optional)
    const int64_t kv_dim = hparams.n_head_kv() * n_embd_head_k;
    yoco_cross_kv_norm = create_tensor(tn(LLM_TENSOR_YOCO_CROSS_KV_NORM, "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    yoco_cross_k       = create_tensor(tn(LLM_TENSOR_YOCO_CROSS_K,       "weight"), {n_embd, kv_dim}, TENSOR_NOT_REQUIRED);
    yoco_cross_v       = create_tensor(tn(LLM_TENSOR_YOCO_CROSS_V,       "weight"), {n_embd, kv_dim}, TENSOR_NOT_REQUIRED);
}

std::unique_ptr<llm_graph_context> llama_model_bitnet::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_bitnet::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const uint32_t yoco_u_iters = hparams.yoco_u_iters;
    const bool is_yoco_u = (yoco_u_iters > 1);

    // For YOCO-U: determine self/cross boundary
    const int n_cross = is_yoco_u ? (n_layer / (yoco_u_iters + 1)) : 0;           // 7
    const int n_self_unrolled = is_yoco_u ? (n_cross * (int)yoco_u_iters) : n_layer; // 21

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    // For YOCO-U cross-decoder: no-cache attention
    auto * inp_attn_nc = is_yoco_u ? build_attn_inp_no_cache() : nullptr;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Shared KV for cross-decoder (computed after self-decoder finishes)
    ggml_tensor * shared_K = nullptr;
    ggml_tensor * shared_V = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        const int64_t n_head_il    = hparams.n_head(il);
        const int64_t n_head_kv_il = hparams.n_head_kv(il);

        const bool is_cross_layer = is_yoco_u && (il >= n_self_unrolled);

        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // attention
        {
            if (!is_cross_layer) {
                // ===== Self-Decoder: normal per-layer KV cache attention =====
                auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                        n_embd_head, n_head_il, n_head_kv_il, il);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        NULL, NULL, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            } else {
                // ===== Cross-Decoder: only compute Q, reuse shared K/V =====
                GGML_ASSERT(shared_K != nullptr && shared_V != nullptr);

                // Compute Q only
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
                cb(Qcur, "Qcur_pre", il);
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head_il, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);

                // Use no-cache attention with shared K/V
                cur = build_attn(inp_attn_nc,
                        NULL, NULL, NULL,
                        Qcur, shared_K, shared_V, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            cur = build_norm(cur,
                    model.layers[il].attn_sub_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_sub_norm", il);

            cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
            if (model.layers[il].wo_b) {
                cur = ggml_add(ctx0, cur, model.layers[il].wo_b);
            }
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward forward
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                NULL,                      NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_sub_out", il);

        cur = build_norm(cur,
                model.layers[il].ffn_sub_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_sub_norm", il);

        cur = build_lora_mm(model.layers[il].ffn_down, cur, model.layers[il].ffn_down_s);
        cb(cur, "ffn_down", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;

        // YOCO-U: compute shared global KV after self-decoder finishes
        if (is_yoco_u && il == n_self_unrolled - 1 && model.yoco_cross_kv_norm) {
            const int64_t n_head_kv_cross = hparams.n_head_kv(n_self_unrolled); // kv heads for cross layers

            // K̂ = LN(self_decoder_output) × W_K_shared
            ggml_tensor * shared_kv_normed = build_norm(inpL,
                    model.yoco_cross_kv_norm, NULL,
                    LLM_NORM_RMS, -1);
            cb(shared_kv_normed, "yoco_cross_kv_norm", -1);

            shared_K = build_lora_mm(model.yoco_cross_k, shared_kv_normed);
            shared_K = ggml_reshape_3d(ctx0, shared_K, n_embd_head, n_head_kv_cross, n_tokens);
            shared_K = ggml_rope_ext(
                    ctx0, shared_K, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(shared_K, "yoco_shared_K", -1);

            // V̂ = LN(self_decoder_output) × W_V_shared
            shared_V = build_lora_mm(model.yoco_cross_v, shared_kv_normed);
            shared_V = ggml_reshape_3d(ctx0, shared_V, n_embd_head, n_head_kv_cross, n_tokens);
            cb(shared_V, "yoco_shared_V", -1);
        }
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    // FIXME: do not use model.tok_embd directly, duplicate as model.output
    cur = build_lora_mm(model.tok_embd, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

void llama_model_bitnet_b158::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 24: type = LLM_TYPE_700M; break;
        case 26: type = LLM_TYPE_3B;   break;
        case 30: type = LLM_TYPE_2B;   break; // bitnet2b_2501
        default: type = LLM_TYPE_UNKNOWN;
    }
}
