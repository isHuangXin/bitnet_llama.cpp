#include "models.h"

void llama_model_bitnet::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // YOCO-U: read self-decoder iteration count (T), default 1
    ml.get_key(LLM_KV_YOCO_U_ITERS, hparams.yoco_u_iters, false);

    // Read sliding window for YOCO self-decoder layers
    // Note: n_swa is stored but swa_type is left as NONE —
    // YOCO uses a plain KV cache; the window is only informational.
    // Cross-decoder layers use shared KV without cache (no_cache mode).
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);

    // MoE: read expert FFN dimension from custom key
    {
        uint32_t moe_ffn_dim = 0;
        if (ml.get_key("bitnet.moe_ffn_dim", moe_ffn_dim, false)) {
            hparams.n_ff_exp = moe_ffn_dim;
        }
    }

    switch (hparams.n_layer()) {
        case 20: type = LLM_TYPE_UNKNOWN; break; // YOCO-MoE 20 layers
        case 26: type = LLM_TYPE_3B; break;
        case 28: type = LLM_TYPE_3B; break;
        case 30: type = LLM_TYPE_2B; break;
        case 40: type = LLM_TYPE_UNKNOWN; break; // YOCO-U-MoE 40 layers
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_bitnet::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp      = hparams.n_ff_exp > 0 ? hparams.n_ff_exp : n_ff;
    const bool    has_moe       = (n_expert > 0);

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    // Embedding projections (optional)
    emb_proj_in  = create_tensor(tn(LLM_TENSOR_EMB_PROJ_IN,  "weight"), {n_embd, n_embd}, TENSOR_NOT_REQUIRED);
    emb_proj_out = create_tensor(tn(LLM_TENSOR_EMB_PROJ_OUT, "weight"), {n_embd, n_embd}, TENSOR_NOT_REQUIRED);
    emb_in_norm  = create_tensor(tn(LLM_TENSOR_EMB_IN_NORM,  "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    emb_out_norm = create_tensor(tn(LLM_TENSOR_EMB_OUT_NORM, "weight"), {n_embd}, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head_i    = hparams.n_head(i);
        const int64_t n_head_kv_i = hparams.n_head_kv(i);
        const int64_t q_dim_i     = n_head_i * n_embd_head_k;
        const int64_t kv_dim_i    = n_head_kv_i * n_embd_head_k;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, q_dim_i}, 0);
        // K/V proj: present in self-decoder, absent in cross-decoder
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, kv_dim_i}, TENSOR_NOT_REQUIRED);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, kv_dim_i}, TENSOR_NOT_REQUIRED);
        // diff_v3: o_proj output dim = real_heads * head_dim = (n_head_i/2) * head_dim
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head_i / 2 * n_embd_head_k, n_embd}, 0);

        // diff_v3 gate: [n_embd, n_head_i]
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head_i}, TENSOR_NOT_REQUIRED);

        // QK norms
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {q_dim_i}, TENSOR_NOT_REQUIRED);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {kv_dim_i}, TENSOR_NOT_REQUIRED);

        // FFN norm
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (has_moe) {
            // MoE FFN
            layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2 * n_ff_exp, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

            // Shared expert
            layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp}, TENSOR_NOT_REQUIRED);
            layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp, n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), {n_embd, 1}, TENSOR_NOT_REQUIRED);
        } else {
            // Dense FFN
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        }
    }

    // YOCO shared cross KV tensors (model-level, optional)
    const int64_t kv_dim = hparams.n_head_kv() * n_embd_head_k;
    yoco_cross_kv_norm = create_tensor(tn(LLM_TENSOR_YOCO_CROSS_KV_NORM, "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    yoco_cross_k       = create_tensor(tn(LLM_TENSOR_YOCO_CROSS_K,       "weight"), {n_embd, kv_dim}, TENSOR_NOT_REQUIRED);
    yoco_cross_v       = create_tensor(tn(LLM_TENSOR_YOCO_CROSS_V,       "weight"), {n_embd, kv_dim}, TENSOR_NOT_REQUIRED);
    yoco_cross_k_norm  = create_tensor(tn(LLM_TENSOR_YOCO_CROSS_K_NORM,  "weight"), {kv_dim}, TENSOR_NOT_REQUIRED);
}

std::unique_ptr<llm_graph_context> llama_model_bitnet::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_bitnet::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int64_t n_expert_count = hparams.n_expert;
    const int64_t n_expert_top_k = hparams.n_expert_used;
    const bool    has_moe        = (n_expert_count > 0);

    // Determine YOCO self/cross boundary by checking for K proj
    int n_self_layers = n_layer;
    for (int il = 0; il < n_layer; ++il) {
        if (model.layers[il].wk == nullptr) {
            n_self_layers = il;
            break;
        }
    }
    const bool is_yoco = (n_self_layers < n_layer);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // Embedding projection in
    if (model.emb_in_norm) {
        inpL = build_norm(inpL, model.emb_in_norm, NULL, LLM_NORM_RMS, -1);
    }
    if (model.emb_proj_in) {
        inpL = ggml_mul_mat(ctx0, model.emb_proj_in, inpL);
    }

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    auto * inp_attn_nc = is_yoco ? build_attn_inp_no_cache() : nullptr;
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * shared_K = nullptr;
    ggml_tensor * shared_V = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        const int64_t n_head_il    = hparams.n_head(il);
        const int64_t n_head_kv_il = hparams.n_head_kv(il);
        const bool is_cross_layer  = (il >= n_self_layers);

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Save attn_norm output for diff_v3 gate (cur will be overwritten by build_attn)
        ggml_tensor * cur_attn_norm = cur;

        // --- Attention ---
        {
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);

            if (model.layers[il].attn_q_norm) {
                Qcur = ggml_mul(ctx0, Qcur, model.layers[il].attn_q_norm);
            }
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head_il, n_tokens);

            if (!is_cross_layer) {
                // Self-decoder
                ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);

                if (model.layers[il].attn_k_norm) {
                    Kcur = ggml_mul(ctx0, Kcur, model.layers[il].attn_k_norm);
                }
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv_il, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv_il, n_tokens);

                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                cb(Qcur, "Qcur", il); cb(Kcur, "Kcur", il); cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn, NULL, NULL, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                        1.0f/sqrtf(float(n_embd_head)), il);
            } else {
                // Cross-decoder with shared KV
                GGML_ASSERT(shared_K && shared_V);
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                cur = build_attn(inp_attn_nc, NULL, NULL, NULL,
                        Qcur, shared_K, shared_V, nullptr, nullptr, nullptr,
                        1.0f/sqrtf(float(n_embd_head)), il);
            }

            // diff_v3: gate + differential attention
            // PyTorch ref: output = output * sigmoid(gate(x)); output = output[:,0::2] - output[:,1::2]
            if (model.layers[il].wqkv_gate) {
                const int64_t real_heads = n_head_il / 2;

                // gate: sigmoid(W_gate @ attn_norm_input) -> [n_head_il, n_tokens]
                ggml_tensor * gate = ggml_mul_mat(ctx0, model.layers[il].wqkv_gate, cur_attn_norm);
                gate = ggml_sigmoid(ctx0, gate);
                gate = ggml_reshape_3d(ctx0, gate, 1, n_head_il, n_tokens);

                // cur from build_attn: [n_embd_head * n_head_il, n_tokens]
                ggml_tensor * attn_3d = ggml_reshape_3d(ctx0, cur, n_embd_head, n_head_il, n_tokens);
                attn_3d = ggml_mul(ctx0, attn_3d, gate);
                attn_3d = ggml_cont(ctx0, attn_3d);

                // Differential: reshape [head_dim, 2, real_heads, tokens] to split even/odd
                ggml_tensor * r4d = ggml_reshape_4d(ctx0, attn_3d, n_embd_head, 2, real_heads, n_tokens);
                ggml_tensor * even = ggml_view_4d(ctx0, r4d, n_embd_head, 1, real_heads, n_tokens,
                        r4d->nb[1], r4d->nb[2], r4d->nb[3], 0);
                even = ggml_cont(ctx0, even);
                ggml_tensor * odd = ggml_view_4d(ctx0, r4d, n_embd_head, 1, real_heads, n_tokens,
                        r4d->nb[1], r4d->nb[2], r4d->nb[3], r4d->nb[1]);
                odd = ggml_cont(ctx0, odd);

                cur = ggml_sub(ctx0, even, odd);
                cur = ggml_reshape_2d(ctx0, cur, real_heads * n_embd_head, n_tokens);
            }

            cb(cur, "attn_out_pre", il);
            cur = ggml_mul_mat(ctx0, model.layers[il].wo, cur);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // --- FFN ---
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (has_moe) {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    nullptr, nullptr,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert_count, n_expert_top_k,
                    LLM_FFN_SILU, false, 0.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr,
                    model.layers[il].ffn_gate_up_exps,
                    nullptr, nullptr, nullptr, nullptr);
            cb(moe_out, "ffn_moe_out", il);

            // Shared expert
            if (model.layers[il].ffn_gate_shexp) {
                ggml_tensor * ffn_shexp = build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                if (model.layers[il].ffn_gate_inp_shexp) {
                    ggml_tensor * sg = ggml_mul_mat(ctx0, model.layers[il].ffn_gate_inp_shexp, cur);
                    sg = ggml_sigmoid(ctx0, sg);
                    ffn_shexp = ggml_mul(ctx0, ffn_shexp, sg);
                }
                cur = ggml_add(ctx0, moe_out, ffn_shexp);
            } else {
                cur = moe_out;
            }
            cb(cur, "ffn_out", il);
        } else {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;

        // Compute shared KV after self-decoder
        if (is_yoco && il == n_self_layers - 1 && model.yoco_cross_kv_norm) {
            const int64_t n_head_kv_cross = hparams.n_head_kv(n_self_layers);

            ggml_tensor * normed = build_norm(inpL, model.yoco_cross_kv_norm, NULL, LLM_NORM_RMS, -1);

            shared_K = ggml_mul_mat(ctx0, model.yoco_cross_k, normed);
            if (model.yoco_cross_k_norm) {
                shared_K = ggml_mul(ctx0, shared_K, model.yoco_cross_k_norm);
            }
            shared_K = ggml_reshape_3d(ctx0, shared_K, n_embd_head, n_head_kv_cross, n_tokens);
            shared_K = ggml_rope_ext(ctx0, shared_K, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            shared_V = ggml_mul_mat(ctx0, model.yoco_cross_v, normed);
            shared_V = ggml_reshape_3d(ctx0, shared_V, n_embd_head, n_head_kv_cross, n_tokens);
        }
    }

    cur = inpL;

    // Embedding projection out
    if (model.emb_out_norm) {
        cur = build_norm(cur, model.emb_out_norm, NULL, LLM_NORM_RMS, -1);
    }
    if (model.emb_proj_out) {
        cur = ggml_mul_mat(ctx0, model.emb_proj_out, cur);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head (weight tying)
    cur = ggml_mul_mat(ctx0, model.tok_embd, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

void llama_model_bitnet_b158::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 24: type = LLM_TYPE_700M; break;
        case 26: type = LLM_TYPE_3B;   break;
        case 30: type = LLM_TYPE_2B;   break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}
