#include "models.h"

void llama_model_bitnet::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // YOCO-U: read self-decoder iteration count (T), default 1
    ml.get_key(LLM_KV_YOCO_U_ITERS, hparams.yoco_u_iters, false);

    // Read sliding window for YOCO self-decoder layers
    // YOCO implements window attention via attention mask, not cache-level SWA.
    // n_swa is stored as metadata but swa_type stays NONE (no cache truncation).
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);

    // MoE: read expert FFN dimension from custom key
    {
        uint32_t moe_ffn_dim = 0;
        if (ml.get_key("bitnet.moe_ffn_dim", moe_ffn_dim, false)) {
            hparams.n_ff_exp = moe_ffn_dim;
        }
    }

    // SwiGLU clamping: default to 10.0 for all layers (matching PyTorch swiglu_limit=10.0)
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP, hparams.swiglu_clamp_exp, hparams.n_layer(), false)) {
        hparams.swiglu_clamp_exp.fill(10.0f);
    }
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer(), false)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    switch (hparams.n_layer()) {
        case 20:
            if (hparams.yoco_u_iters > 1) {
                type = LLM_TYPE_30B_A6B;  // YOCO-U-MoE: 20 stored layers, T=3 loop
            } else {
                type = LLM_TYPE_30B_A3B;  // YOCO-MoE: 20 layers
            }
            break;
        case 26: type = LLM_TYPE_3B; break;
        case 28: type = LLM_TYPE_3B; break;
        case 30: type = LLM_TYPE_2B; break;
        case 40: type = LLM_TYPE_30B_A6B; break; // YOCO-U-MoE 40 layers (unrolled format)
        default: type = LLM_TYPE_UNKNOWN;
    }

    // YOCO-U compact mode: KV cache needs T * n_self_layers slots
    // Expand n_layer_all and head arrays to match
    if (hparams.yoco_u_iters > 1 && hparams.n_layer() == 20) {
        const uint32_t T = hparams.yoco_u_iters;
        const uint32_t n_stored = hparams.n_layer();  // 20
        const uint32_t n_self = n_stored / 2;          // 10
        const uint32_t n_cross = n_stored - n_self;    // 10
        const uint32_t n_effective = T * n_self + n_cross;  // 40

        // Expand head count arrays: repeat self-decoder T times, then cross
        auto orig_n_head = hparams.n_head_arr;
        auto orig_n_head_kv = hparams.n_head_kv_arr;
        hparams.n_head_arr.fill(0);
        hparams.n_head_kv_arr.fill(0);
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t i = 0; i < n_self; ++i) {
                hparams.n_head_arr[t * n_self + i] = orig_n_head[i];
                hparams.n_head_kv_arr[t * n_self + i] = orig_n_head_kv[i];
            }
        }
        for (uint32_t i = 0; i < n_cross; ++i) {
            hparams.n_head_arr[T * n_self + i] = orig_n_head[n_self + i];
            hparams.n_head_kv_arr[T * n_self + i] = orig_n_head_kv[n_self + i];
        }

        hparams.n_layer_all = n_effective;
    }
}

void llama_model_bitnet::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp      = hparams.n_ff_exp > 0 ? hparams.n_ff_exp : n_ff;
    const bool    has_moe       = (n_expert > 0);

    // YOCO-U: only load physical (stored) layers from GGUF.
    // With yoco_u_iters=T, n_layer_all was expanded to T*n_self + n_cross,
    // but the GGUF only contains n_stored = n_self + n_cross unique layers.
    const uint32_t T = hparams.yoco_u_iters;
    int n_stored_layers = n_layer;  // default: all layers are stored
    if (T > 1) {
        // n_layer (expanded) = T * n_self + n_cross
        // n_stored = n_self + n_cross = (n_layer + (T-1)*n_cross) / T ... solve:
        // Actually: n_self = (n_layer - n_cross) / T, n_stored = n_self + n_cross
        // From the expansion: n_cross = n_stored - n_self, n_layer = T*n_self + n_cross
        // => n_stored = n_layer / T + n_cross * (T-1) / T ... simpler: use layers.size()
        // layers was resized to n_layer_all (expanded), but GGUF has block_count tensors.
        // The original block_count = n_stored = n_layer_all / T + cross*(T-1)/T
        // Simplest: n_self = n_stored/2, n_cross = n_stored/2, n_layer = T*n_self + n_cross
        // => n_stored = (n_layer + n_cross*(T-1)) hmm... let's just compute directly:
        // n_layer = T * n_self + n_cross, n_stored = n_self + n_cross
        // => n_self = (n_layer - n_cross) / T, and n_stored = n_self + n_cross
        // We know n_self = n_cross (by design: half self, half cross)
        // => n_layer = T * n_self + n_self = (T+1) * n_self => n_self = n_layer / (T+1)
        // => n_stored = 2 * n_self = 2 * n_layer / (T+1)
        n_stored_layers = 2 * n_layer / (T + 1);
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    // Embedding projections (optional)
    emb_proj_in  = create_tensor(tn(LLM_TENSOR_EMB_PROJ_IN,  "weight"), {n_embd, n_embd}, TENSOR_NOT_REQUIRED);
    emb_proj_out = create_tensor(tn(LLM_TENSOR_EMB_PROJ_OUT, "weight"), {n_embd, n_embd}, TENSOR_NOT_REQUIRED);
    emb_in_norm  = create_tensor(tn(LLM_TENSOR_EMB_IN_NORM,  "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    emb_out_norm = create_tensor(tn(LLM_TENSOR_EMB_OUT_NORM, "weight"), {n_embd}, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_stored_layers; ++i) {
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

        // ADP8 activation quantization scale/bias
        layer.attn_q_act_scale = create_tensor(tn(LLM_TENSOR_ATTN_Q_ACT_SCALE, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.attn_q_act_bias  = create_tensor(tn(LLM_TENSOR_ATTN_Q_ACT_BIAS,  "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.attn_k_act_scale = create_tensor(tn(LLM_TENSOR_ATTN_K_ACT_SCALE, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.attn_k_act_bias  = create_tensor(tn(LLM_TENSOR_ATTN_K_ACT_BIAS,  "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.attn_v_act_scale = create_tensor(tn(LLM_TENSOR_ATTN_V_ACT_SCALE, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.attn_v_act_bias  = create_tensor(tn(LLM_TENSOR_ATTN_V_ACT_BIAS,  "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.attn_out_act_scale = create_tensor(tn(LLM_TENSOR_ATTN_OUT_ACT_SCALE, "weight", i), {n_head_i / 2 * n_embd_head_k}, TENSOR_NOT_REQUIRED);
        layer.attn_out_act_bias  = create_tensor(tn(LLM_TENSOR_ATTN_OUT_ACT_BIAS,  "weight", i), {n_head_i / 2 * n_embd_head_k}, TENSOR_NOT_REQUIRED);
        layer.attn_gate_act_scale = create_tensor(tn(LLM_TENSOR_ATTN_GATE_ACT_SCALE, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.attn_gate_act_bias  = create_tensor(tn(LLM_TENSOR_ATTN_GATE_ACT_BIAS,  "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);

        // FFN norm
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (has_moe) {
            // MoE FFN
            layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2 * n_ff_exp, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

            // ADP8 for MoE experts
            layer.ffn_gate_up_exps_act_scale = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS_ACT_SCALE, "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_gate_up_exps_act_bias  = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS_ACT_BIAS, "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps_act_scale    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS_ACT_SCALE, "weight", i), {n_ff_exp, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps_act_bias     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS_ACT_BIAS, "weight", i), {n_ff_exp, n_expert}, TENSOR_NOT_REQUIRED);

            // Shared expert
            layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp}, TENSOR_NOT_REQUIRED);
            layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp, n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), {n_embd, 1}, TENSOR_NOT_REQUIRED);

            // ADP8 for shared expert
            layer.ffn_shexp_gate_up_act_scale = create_tensor(tn(LLM_TENSOR_FFN_SHEXP_GATE_UP_ACT_SCALE, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_shexp_gate_up_act_bias  = create_tensor(tn(LLM_TENSOR_FFN_SHEXP_GATE_UP_ACT_BIAS, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_shexp_down_act_scale    = create_tensor(tn(LLM_TENSOR_FFN_SHEXP_DOWN_ACT_SCALE, "weight", i), {n_ff_exp}, TENSOR_NOT_REQUIRED);
            layer.ffn_shexp_down_act_bias     = create_tensor(tn(LLM_TENSOR_FFN_SHEXP_DOWN_ACT_BIAS, "weight", i), {n_ff_exp}, TENSOR_NOT_REQUIRED);
        } else {
            // Dense FFN
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);

            // ADP8 for dense FFN
            layer.ffn_gate_up_act_scale = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_ACT_SCALE, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_gate_up_act_bias  = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_ACT_BIAS,  "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_act_scale    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_ACT_SCALE,    "weight", i), {n_ff}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_act_bias     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_ACT_BIAS,     "weight", i), {n_ff}, TENSOR_NOT_REQUIRED);
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

    // YOCO-U: determine stored vs effective layer counts
    const int n_stored_layers = (int)model.layers.size();  // actual stored layers (20)
    const int yoco_u_iters = hparams.yoco_u_iters;
    const int n_self_stored = n_stored_layers / 2;   // 10 self layers stored
    const int n_cross_stored = n_stored_layers - n_self_stored;  // 10 cross layers stored

    // Determine YOCO self/cross boundary from stored layers
    int n_self_layers_stored = n_stored_layers;
    for (int il = 0; il < n_stored_layers; ++il) {
        if (model.layers[il].wk == nullptr) {
            n_self_layers_stored = il;
            break;
        }
    }
    const bool is_yoco = (n_self_layers_stored < n_stored_layers);

    // Effective layer count for inference (with T iterations)
    const int n_effective = yoco_u_iters * n_self_layers_stored + (n_stored_layers - n_self_layers_stored);

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
        // Map effective layer index to stored layer index
        int il_stored;
        bool is_cross_layer;
        if (il < yoco_u_iters * n_self_layers_stored) {
            // Self-decoder pass: il_stored cycles through 0..n_self_stored-1
            il_stored = il % n_self_layers_stored;
            is_cross_layer = false;
        } else {
            // Cross-decoder: offset into cross layers
            il_stored = n_self_layers_stored + (il - yoco_u_iters * n_self_layers_stored);
            is_cross_layer = true;
        }

        ggml_tensor * inpSA = inpL;

        const int64_t n_head_il    = hparams.n_head(il);
        const int64_t n_head_kv_il = hparams.n_head_kv(il);

        cur = build_norm(inpL, model.layers[il_stored].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Save attn_norm output for diff_v3 gate (cur will be overwritten by build_attn)
        ggml_tensor * cur_attn_norm = cur;

        // --- Attention ---
        {
            // ADP8 activation quantization: x_q = x * act_scale + act_bias
            ggml_tensor * cur_q = cur;
            if (model.layers[il_stored].attn_q_act_scale) {
                cur_q = ggml_mul(ctx0, cur, model.layers[il_stored].attn_q_act_scale);
                if (model.layers[il_stored].attn_q_act_bias) {
                    cur_q = ggml_add(ctx0, cur_q, model.layers[il_stored].attn_q_act_bias);
                }
            }
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il_stored].wq, cur_q);

            // Reshape Q to 3D [head_dim, n_head, n_tokens] BEFORE RMS-clip
            // so that RMS-clip operates per-head (matching PyTorch which does
            // mean(dim=-1) over head_dim after view(n_tokens, n_head, head_dim))
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head_il, n_tokens);

            if (model.layers[il_stored].attn_q_norm) {
                // RMS-clip per head: result = x * min(limit/rms, 1.0) * weight
                // Uses ggml_rms_norm (single fused kernel for per-head RMS normalization)
                // then conditional selection: take scaled when rms > limit, else take x
                const float qk_rms_limit = 3.0f;
                const float eps = hparams.f_norm_rms_eps;

                ggml_tensor * normed = ggml_rms_norm(ctx0, Qcur, eps);            // x/rms
                ggml_tensor * scaled = ggml_scale(ctx0, normed, qk_rms_limit);    // x*limit/rms
                ggml_tensor * diff   = ggml_sub(ctx0, Qcur, scaled);              // x*(1-limit/rms)
                ggml_tensor * mask   = ggml_step(ctx0, ggml_mul(ctx0, Qcur, diff)); // 1 when rms>limit
                Qcur = ggml_sub(ctx0, Qcur, ggml_mul(ctx0, diff, mask));          // x or scaled
                // apply per-element weight
                ggml_tensor * q_norm_3d = ggml_reshape_3d(ctx0, model.layers[il_stored].attn_q_norm, n_embd_head, n_head_il, 1);
                Qcur = ggml_mul(ctx0, Qcur, q_norm_3d);
            }

            if (!is_cross_layer) {
                // Self-decoder
                ggml_tensor * cur_k = cur;
                if (model.layers[il_stored].attn_k_act_scale) {
                    cur_k = ggml_mul(ctx0, cur, model.layers[il_stored].attn_k_act_scale);
                    if (model.layers[il_stored].attn_k_act_bias) {
                        cur_k = ggml_add(ctx0, cur_k, model.layers[il_stored].attn_k_act_bias);
                    }
                }
                ggml_tensor * cur_v = cur;
                if (model.layers[il_stored].attn_v_act_scale) {
                    cur_v = ggml_mul(ctx0, cur, model.layers[il_stored].attn_v_act_scale);
                    if (model.layers[il_stored].attn_v_act_bias) {
                        cur_v = ggml_add(ctx0, cur_v, model.layers[il_stored].attn_v_act_bias);
                    }
                }
                ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il_stored].wk, cur_k);
                ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il_stored].wv, cur_v);

                // Reshape K to 3D before RMS-clip (per-head normalization)
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv_il, n_tokens);

                if (model.layers[il_stored].attn_k_norm) {
                    const float qk_rms_limit = 3.0f;
                    const float eps = hparams.f_norm_rms_eps;

                    ggml_tensor * k_normed = ggml_rms_norm(ctx0, Kcur, eps);
                    ggml_tensor * k_scaled = ggml_scale(ctx0, k_normed, qk_rms_limit);
                    ggml_tensor * k_diff   = ggml_sub(ctx0, Kcur, k_scaled);
                    ggml_tensor * k_mask   = ggml_step(ctx0, ggml_mul(ctx0, Kcur, k_diff));
                    Kcur = ggml_sub(ctx0, Kcur, ggml_mul(ctx0, k_diff, k_mask));
                    ggml_tensor * k_norm_3d = ggml_reshape_3d(ctx0, model.layers[il_stored].attn_k_norm, n_embd_head, n_head_kv_il, 1);
                    Kcur = ggml_mul(ctx0, Kcur, k_norm_3d);
                }
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
                // Cross-decoder: no RoPE on Q (shared K already has RoPE applied)
                cb(Qcur, "Qcur", il);

                cur = build_attn(inp_attn_nc, NULL, NULL, NULL,
                        Qcur, shared_K, shared_V, nullptr, nullptr, nullptr,
                        1.0f/sqrtf(float(n_embd_head)), il);
            }

            // diff_v3: gate + differential attention
            // PyTorch ref: output = output * sigmoid(gate(x)); output = output[:,0::2] - output[:,1::2]
            if (model.layers[il_stored].wqkv_gate) {
                const int64_t real_heads = n_head_il / 2;

                // gate: sigmoid(W_gate @ attn_norm_input) -> [n_head_il, n_tokens]
                ggml_tensor * gate_input = cur_attn_norm;
                if (model.layers[il_stored].attn_gate_act_scale) {
                    gate_input = ggml_mul(ctx0, cur_attn_norm, model.layers[il_stored].attn_gate_act_scale);
                    if (model.layers[il_stored].attn_gate_act_bias) {
                        gate_input = ggml_add(ctx0, gate_input, model.layers[il_stored].attn_gate_act_bias);
                    }
                }
                ggml_tensor * gate = ggml_mul_mat(ctx0, model.layers[il_stored].wqkv_gate, gate_input);
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
            // ADP8 for o_proj
            if (model.layers[il_stored].attn_out_act_scale) {
                cur = ggml_mul(ctx0, cur, model.layers[il_stored].attn_out_act_scale);
                if (model.layers[il_stored].attn_out_act_bias) {
                    cur = ggml_add(ctx0, cur, model.layers[il_stored].attn_out_act_bias);
                }
            }
            cur = ggml_mul_mat(ctx0, model.layers[il_stored].wo, cur);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // --- FFN ---
        cur = build_norm(ffn_inp, model.layers[il_stored].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (has_moe) {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il_stored].ffn_gate_inp,
                    nullptr,  // gate_inp_b
                    nullptr,  // up_exps
                    nullptr,  // up_exps_b
                    nullptr,  // gate_exps
                    nullptr,  // gate_exps_b
                    model.layers[il_stored].ffn_down_exps,
                    nullptr,  // down_exps_b
                    nullptr,  // exp_probs_b
                    n_expert_count, n_expert_top_k,
                    LLM_FFN_SILU, true, 0.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr,  // probs_in
                    model.layers[il_stored].ffn_gate_up_exps,
                    nullptr,  // gate_up_exps_b
                    nullptr,  // up_exps_s
                    nullptr,  // gate_exps_s
                    nullptr,  // down_exps_s
                    nullptr,  // selected_experts_in
                    model.layers[il_stored].ffn_gate_up_exps_act_scale,
                    model.layers[il_stored].ffn_gate_up_exps_act_bias,
                    model.layers[il_stored].ffn_down_exps_act_scale,
                    model.layers[il_stored].ffn_down_exps_act_bias);
            cb(moe_out, "ffn_moe_out", il);

            // Shared expert — manual SwiGLU with correct clamp-before-silu order
            // PyTorch: silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
            if (model.layers[il_stored].ffn_gate_shexp) {
                const float shexp_limit = hparams.swiglu_clamp_shexp[il];

                // ADP8 for shared expert gate_up input
                ggml_tensor * shexp_input = cur;
                if (model.layers[il_stored].ffn_shexp_gate_up_act_scale) {
                    shexp_input = ggml_mul(ctx0, cur, model.layers[il_stored].ffn_shexp_gate_up_act_scale);
                    if (model.layers[il_stored].ffn_shexp_gate_up_act_bias) {
                        shexp_input = ggml_add(ctx0, shexp_input, model.layers[il_stored].ffn_shexp_gate_up_act_bias);
                    }
                }

                ggml_tensor * up_shexp   = ggml_mul_mat(ctx0, model.layers[il_stored].ffn_up_shexp,   shexp_input);
                ggml_tensor * gate_shexp = ggml_mul_mat(ctx0, model.layers[il_stored].ffn_gate_shexp, shexp_input);

                ggml_tensor * ffn_shexp;
                if (shexp_limit > 1e-6f) {
                    up_shexp   = ggml_clamp(ctx0, up_shexp,   -shexp_limit, shexp_limit);
                    gate_shexp = ggml_clamp(ctx0, gate_shexp, -INFINITY,    shexp_limit);
                    gate_shexp = ggml_silu(ctx0, gate_shexp);
                    ffn_shexp  = ggml_mul(ctx0, gate_shexp, up_shexp);
                } else {
                    ffn_shexp = ggml_swiglu_split(ctx0, gate_shexp, up_shexp);
                }

                // ADP8 for shared expert down input
                if (model.layers[il_stored].ffn_shexp_down_act_scale) {
                    ffn_shexp = ggml_mul(ctx0, ffn_shexp, model.layers[il_stored].ffn_shexp_down_act_scale);
                    if (model.layers[il_stored].ffn_shexp_down_act_bias) {
                        ffn_shexp = ggml_add(ctx0, ffn_shexp, model.layers[il_stored].ffn_shexp_down_act_bias);
                    }
                }

                ffn_shexp = ggml_mul_mat(ctx0, model.layers[il_stored].ffn_down_shexp, ffn_shexp);
                cb(ffn_shexp, "ffn_shexp", il);

                if (model.layers[il_stored].ffn_gate_inp_shexp) {
                    ggml_tensor * sg = ggml_mul_mat(ctx0, model.layers[il_stored].ffn_gate_inp_shexp, cur);
                    sg = ggml_sigmoid(ctx0, sg);
                    ffn_shexp = ggml_mul(ctx0, ffn_shexp, sg);
                }
                cur = ggml_add(ctx0, moe_out, ffn_shexp);
            } else {
                cur = moe_out;
            }
            cb(cur, "ffn_out", il);
        } else {
            // Dense FFN — manual SwiGLU with ADP8 activation quantization
            // (mirrors shared-expert path: clamp-before-silu + x*scale+bias hooks)
            const float ffn_limit = hparams.swiglu_clamp_exp[il];

            // ADP8 for gate/up input
            ggml_tensor * ffn_input = cur;
            if (model.layers[il_stored].ffn_gate_up_act_scale) {
                ffn_input = ggml_mul(ctx0, cur, model.layers[il_stored].ffn_gate_up_act_scale);
                if (model.layers[il_stored].ffn_gate_up_act_bias) {
                    ffn_input = ggml_add(ctx0, ffn_input, model.layers[il_stored].ffn_gate_up_act_bias);
                }
            }

            ggml_tensor * up_ffn   = ggml_mul_mat(ctx0, model.layers[il_stored].ffn_up,   ffn_input);
            ggml_tensor * gate_ffn = ggml_mul_mat(ctx0, model.layers[il_stored].ffn_gate, ffn_input);

            if (ffn_limit > 1e-6f) {
                up_ffn   = ggml_clamp(ctx0, up_ffn,   -ffn_limit, ffn_limit);
                gate_ffn = ggml_clamp(ctx0, gate_ffn, -INFINITY,  ffn_limit);
                gate_ffn = ggml_silu(ctx0, gate_ffn);
                cur = ggml_mul(ctx0, gate_ffn, up_ffn);
            } else {
                cur = ggml_swiglu_split(ctx0, gate_ffn, up_ffn);
            }

            // ADP8 for down input
            if (model.layers[il_stored].ffn_down_act_scale) {
                cur = ggml_mul(ctx0, cur, model.layers[il_stored].ffn_down_act_scale);
                if (model.layers[il_stored].ffn_down_act_bias) {
                    cur = ggml_add(ctx0, cur, model.layers[il_stored].ffn_down_act_bias);
                }
            }

            cur = ggml_mul_mat(ctx0, model.layers[il_stored].ffn_down, cur);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;

        // Compute shared KV after self-decoder
        if (is_yoco && il == yoco_u_iters * n_self_layers_stored - 1 && model.yoco_cross_kv_norm) {
            const int64_t n_head_kv_cross = hparams.n_head_kv(yoco_u_iters * n_self_layers_stored);

            ggml_tensor * normed = build_norm(inpL, model.yoco_cross_kv_norm, NULL, LLM_NORM_RMS, -1);

            shared_K = ggml_mul_mat(ctx0, model.yoco_cross_k, normed);
            // Reshape to 3D before RMS-clip (per-head normalization)
            shared_K = ggml_reshape_3d(ctx0, shared_K, n_embd_head, n_head_kv_cross, n_tokens);
            if (model.yoco_cross_k_norm) {
                const float qk_rms_limit = 3.0f;
                const float eps = hparams.f_norm_rms_eps;

                ggml_tensor * ck_normed = ggml_rms_norm(ctx0, shared_K, eps);
                ggml_tensor * ck_scaled = ggml_scale(ctx0, ck_normed, qk_rms_limit);
                ggml_tensor * ck_diff   = ggml_sub(ctx0, shared_K, ck_scaled);
                ggml_tensor * ck_mask   = ggml_step(ctx0, ggml_mul(ctx0, shared_K, ck_diff));
                shared_K = ggml_sub(ctx0, shared_K, ggml_mul(ctx0, ck_diff, ck_mask));
                ggml_tensor * ck_norm_3d = ggml_reshape_3d(ctx0, model.yoco_cross_k_norm, n_embd_head, n_head_kv_cross, 1);
                shared_K = ggml_mul(ctx0, shared_K, ck_norm_3d);
            }
            // shared_K is already 3D [head_dim, n_head_kv, n_tokens]
            shared_K = ggml_rope_ext(ctx0, shared_K, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            shared_V = ggml_mul_mat(ctx0, model.yoco_cross_v, normed);
            shared_V = ggml_reshape_3d(ctx0, shared_V, n_embd_head, n_head_kv_cross, n_tokens);
        }
    }

    cur = inpL;

    // Embedding projection out: output_norm → emb_proj_out → emb_out_norm
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    if (model.emb_proj_out) {
        cur = ggml_mul_mat(ctx0, model.emb_proj_out, cur);
    }
    if (model.emb_out_norm) {
        cur = build_norm(cur, model.emb_out_norm, NULL, LLM_NORM_RMS, -1);
    }
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
