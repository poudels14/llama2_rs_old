use crate::llama::{Config, RunState, TransformerWeights};
use crate::math::{accum, matmul, rmsnorm, softmax};
// use crate::{Config, RunState, TransformerWeights};

pub(crate) fn transformer(
    token: usize,
    pos: usize,
    p: &Config,
    s: &mut RunState,
    w: &TransformerWeights,
) {
    // a few convenice variables
    let x = s.x.as_mut_slice();
    let dim = p.dim as usize;
    let hidden_dim = p.hidden_dim as usize;
    let head_size = (p.dim / p.n_heads) as usize;

    // copy the token embedding into x
    let content_row: &[f32] = &w.token_embedding_table[token * dim..];
    x[0..dim].clone_from_slice(&content_row[0..dim]);

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    let freq_cis_real_row: &[f32] = &w.freq_cis_real[pos * head_size / 2..];
    let freq_cis_imag_row: &[f32] = &w.freq_cis_imag[pos * head_size / 2..];

    // forward all the layers
    for l in 0..p.n_layers as usize {
        // attention rmsnorm
        rmsnorm(
            &mut s.xb,
            x.as_ptr(),
            &w.rms_att_weight[(l * dim) as usize..],
            dim,
        );

        // qkv matmuls for this position
        let l_dim_dim = (l * dim * dim) as usize;
        matmul(&mut s.q, &s.xb, &w.wq[l_dim_dim..], dim, dim);
        matmul(&mut s.k, &s.xb, &w.wk[l_dim_dim..], dim, dim);
        matmul(&mut s.v, &s.xb, &w.wv[l_dim_dim..], dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        let mut h = 0;
        while h < p.n_heads {
            // get the q and k vectors for this head
            let q = &mut s.q[(h * head_size as i32) as usize..];
            let k = &mut s.k[(h * head_size as i32) as usize..];
            // rotate q and k by the freq_cis_real and freq_cis_imag
            let mut i = 0;
            while i < head_size {
                let q0 = q[i];
                let q1 = q[i + 1];
                let k0 = k[i];
                let k1 = k[i + 1];
                let fcr = freq_cis_real_row[i / 2];
                let fci = freq_cis_imag_row[i / 2];
                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;

                i = i + 2;
            }
            h += 1;
        }

        // save key,value at this time step (pos) to our kv cache
        let loff: usize = l * p.seq_len as usize * dim; // kv cache layer offset for convenience
        let lopp_pos_dim = loff + pos * dim;
        let key_cache_row = &mut s.key_cache[lopp_pos_dim..];
        let value_cache_row = &mut s.value_cache[lopp_pos_dim..];
        key_cache_row[0..dim].clone_from_slice(&s.k);
        value_cache_row[0..dim].clone_from_slice(&s.v);

        // multihead attention. iterate over all heads
        let mut h = 0;
        while h < p.n_heads as usize {
            // get the query vector for this head
            let q = &s.q[h * head_size..];
            // iterate over all timesteps, including the current one
            for t in 0..pos + 1 {
                // get the key vector for this head and at this timestep
                let k = &s.key_cache[loff + t * dim + h * head_size..];
                // calculate the attention score as the dot product of q and k
                let mut score = 0.0;
                for i in 0..head_size {
                    score += q[i] * k[i];
                }
                score = score / (head_size as f32).sqrt();
                // save the score to the attention buffer
                s.att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(&mut s.att, pos + 1);

            // weighted sum of the values, store back into xb
            for i in 0..head_size {
                let mut val = 0.0;
                for t in 0..pos + 1 {
                    val += s.att[t] * s.value_cache[loff + t * dim + h * head_size + i];
                    // note bad locality
                }
                s.xb[h * head_size + i] = val;
            }
            h += 1;
        }

        // final matmul to get the output of the attention
        matmul(&mut s.xb2, &s.xb, &w.wo[l * dim * dim..], dim, dim);

        // residual connection back into x
        accum(x, &s.xb2, dim);

        // ffn rmsnorm
        rmsnorm(&mut s.xb, x.as_ptr(), &w.rms_ffn_weight[l * dim..], dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(
            &mut s.hb,
            &s.xb,
            &w.w1[l * dim * hidden_dim..],
            dim,
            hidden_dim,
        );
        matmul(
            &mut s.hb2,
            &s.xb,
            &w.w3[l * dim * hidden_dim..],
            dim,
            hidden_dim,
        );

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for i in 0..hidden_dim {
            s.hb[i] = s.hb[i] * (1.0 / (1.0 + (-s.hb[i]).exp()));
        }

        // elementwise multiply with w3(x)
        for i in 0..hidden_dim {
            s.hb[i] = s.hb[i] * s.hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(
            &mut s.xb,
            &s.hb,
            &w.w2[l * dim * hidden_dim..],
            hidden_dim,
            dim,
        );

        // residual connection
        accum(x, &s.xb, dim);
    }

    // final rmsnorm
    rmsnorm(x, x.as_ptr(), &w.rms_final_weight, dim);

    // classifier into logits
    matmul(
        &mut s.logits,
        x,
        &w.token_embedding_table,
        dim,
        p.vocab_size as usize,
    );
}
