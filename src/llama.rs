use crate::math;
use crate::reader::FloatReader;
use crate::transformer;
use anyhow::Result;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub dim: i32,        // transformer dimension
    pub hidden_dim: i32, // for ffn layers
    pub n_layers: i32,   // number of layers
    pub n_heads: i32,    // number of query heads
    #[allow(dead_code)]
    pub n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    pub seq_len: i32,    // max sequence length
}

pub struct TransformerWeights {
    // token embedding table
    // (vocab_size, dim)
    pub token_embedding_table: Vec<f32>,

    // weights for rmsnorms
    pub rms_att_weight: Vec<f32>, // (layer, dim) rmsnorm weights
    pub rms_ffn_weight: Vec<f32>, // (layer, dim)
    // weights for matmuls
    pub wq: Vec<f32>, // (layer, dim, dim)
    pub wk: Vec<f32>, // (layer, dim, dim)
    pub wv: Vec<f32>, // (layer, dim, dim)
    pub wo: Vec<f32>, // (layer, dim, dim)
    // weights for ffn
    pub w1: Vec<f32>, // (layer, hidden_dim, dim)
    pub w2: Vec<f32>, // (layer, dim, hidden_dim)
    pub w3: Vec<f32>, // (layer, hidden_dim, dim)
    // final rmsnorm
    pub rms_final_weight: Vec<f32>, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    pub freq_cis_real: Vec<f32>, // (seq_len, dim/2)
    pub freq_cis_imag: Vec<f32>, // (seq_len, dim/2)
}

pub struct RunState {
    // current wave of activations
    pub x: Vec<f32>,      // activation at current time stamp (dim,)
    pub xb: Vec<f32>,     // same, but inside a residual branch (dim,)
    pub xb2: Vec<f32>,    // an additional buffer just for convenience (dim,)
    pub hb: Vec<f32>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb2: Vec<f32>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    pub q: Vec<f32>,      // query (dim,)
    pub k: Vec<f32>,      // key (dim,)
    pub v: Vec<f32>,      // value (dim,)
    pub att: Vec<f32>,    // buffer for scores/attention values (seq_len,)
    pub logits: Vec<f32>, // output logits
    // kv cache
    pub key_cache: Vec<f32>,   // (layer, seq_len, dim)
    pub value_cache: Vec<f32>, // (layer, seq_len, dim)
}

pub struct RunOptions {
    pub temperature: f32,
}

// TODO(poudels14): change this to iterator
pub fn run(
    config: &Config,
    state: &mut RunState,
    weights: &TransformerWeights,
    options: RunOptions,
) {
    let mut next;
    let mut token = 1; // 1 = BOS token in Llama-2 sentencepiece
    let mut pos = 0;
    while pos < config.seq_len {
        // forward the transformer to get logits for the next token
        transformer::transformer(token, pos as usize, config, state, weights);

        // sample the next token
        if options.temperature == 0.0 {
            // greedy argmax sampling
            next = math::argmax(&state.logits, config.vocab_size as usize);
        } else {
            // apply the temperature to the logits
            for q in 0..config.vocab_size as usize {
                state.logits[q] /= options.temperature;
            }
            // apply softmax to the logits to get the probabilities for next token
            math::softmax(&mut state.logits, config.vocab_size as usize);
            // we now want to sample from this distribution to get the next token
            next = math::sample(&state.logits, config.vocab_size as usize);
        }
        println!("{:?}", next);

        // advance forward
        token = next;
        pos += 1;
    }
}

pub fn init_checkpoint_weights(
    mut reader: BufReader<File>,
    config: &Config,
) -> Result<TransformerWeights> {
    let mut r = FloatReader::new(&mut reader);
    let head_size = config.dim / config.n_heads;
    let weights = TransformerWeights {
        token_embedding_table: r.read_vec(config.vocab_size * config.dim)?,
        rms_att_weight: r.read_vec(config.n_layers * config.dim)?,
        wq: r.read_vec(config.n_layers * config.dim * config.dim)?,
        wk: r.read_vec(config.n_layers * config.dim * config.dim)?,
        wv: r.read_vec(config.n_layers * config.dim * config.dim)?,
        wo: r.read_vec(config.n_layers * config.dim * config.dim)?,
        rms_ffn_weight: r.read_vec(config.n_layers * config.dim)?,
        w1: r.read_vec(config.n_layers * config.dim * config.hidden_dim)?,
        w2: r.read_vec(config.n_layers * config.dim * config.hidden_dim)?,
        w3: r.read_vec(config.n_layers * config.dim * config.hidden_dim)?,
        rms_final_weight: r.read_vec(config.dim)?,
        freq_cis_real: r.read_vec(config.seq_len * head_size / 2)?,
        freq_cis_imag: r.read_vec(config.seq_len * head_size / 2)?,
    };

    Ok(weights)
}

pub fn init_run_state(config: &Config) -> RunState {
    let dim = config.dim as usize;
    let hidden_dim = config.hidden_dim as usize;
    let cache = (config.n_layers * config.seq_len * config.dim) as usize;
    RunState {
        x: vec![0.; dim],
        xb: vec![0.; dim],
        xb2: vec![0.; dim],
        hb: vec![0.; hidden_dim],
        hb2: vec![0.; hidden_dim],
        q: vec![0.; dim],
        k: vec![0.; dim],
        v: vec![0.; dim],
        att: vec![0.; config.seq_len as usize],
        logits: vec![0.; config.vocab_size as usize],
        key_cache: vec![0.; cache],
        value_cache: vec![0.; cache],
    }
}
