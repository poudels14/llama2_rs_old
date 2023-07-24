use crate::math;
use crate::reader::ModelLoader;
use crate::transformer;
use anyhow::Result;
use nalgebra::DVector;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub dim: usize,        // transformer dimension
    pub hidden_dim: usize, // for ffn layers
    pub n_layers: usize,   // number of layers
    pub n_heads: usize,    // number of query heads
    #[allow(dead_code)]
    pub n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    pub seq_len: usize,    // max sequence length
}

pub struct TransformerWeights {
    // token embedding table
    pub token_embedding_table: Vec<DVector<f32>>, // (vocab_size, dim)
    // weights for rmsnorms
    pub rms_att_weight: Vec<DVector<f32>>, // (layer, dim) rmsnorm weights
    pub rms_ffn_weight: Vec<DVector<f32>>, // (layer, dim)
    // weights for matmuls
    pub wq: Vec<Vec<DVector<f32>>>, // (layer, dim, dim)
    pub wk: Vec<Vec<DVector<f32>>>, // (layer, dim, dim)
    pub wv: Vec<Vec<DVector<f32>>>, // (layer, dim, dim)
    pub wo: Vec<Vec<DVector<f32>>>, // (layer, dim, dim)
    // weights for ffn
    pub w1: Vec<Vec<DVector<f32>>>, // (layer, hidden_dim, dim)
    pub w2: Vec<Vec<DVector<f32>>>, // (layer, dim, hidden_dim)
    pub w3: Vec<Vec<DVector<f32>>>, // (layer, hidden_dim, dim)
    // final rmsnorm
    pub rms_final_weight: DVector<f32>, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    pub freq_cis_real: Vec<f32>, // (seq_len, dim/2)
    pub freq_cis_imag: Vec<f32>, // (seq_len, dim/2)
}

pub struct RunState {
    // current wave of activations
    pub x: DVector<f32>,      // activation at current time stamp (dim,)
    pub xb: DVector<f32>,     // same, but inside a residual branch (dim,)
    pub xb2: DVector<f32>,    // an additional buffer just for convenience (dim,)
    pub hb: DVector<f32>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb2: DVector<f32>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    pub q: DVector<f32>,      // query (dim,)
    pub k: DVector<f32>,      // key (dim,)
    pub v: DVector<f32>,      // value (dim,)
    pub att: DVector<f32>,    // buffer for scores/attention values (seq_len,)
    pub logits: DVector<f32>, // output logits
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
            next = math::argmax(&state.logits.as_slice());
        } else {
            // apply the temperature to the logits
            for q in 0..config.vocab_size as usize {
                state.logits[q] /= options.temperature;
            }
            // apply softmax to the logits to get the probabilities for next token
            math::softmax(state.logits.as_mut_slice(), config.vocab_size as usize);
            // we now want to sample from this distribution to get the next token
            next = math::sample(&state.logits.as_slice(), config.vocab_size as usize);
        }
        println!("{:?}", next);

        // advance forward
        token = next;
        pos += 1;
    }
}

pub fn init_checkpoint_weights(
    mut r: ModelLoader<'_>,
    config: &Config,
) -> Result<TransformerWeights> {
    let head_size = config.dim / config.n_heads;
    let weights = TransformerWeights {
        token_embedding_table: r.read_matrix(config.vocab_size, config.dim)?,
        rms_att_weight: r.read_matrix(config.n_layers, config.dim)?,
        wq: (0..config.n_layers)
            .map(|_| r.read_matrix(config.dim, config.dim))
            .collect::<Result<Vec<Vec<DVector<f32>>>>>()?,
        wk: (0..config.n_layers)
            .map(|_| r.read_matrix(config.dim, config.dim))
            .collect::<Result<Vec<Vec<DVector<f32>>>>>()?,
        wv: (0..config.n_layers)
            .map(|_| r.read_matrix(config.dim, config.dim))
            .collect::<Result<Vec<Vec<DVector<f32>>>>>()?,
        wo: (0..config.n_layers)
            .map(|_| r.read_matrix(config.dim, config.dim))
            .collect::<Result<Vec<Vec<DVector<f32>>>>>()?,
        rms_ffn_weight: r.read_matrix(config.n_layers, config.dim)?,
        w1: (0..config.n_layers)
            .map(|_| r.read_matrix(config.hidden_dim, config.dim))
            .collect::<Result<Vec<Vec<DVector<f32>>>>>()?,
        w2: (0..config.n_layers)
            .map(|_| r.read_matrix(config.dim, config.hidden_dim))
            .collect::<Result<Vec<Vec<DVector<f32>>>>>()?,
        w3: (0..config.n_layers)
            .map(|_| r.read_matrix(config.hidden_dim, config.dim))
            .collect::<Result<Vec<Vec<DVector<f32>>>>>()?,
        rms_final_weight: DVector::from(r.read_vec(config.dim)?),
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
        x: DVector::zeros(dim),
        xb: DVector::zeros(dim),
        xb2: DVector::zeros(dim),
        hb: DVector::zeros(hidden_dim),
        hb2: DVector::zeros(hidden_dim),
        q: DVector::zeros(dim),
        k: DVector::zeros(dim),
        v: DVector::zeros(dim),
        att: DVector::zeros(config.seq_len as usize),
        logits: DVector::zeros(config.vocab_size as usize),
        key_cache: vec![0.; cache],
        value_cache: vec![0.; cache],
    }
}
