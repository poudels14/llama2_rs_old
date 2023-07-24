use rayon::prelude::*;

pub fn argmax(v: &[f32], n: usize) -> usize {
    // return argmax of v in elements 0..n
    let mut max_i = 0;
    let mut max_p = v[0];
    for i in 1..n {
        if v[i] > max_p {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}

pub fn sample(probabilities: &[f32], n: usize) -> usize {
    // sample index from probabilities, they must sum to 1
    let r = rand::random::<f32>();
    let mut cdf = 0.0;
    for i in 0..n {
        cdf += probabilities[i];
        if r < cdf {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

pub fn rmsnorm(o: &mut [f32], x_ptr: *const f32, weight: &[f32], size: usize) {
    let x: &[f32] = unsafe { core::slice::from_raw_parts(x_ptr, o.len()) };
    // calculate sum of squares
    let mut ss = 0.;
    for j in 0..size {
        ss += x[j] * x[j];
    }

    ss = ss / size as f32;
    ss = ss + 1e-5;
    ss = 1.0 / ss.sqrt();
    // normalize and scale
    for j in 0..size {
        o[j] = weight[j] * (ss * x[j]);
    }
}

pub fn softmax(x: &mut [f32], size: usize) {
    if size == 1 {
        x[0] = 1.0;
        return;
    }

    // find max value (for numerical stability)
    let mut max_val = x[0];
    for i in 1..size {
        if x[i] > max_val {
            max_val = x[i];
        }
    }
    // e^x
    for i in 0..size {
        x[i] = (x[i] - max_val).exp();
    }
    // normalize
    let mut sum = 0.0;
    for i in 0..size {
        sum = sum + x[i];
    }
    for i in 0..size {
        x[i] /= sum;
    }
}

pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    // W (d,n) @ x (n,) -> xout (d,)
    xout[0..d].par_iter_mut().enumerate().for_each(|(i, xo)| {
        let mut val: f32 = 0.0;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        *xo = val;
    });
}

pub fn accum(a: &mut [f32], b: &[f32], size: usize) {
    for i in 0..size {
        a[i] += b[i];
    }
}
