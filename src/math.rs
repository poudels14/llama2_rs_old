use nalgebra::DVector;
use rayon::prelude::*;
use std::ops::AddAssign;

pub fn argmax(v: &[f32]) -> usize {
    // return argmax of v in elements 0..n
    let mut max_i = 0;
    let mut max_p = v[0];
    for i in 1..v.len() {
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

pub fn rmsnorm(o: &mut DVector<f32>, x: &DVector<f32>, weight: &DVector<f32>, size: usize) {
    // calculate sum of squares
    let mut ss = x.dot(&x);

    ss = ss / size as f32;
    ss = ss + 1e-5;
    ss = 1.0 / ss.sqrt();

    o.copy_from(&weight.component_mul(&x.scale(ss)));
}

pub fn rmsnorm_self(o: &mut DVector<f32>, weight: &DVector<f32>, size: usize) {
    // calculate sum of squares

    let mut ss = o.dot(&o);

    ss = ss / size as f32;
    ss = ss + 1e-5;
    ss = 1.0 / ss.sqrt();

    o.copy_from(&weight.component_mul(&o.scale(ss)));
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

pub fn matmul(xout: &mut DVector<f32>, x: &DVector<f32>, w: &Vec<DVector<f32>>, d: usize) {
    let xout = xout.as_mut_ptr() as usize;
    w.into_par_iter().enumerate().for_each(|(i, w)| {
        let xout = unsafe { core::slice::from_raw_parts_mut(xout as *mut f32, d) };
        xout[i] = x.dot(w);
    });
}

pub fn accum(a: &mut DVector<f32>, b: &DVector<f32>) {
    a.add_assign(b);
}
