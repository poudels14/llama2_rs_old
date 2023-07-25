use crunchy::unroll;
use std::ops::{AddAssign, MulAssign};
use wide::f32x8;

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
    // Note(sagar): since the loop is unrolled, assert for n here
    assert!(n % 32 == 0);

    // Note(sagar): best to set this to be same as rayon threadpool threads
    let threads = 2;
    let xout = xout.as_ptr() as usize;
    let x = x.as_ptr() as usize;
    let w = w.as_ptr() as usize;

    rayon::join(
        || unsafe {
            let st = 0 * d / threads;

            let xout = (xout) as *mut f32;
            let xout = xout.offset(st as isize);
            let xout: &mut [f32] = core::slice::from_raw_parts_mut(xout, d / threads);
            let x: &mut [f32] = core::slice::from_raw_parts_mut(x as *mut f32, n);
            let w: &mut [f32] = core::slice::from_raw_parts_mut(w as *mut f32, n * d);
            matmul_partial(xout, x, &w[st * n..], n, d / threads);
        },
        || unsafe {
            let st = 1 * d / threads;

            let xout = (xout) as *mut f32;
            let xout = xout.offset(st as isize);
            let xout: &mut [f32] = core::slice::from_raw_parts_mut(xout, d / threads);
            let x: &mut [f32] = core::slice::from_raw_parts_mut(x as *mut f32, n);
            let w: &mut [f32] = core::slice::from_raw_parts_mut(w as *mut f32, n * d);
            matmul_partial(xout, x, &w[st * n..], n, d / threads);
        },
    );
}

fn matmul_partial(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    xout[0..d].iter_mut().enumerate().for_each(|(i, xo)| {
        let w = &w[i * n..];
        let xs = &x[..n];
        let ys = &w[..n];

        let mut sum = f32x8::new([0., 0., 0., 0., 0., 0., 0., 0.]);
        unroll! {
            for l in 0..2 {
                for k in 0..n / (8 * 2) {
                    let start = (n / 2 * l) + (k * 8);
                    let mut xs1 = f32x8::from(&xs[start..start + 8]);
                    let ys1 = f32x8::from(&ys[start..start + 8]);
                    xs1.mul_assign(ys1);
                    sum.add_assign(xs1);
                }
            }
        }
        *xo = sum.reduce_add();
    });
}

pub fn accum(a: &mut [f32], b: &[f32], size: usize) {
    for i in 0..size {
        a[i] += b[i];
    }
}
