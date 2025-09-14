use ndarray::Array2;
use rustfft::{FftPlanner, num_complex::Complex};

/// Compute convolution via FFT for a single-channel 2D image and kernel.
/// Inputs are assumed to be small enough to allocate complex arrays of size MxN.
pub fn conv2d_fft(input: &Array2<f32>, kernel: &Array2<f32>) -> Array2<f32> {
    let (h, w) = (input.dim().0, input.dim().1);
    let (kh, kw) = (kernel.dim().0, kernel.dim().1);
    let out_h = h + kh - 1;
    let out_w = w + kw - 1;
    let mut planner = FftPlanner::<f32>::new();
    let n_fft_h = out_h.next_power_of_two();
    let n_fft_w = out_w.next_power_of_two();
    let len = n_fft_h * n_fft_w;

    // pack into complex arrays
    let mut in_freq: Vec<Complex<f32>> = vec![Complex { re: 0.0, im: 0.0 }; len];
    let mut ker_freq: Vec<Complex<f32>> = vec![Complex { re: 0.0, im: 0.0 }; len];

    for y in 0..h {
        for x in 0..w {
            let idx = y * n_fft_w + x;
            in_freq[idx].re = input[[y, x]];
        }
    }
    for y in 0..kh {
        for x in 0..kw {
            let idx = y * n_fft_w + x;
            ker_freq[idx].re = kernel[[y, x]];
        }
    }

    // 1D FFT plan for rows then cols (simple approach: use 1D transforms on flattened)
    let fft_h = planner.plan_fft_forward(n_fft_h);
    let fft_w = planner.plan_fft_forward(n_fft_w);
    let ifft_h = planner.plan_fft_inverse(n_fft_h);
    let ifft_w = planner.plan_fft_inverse(n_fft_w);

    // Row-wise FFT
    for r in 0..n_fft_h {
        let start = r * n_fft_w;
        fft_w.process(&mut in_freq[start..start + n_fft_w]);
        fft_w.process(&mut ker_freq[start..start + n_fft_w]);
    }
    // Column-wise FFT
    // transpose-like access
    let mut col_in = vec![Complex { re: 0.0, im: 0.0 }; n_fft_h];
    let mut col_ker = vec![Complex { re: 0.0, im: 0.0 }; n_fft_h];
    for c in 0..n_fft_w {
        for r in 0..n_fft_h {
            col_in[r] = in_freq[r * n_fft_w + c];
            col_ker[r] = ker_freq[r * n_fft_w + c];
        }
        fft_h.process(&mut col_in);
        fft_h.process(&mut col_ker);
        for r in 0..n_fft_h {
            in_freq[r * n_fft_w + c] = col_in[r];
            ker_freq[r * n_fft_w + c] = col_ker[r];
        }
    }

    // pointwise multiply
    for i in 0..len {
        in_freq[i] = in_freq[i] * ker_freq[i];
    }

    // inverse column
    for c in 0..n_fft_w {
        for r in 0..n_fft_h {
            col_in[r] = in_freq[r * n_fft_w + c];
        }
        ifft_h.process(&mut col_in);
        for r in 0..n_fft_h {
            in_freq[r * n_fft_w + c] = col_in[r];
        }
    }
    // inverse row
    for r in 0..n_fft_h {
        let start = r * n_fft_w;
        ifft_w.process(&mut in_freq[start..start + n_fft_w]);
    }

    // extract real part and trim
    let mut out = Array2::<f32>::zeros((out_h, out_w));
    for y in 0..out_h {
        for x in 0..out_w {
            let idx = y * n_fft_w + x;
            out[[y, x]] = in_freq[idx].re / (n_fft_h as f32 * n_fft_w as f32);
        }
    }
    out
}
