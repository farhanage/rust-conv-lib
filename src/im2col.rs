use ndarray::{ArrayView4, ArrayViewMut2};

pub fn im2col_single(
    input: ArrayView4<f32>, // shape (1, C, H, W)
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    mut out: ArrayViewMut2<f32>,
) {
    // We keep this implementation straightforward and safe. For hot loops you can
    // provide an `unsafe` version that uses raw pointers and nobounds checks.
    let (_n, c, h, w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let out_h = (h + 2 * pad_h - k_h) / stride_h + 1;
    let out_w = (w + 2 * pad_w - k_w) / stride_w + 1;
    let patch_size = k_h * k_w * c;

    assert_eq!(out.dim(), (patch_size, out_h * out_w));

    for oh in 0..out_h {
        for ow in 0..out_w {
            let col = oh * out_w + ow;
            let ih0 = oh * stride_h;
            let iw0 = ow * stride_w;
            let mut dst_row = 0usize;

            for ch in 0..c {
                for kh in 0..k_h {
                    for kw in 0..k_w {
                        let ih = ih0 as isize + kh as isize - pad_h as isize;
                        let iw = iw0 as isize + kw as isize - pad_w as isize;

                        let val = if ih < 0 || iw < 0 || ih >= h as isize || iw >= w as isize {
                            0.0f32
                        } else {
                            input[[0, ch, ih as usize, iw as usize]]
                        };

                        out[[dst_row, col]] = val;

                        dst_row += 1;
                    }
                }
            }
        }
    }
}
