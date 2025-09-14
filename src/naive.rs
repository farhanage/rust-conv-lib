//! Naive convolution implementations for 2D and multi-channel convolutions.
//!
//! This module provides direct convolution implementations using nested loops.
//! While not optimized for performance, these implementations serve as:
//! - Reference implementations for correctness verification
//! - Baseline for performance comparisons
//! - Educational examples of direct convolution computation

use ndarray::{Array2, Array4, ArrayView2, ArrayView4};

/// Naive 2D convolution for single-channel inputs.
/// 
/// Performs direct convolution using nested loops with no stride or padding.
/// 
/// # Arguments
/// * `input` - Input 2D array (H, W)
/// * `kernel` - Convolution kernel (K_H, K_W)
/// 
/// # Returns
/// * Output array with dimensions (H - K_H + 1, W - K_W + 1)
pub fn naive_conv_2d(input: &Array2<f32>, kernel: &Array2<f32>) -> Array2<f32> {
    let (in_h, in_w) = input.dim();
    let (k_h, k_w) = kernel.dim();
    let out_h = in_h - k_h + 1;
    let out_w = in_w - k_w + 1;
    
    let mut output = Array2::zeros((out_h, out_w));
    
    for oh in 0..out_h {
        for ow in 0..out_w {
            let mut sum = 0.0;
            for kh in 0..k_h {
                for kw in 0..k_w {
                    sum += input[[oh + kh, ow + kw]] * kernel[[kh, kw]];
                }
            }
            output[[oh, ow]] = sum;
        }
    }
    
    output
}

/// Naive 2D convolution with stride and padding support.
/// 
/// # Arguments
/// * `input` - Input 2D array (H, W)
/// * `kernel` - Convolution kernel (K_H, K_W)
/// * `stride` - Stride as (stride_h, stride_w)
/// * `padding` - Padding as (pad_h, pad_w)
/// 
/// # Returns
/// * Output array with computed dimensions based on stride and padding
pub fn naive_conv_2d_with_params(
    input: &ArrayView2<f32>,
    kernel: &ArrayView2<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Array2<f32> {
    let (in_h, in_w) = input.dim();
    let (k_h, k_w) = kernel.dim();
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    
    // Calculate output dimensions
    let out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
    
    let mut output = Array2::zeros((out_h, out_w));
    
    for oh in 0..out_h {
        for ow in 0..out_w {
            let mut sum = 0.0;
            
            for kh in 0..k_h {
                for kw in 0..k_w {
                    let ih = oh * stride_h + kh;
                    let iw = ow * stride_w + kw;
                    
                    // Check bounds with padding
                    if ih >= pad_h && ih < in_h + pad_h && iw >= pad_w && iw < in_w + pad_w {
                        let input_h = ih - pad_h;
                        let input_w = iw - pad_w;
                        
                        if input_h < in_h && input_w < in_w {
                            sum += input[[input_h, input_w]] * kernel[[kh, kw]];
                        }
                    }
                }
            }
            
            output[[oh, ow]] = sum;
        }
    }
    
    output
}

/// Naive 4D convolution for batched multi-channel inputs.
/// 
/// Performs direct convolution on 4D tensors with batch and channel dimensions.
/// Format: NCHW (batch, channels, height, width)
/// 
/// # Arguments
/// * `input` - Input 4D array (N, C_in, H, W)
/// * `kernel` - Convolution kernel (C_out, C_in, K_H, K_W)
/// * `stride` - Stride as (stride_h, stride_w)
/// * `padding` - Padding as (pad_h, pad_w)
/// 
/// # Returns
/// * Output array (N, C_out, H_out, W_out)
pub fn naive_conv_4d(
    input: &ArrayView4<f32>,
    kernel: &ArrayView4<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Array4<f32> {
    let (batch, in_ch, in_h, in_w) = input.dim();
    let (out_ch, _, k_h, k_w) = kernel.dim();
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    
    // Calculate output dimensions
    let out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
    
    let mut output = Array4::zeros((batch, out_ch, out_h, out_w));
    
    for b in 0..batch {
        for oc in 0..out_ch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0;
                    
                    for ic in 0..in_ch {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;
                                
                                // Check bounds with padding
                                if ih >= pad_h && ih < in_h + pad_h && iw >= pad_w && iw < in_w + pad_w {
                                    let input_h = ih - pad_h;
                                    let input_w = iw - pad_w;
                                    
                                    if input_h < in_h && input_w < in_w {
                                        sum += input[[b, ic, input_h, input_w]] * kernel[[oc, ic, kh, kw]];
                                    }
                                }
                            }
                        }
                    }
                    
                    output[[b, oc, oh, ow]] = sum;
                }
            }
        }
    }
    
    output
}

/// Struct for naive convolution implementation that can be used with trait objects
pub struct NaiveConv;

impl NaiveConv {
    /// Create a new naive convolution implementation
    pub fn new() -> Self {
        Self
    }
    
    /// Perform 2D convolution
    pub fn conv_2d(&self, input: &Array2<f32>, kernel: &Array2<f32>) -> Array2<f32> {
        naive_conv_2d(input, kernel)
    }
    
    /// Perform 2D convolution with stride and padding
    pub fn conv_2d_with_params(
        &self,
        input: &ArrayView2<f32>,
        kernel: &ArrayView2<f32>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Array2<f32> {
        naive_conv_2d_with_params(input, kernel, stride, padding)
    }
    
    /// Perform 4D convolution
    pub fn conv_4d(
        &self,
        input: &ArrayView4<f32>,
        kernel: &ArrayView4<f32>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Array4<f32> {
        naive_conv_4d(input, kernel, stride, padding)
    }
}

impl Default for NaiveConv {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_naive_conv_2d_basic() {
        let input = Array::from_shape_vec((3, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        
        let kernel = Array::from_shape_vec((2, 2), vec![
            1.0, 0.0,
            0.0, 1.0,
        ]).unwrap();
        
        let result = naive_conv_2d(&input, &kernel);
        
        assert_eq!(result.dim(), (2, 2));
        assert_eq!(result[[0, 0]], 6.0);  // 1*1 + 2*0 + 4*0 + 5*1 = 6
        assert_eq!(result[[0, 1]], 8.0);  // 2*1 + 3*0 + 5*0 + 6*1 = 8
        assert_eq!(result[[1, 0]], 12.0); // 4*1 + 5*0 + 7*0 + 8*1 = 12
        assert_eq!(result[[1, 1]], 14.0); // 5*1 + 6*0 + 8*0 + 9*1 = 14
    }
    
    #[test]
    fn test_naive_conv_2d_with_stride() {
        let input = Array::from_shape_vec((4, 4), (1..=16).map(|x| x as f32).collect()).unwrap();
        let kernel = Array::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        
        let result = naive_conv_2d_with_params(
            &input.view(),
            &kernel.view(),
            (2, 2), // stride
            (0, 0), // padding
        );
        
        assert_eq!(result.dim(), (2, 2));
        // First output: sum of [1,2,5,6] = 14
        assert_eq!(result[[0, 0]], 14.0);
        // Second output: sum of [3,4,7,8] = 22
        assert_eq!(result[[0, 1]], 22.0);
    }
    
    #[test]
    fn test_naive_conv_4d_basic() {
        // Single batch, single input/output channel
        let input = Array4::from_shape_vec((1, 1, 3, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        
        let kernel = Array4::from_shape_vec((1, 1, 2, 2), vec![
            1.0, 0.0,
            0.0, 1.0,
        ]).unwrap();
        
        let result = naive_conv_4d(&input.view(), &kernel.view(), (1, 1), (0, 0));
        
        assert_eq!(result.dim(), (1, 1, 2, 2));
        assert_eq!(result[[0, 0, 0, 0]], 6.0);
        assert_eq!(result[[0, 0, 0, 1]], 8.0);
        assert_eq!(result[[0, 0, 1, 0]], 12.0);
        assert_eq!(result[[0, 0, 1, 1]], 14.0);
    }
    
    #[test]
    fn test_naive_conv_multi_channel() {
        // 1 batch, 2 input channels, 1 output channel
        let input = Array4::zeros((1, 2, 3, 3));
        let kernel = Array4::ones((1, 2, 2, 2));
        
        let result = naive_conv_4d(&input.view(), &kernel.view(), (1, 1), (0, 0));
        
        assert_eq!(result.dim(), (1, 1, 2, 2));
        // All zeros input should produce zeros output
        assert!(result.iter().all(|&x| x == 0.0));
    }
    
    #[test]
    fn test_naive_conv_struct() {
        let conv = NaiveConv::new();
        
        let input = Array::from_shape_vec((3, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        
        let kernel = Array::from_shape_vec((2, 2), vec![
            1.0, 0.0,
            0.0, 1.0,
        ]).unwrap();
        
        let result = conv.conv_2d(&input, &kernel);
        
        assert_eq!(result.dim(), (2, 2));
        assert_eq!(result[[0, 0]], 6.0);
    }
}
