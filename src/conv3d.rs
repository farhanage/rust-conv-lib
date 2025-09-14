//! 3D Convolution implementations
//!
//! This module provides various 3D convolution algorithms including:
//! - Direct (naive) 3D convolution
//! - vol2col + GEMM (volume-to-column transformation)
//!
//! All algorithms work with 3D tensors in the format (depth, height, width).

use ndarray::{Array2, Array4, ArrayView2, ArrayView4};
use crate::gemm::gemm;

/// Trait for 3D convolution operations
pub trait Conv3D {
    /// Perform 3D convolution
    /// 
    /// # Arguments
    /// * `input` - Input volume of shape (in_channels, depth, height, width)
    /// * `kernel` - Convolution kernel in flattened format (out_channels, in_channels * kd * kh * kw)
    /// * `kernel_shape` - Original kernel dimensions (kd, kh, kw)
    /// * `stride` - Stride for each dimension (stride_d, stride_h, stride_w)
    /// * `padding` - Padding for each dimension (pad_d, pad_h, pad_w)
    /// 
    /// # Returns
    /// Output volume of shape (out_channels, out_depth, out_height, out_width)
    fn conv3d(
        &self,
        input: &ArrayView4<f32>,  // (in_channels, depth, height, width)
        kernel: &ArrayView2<f32>, // (out_channels, in_channels * kd * kh * kw) - flattened for simplicity
        kernel_shape: (usize, usize, usize), // (kd, kh, kw)
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> Array4<f32>;
}

/// Direct 3D convolution implementation
pub struct NaiveConv3D;

impl Conv3D for NaiveConv3D {
    fn conv3d(
        &self,
        input: &ArrayView4<f32>,
        kernel: &ArrayView2<f32>,
        kernel_shape: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> Array4<f32> {
        naive_conv3d(input, kernel, kernel_shape, stride, padding)
    }
}

/// vol2col + GEMM 3D convolution implementation
pub struct Vol2colGemmConv3D;

impl Conv3D for Vol2colGemmConv3D {
    fn conv3d(
        &self,
        input: &ArrayView4<f32>,
        kernel: &ArrayView2<f32>,
        kernel_shape: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> Array4<f32> {
        vol2col_gemm_conv3d(input, kernel, kernel_shape, stride, padding)
    }
}

/// Direct 3D convolution (naive implementation)
pub fn naive_conv3d(
    input: &ArrayView4<f32>,
    kernel: &ArrayView2<f32>,
    kernel_shape: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
) -> Array4<f32> {
    let (in_channels, in_depth, in_height, in_width) = input.dim();
    let (out_channels, kernel_total) = kernel.dim();
    let (kernel_depth, kernel_height, kernel_width) = kernel_shape;
    
    assert_eq!(kernel_total, in_channels * kernel_depth * kernel_height * kernel_width,
               "Kernel size must match input channels * kernel dimensions");
    
    let (stride_d, stride_h, stride_w) = stride;
    let (pad_d, pad_h, pad_w) = padding;
    
    // Calculate output dimensions
    let out_depth = (in_depth + 2 * pad_d - kernel_depth) / stride_d + 1;
    let out_height = (in_height + 2 * pad_h - kernel_height) / stride_h + 1;
    let out_width = (in_width + 2 * pad_w - kernel_width) / stride_w + 1;
    
    let mut output = Array4::<f32>::zeros((out_channels, out_depth, out_height, out_width));
    
    // Perform convolution
    for out_c in 0..out_channels {
        for out_d in 0..out_depth {
            for out_h in 0..out_height {
                for out_w in 0..out_width {
                    let mut sum = 0.0;
                    
                    for in_c in 0..in_channels {
                        for kd in 0..kernel_depth {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let in_d = out_d * stride_d + kd;
                                    let in_h = out_h * stride_h + kh;
                                    let in_w = out_w * stride_w + kw;
                                    
                                    // Check bounds with padding
                                    if in_d >= pad_d && in_h >= pad_h && in_w >= pad_w {
                                        let actual_d = in_d - pad_d;
                                        let actual_h = in_h - pad_h;
                                        let actual_w = in_w - pad_w;
                                        
                                        if actual_d < in_depth && actual_h < in_height && actual_w < in_width {
                                            let kernel_idx = in_c * kernel_depth * kernel_height * kernel_width +
                                                           kd * kernel_height * kernel_width +
                                                           kh * kernel_width + kw;
                                            sum += input[[in_c, actual_d, actual_h, actual_w]] 
                                                 * kernel[[out_c, kernel_idx]];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    output[[out_c, out_d, out_h, out_w]] = sum;
                }
            }
        }
    }
    
    output
}

/// Transform 3D volume to column matrix (vol2col)
/// Similar to im2col but for 3D volumes
pub fn vol2col(
    input: &ArrayView4<f32>,
    kernel_shape: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
) -> Array2<f32> {
    let (in_channels, in_depth, in_height, in_width) = input.dim();
    let (kernel_depth, kernel_height, kernel_width) = kernel_shape;
    let (stride_d, stride_h, stride_w) = stride;
    let (pad_d, pad_h, pad_w) = padding;
    
    // Calculate output dimensions
    let out_depth = (in_depth + 2 * pad_d - kernel_depth) / stride_d + 1;
    let out_height = (in_height + 2 * pad_h - kernel_height) / stride_h + 1;
    let out_width = (in_width + 2 * pad_w - kernel_width) / stride_w + 1;
    
    let kernel_size_total = in_channels * kernel_depth * kernel_height * kernel_width;
    let patches_count = out_depth * out_height * out_width;
    
    // Column matrix: (in_channels * kernel_volume, num_patches)
    let mut col_matrix = Array2::<f32>::zeros((kernel_size_total, patches_count));
    
    let mut patch_idx = 0;
    for out_d in 0..out_depth {
        for out_h in 0..out_height {
            for out_w in 0..out_width {
                let mut kernel_idx = 0;
                
                for in_c in 0..in_channels {
                    for kd in 0..kernel_depth {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let in_d = out_d * stride_d + kd;
                                let in_h = out_h * stride_h + kh;
                                let in_w = out_w * stride_w + kw;
                                
                                // Handle padding
                                if in_d >= pad_d && in_h >= pad_h && in_w >= pad_w {
                                    let actual_d = in_d - pad_d;
                                    let actual_h = in_h - pad_h;
                                    let actual_w = in_w - pad_w;
                                    
                                    if actual_d < in_depth && actual_h < in_height && actual_w < in_width {
                                        col_matrix[[kernel_idx, patch_idx]] = 
                                            input[[in_c, actual_d, actual_h, actual_w]];
                                    }
                                    // else: padding area - values remain 0
                                }
                                // else: padding area - values remain 0
                                
                                kernel_idx += 1;
                            }
                        }
                    }
                }
                patch_idx += 1;
            }
        }
    }
    
    col_matrix
}

/// 3D convolution using vol2col + GEMM
pub fn vol2col_gemm_conv3d(
    input: &ArrayView4<f32>,
    kernel: &ArrayView2<f32>,
    kernel_shape: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
) -> Array4<f32> {
    let (in_channels, in_depth, in_height, in_width) = input.dim();
    let (out_channels, kernel_total) = kernel.dim();
    let (kernel_depth, kernel_height, kernel_width) = kernel_shape;
    
    assert_eq!(kernel_total, in_channels * kernel_depth * kernel_height * kernel_width,
               "Kernel size must match input channels * kernel dimensions");
    
    let (stride_d, stride_h, stride_w) = stride;
    let (pad_d, pad_h, pad_w) = padding;
    
    // Calculate output dimensions
    let out_depth = (in_depth + 2 * pad_d - kernel_depth) / stride_d + 1;
    let out_height = (in_height + 2 * pad_h - kernel_height) / stride_h + 1;
    let out_width = (in_width + 2 * pad_w - kernel_width) / stride_w + 1;
    
    // Transform input to column matrix
    let col_matrix = vol2col(input, kernel_shape, stride, padding);
    
    let patches_count = out_depth * out_height * out_width;
    
    // Perform GEMM: kernel * col_matrix = output_matrix
    // (out_channels, kernel_total) * (kernel_total, patches_count) = (out_channels, patches_count)
    let mut output_matrix = Array2::<f32>::zeros((out_channels, patches_count));
    gemm(&kernel.view(), &col_matrix.view(), output_matrix.view_mut()).expect("GEMM failed");
    
    // Reshape output back to 4D
    output_matrix.to_shape((out_channels, out_depth, out_height, out_width))
        .expect("Failed to reshape output")
        .to_owned()
}

/// Helper function to create a 3D kernel in the flattened format expected by the functions
pub fn create_3d_kernel(kernel_data: &[f32], out_channels: usize, in_channels: usize, 
                       kernel_depth: usize, kernel_height: usize, kernel_width: usize) -> Array2<f32> {
    let kernel_size = in_channels * kernel_depth * kernel_height * kernel_width;
    assert_eq!(kernel_data.len(), out_channels * kernel_size, 
               "Kernel data length must match out_channels * kernel_size");
    
    Array2::from_shape_vec((out_channels, kernel_size), kernel_data.to_vec())
        .expect("Failed to create kernel")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_vol2col_basic() {
        // Create a simple 3D input: 1 channel, 3x3x3 volume
        let input = Array::from_shape_vec((1, 3, 3, 3), (0..27).map(|x| x as f32).collect()).unwrap();
        
        // Use 2x2x2 kernel
        let kernel_shape = (2, 2, 2);
        let stride = (1, 1, 1);
        let padding = (0, 0, 0);
        
        let col_matrix = vol2col(&input.view(), kernel_shape, stride, padding);
        
        // Output should be (8, 8) - 8 kernel elements, 8 patches
        assert_eq!(col_matrix.dim(), (8, 8));
    }

    #[test]
    fn test_naive_conv3d_basic() {
        // Create simple input and kernel
        let input = Array::from_shape_vec((1, 2, 2, 2), vec![1.0; 8]).unwrap();
        let kernel = create_3d_kernel(&vec![1.0; 8], 1, 1, 2, 2, 2);
        
        let result = naive_conv3d(&input.view(), &kernel.view(), (2, 2, 2), (1, 1, 1), (0, 0, 0));
        
        // Output should be (1, 1, 1, 1) with value 8.0
        assert_eq!(result.dim(), (1, 1, 1, 1));
        assert_eq!(result[[0, 0, 0, 0]], 8.0);
    }

    #[test]
    fn test_vol2col_gemm_conv3d_basic() {
        // Create simple input and kernel
        let input = Array::from_shape_vec((1, 2, 2, 2), vec![1.0; 8]).unwrap();
        let kernel = create_3d_kernel(&vec![1.0; 8], 1, 1, 2, 2, 2);
        
        let result = vol2col_gemm_conv3d(&input.view(), &kernel.view(), (2, 2, 2), (1, 1, 1), (0, 0, 0));
        
        // Output should be (1, 1, 1, 1) with value 8.0
        assert_eq!(result.dim(), (1, 1, 1, 1));
        assert_eq!(result[[0, 0, 0, 0]], 8.0);
    }
}
