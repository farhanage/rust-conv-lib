//! rust_conv_lib
//!
//! Implements multiple convolution strategies: direct, im2col+GEMM, FFT-based,
//! and Winograd for 2D and 3D. Focused on memory reuse, alignment, and
//! multi-threading.

pub mod naive;
pub mod im2col;
pub mod gemm;
pub mod fft_conv;
pub mod winograd;
pub mod conv3d;

pub use naive::*;
pub use im2col::*;
pub use gemm::*;
pub use fft_conv::*;
pub use winograd::*;
pub use conv3d::*;