# Rust Convolution Library

A high-performance Rust library implementing multiple convolution algorithms for 2D and 3D data, optimized for both CPU and GPU execution.

## Features

### 2D Convolution Algorithms
- **Direct/Naive Convolution**: Simple nested loops implementation
- **Im2col + GEMM**: Transforms convolution into matrix multiplication
- **FFT-based Convolution**: Frequency domain convolution using rustfft
- **Winograd Convolution**: Optimized algorithm for 3x3 kernels

### 3D Convolution Algorithms
- **Direct/Naive 3D Convolution**: Simple nested loops for volumetric data
- **Vol2col + GEMM**: 3D extension of im2col for matrix multiplication

### Performance Optimizations
- **CUDA Acceleration**: GPU-accelerated matrix multiplication using cudarc
- **Optimized BLAS**: CPU fallback using OpenBLAS for matrix operations
- **Memory Efficiency**: Minimal data copying and optimal memory layouts

## Usage

### 2D Convolution Example

```rust
use rust_conv_lib::*;
use ndarray::Array;

// Create input and kernel
let input = Array::from_shape_vec((1, 32, 32), vec![1.0; 1024]).unwrap();
let kernel = Array::from_shape_vec((1, 1, 3, 3), vec![1.0; 9]).unwrap();

// Different convolution methods
let conv_naive = NaiveConv;
let result1 = conv_naive.conv(&input.view(), &kernel.view(), (1, 1), (0, 0));

let conv_gemm = Im2colGemmConv;
let result2 = conv_gemm.conv(&input.view(), &kernel.view(), (1, 1), (0, 0));

let conv_fft = FftConv;
let result3 = conv_fft.conv(&input.view(), &kernel.view(), (1, 1), (0, 0));

let conv_winograd = WinogradConv;
let result4 = conv_winograd.conv(&input.view(), &kernel.view(), (1, 1), (0, 0));
```

### 3D Convolution Example

```rust
use rust_conv_lib::conv3d::*;
use ndarray::Array;

// Create 3D input and kernel
let input = Array::from_shape_vec((1, 16, 16, 16), vec![1.0; 4096]).unwrap();
let kernel = create_3d_kernel(&vec![1.0; 27], 1, 1, 3, 3, 3);

// Different 3D convolution methods
let conv_naive = NaiveConv3D;
let result1 = conv_naive.conv3d(&input.view(), &kernel.view(), (3, 3, 3), (1, 1, 1), (0, 0, 0));

let conv_gemm = Vol2colGemmConv3D;
let result2 = conv_gemm.conv3d(&input.view(), &kernel.view(), (3, 3, 3), (1, 1, 1), (0, 0, 0));
```

## Performance Results

### 2D Convolution Benchmarks (3x3 kernel)

| Input Size | Naive | Im2col+GEMM | FFT | Winograd |
|------------|-------|-------------|-----|----------|
| 32x32      | 0.12ms | 0.31ms | 0.89ms | 0.08ms |
| 64x64      | 0.48ms | 1.02ms | 1.47ms | 0.32ms |
| 128x128    | 1.96ms | 3.45ms | 3.21ms | 1.24ms |
| 256x256    | 7.82ms | 12.8ms | 9.54ms | 4.89ms |

### 3D Convolution Benchmarks (3x3x3 kernel)

| Input Size     | Naive | Vol2col+GEMM |
|----------------|-------|--------------|
| 16x16x16       | 30.1ms | 37.4ms |
| 32x32x32       | 299.7ms | 359.7ms |
| 64x64x64 (2ch) | 5255.8ms | 6572.1ms |

**Note**: The vol2col+GEMM implementation includes CUDA acceleration when available.

## Building and Testing

```bash
# Build the library
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo run --example imagenet_bench
cargo run --example conv3d_benchmark

# Run with CUDA support (requires NVIDIA GPU and CUDA toolkit)
cargo build --release --features cuda
```

## Requirements

- Rust 1.70+
- For CUDA support: NVIDIA GPU, CUDA Toolkit 11.0+
- OpenBLAS (automatically installed via openblas-src)

## Architecture

The library is designed with modularity in mind:

- `im2col.rs`: 2D image-to-column transformations
- `gemm.rs`: Matrix multiplication with CUDA/CPU backends
- `fft_conv.rs`: FFT-based convolution using rustfft
- `winograd.rs`: Winograd algorithm for 3x3 kernels
- `conv3d.rs`: 3D convolution algorithms and vol2col transformation

The CUDA backend automatically falls back to CPU implementation when GPU operations fail, ensuring reliability across different hardware configurations.

