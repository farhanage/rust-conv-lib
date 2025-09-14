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
use ndarray::Array2;

// Create 2D input and kernel
let input = Array2::from_elem((32, 32), 1.0);
let kernel = Array2::from_elem((3, 3), 0.1);

// Naive direct convolution
let result_naive = naive_conv_2d(&input, &kernel);

// FFT-based convolution
let result_fft = conv2d_fft(&input, &kernel);

// Winograd convolution (3x3 kernels only)
let result_winograd = winograd_conv_2d_3x3(&input, &kernel);

// Using the NaiveConv struct
let conv = NaiveConv::new();
let result_struct = conv.conv_2d(&input, &kernel);

// For im2col + GEMM convolution, use the functions directly:
// You need to use im2col_single and gemm functions manually
// (see examples/quick_bench.rs for complete implementation)
```

### 3D Convolution Example

```rust
use rust_conv_lib::conv3d::*;
use ndarray::Array;

// Create 3D input (1 channel, 16x16x16 volume)
let input = Array::from_shape_vec((1, 16, 16, 16), vec![1.0; 4096]).unwrap();

// Create 3D kernel (1 output channel, 1 input channel, 3x3x3 kernel)
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
| 32x32      | 7.965 $\mu\text{s}$ | 23.038 $\mu\text{s}$ | 42.131 $\mu\text{s}$ | 9.1619 $\mu\text{s}$ |
| 128x128    | 137.19 $\mu\text{s}$ | 412.02 $\mu\text{s}$ | 964.61 $\mu\text{s}$ | 167.87 $\mu\text{s}$ |
| 256x256    | 553.92 $\mu\text{s}$ | 1.7373 ms | 9.54ms | 681.25 $\mu\text{s}$ |

### 2D Convolution on Different Kernel Sizes (256x256 input)

| Kernel Size | Im2col+GEMM | FFT |
|------------|-------------|-----|
| 3x3      | 1.7599 ms | 9.4502 ms |
| 5x5    | 4.2700 ms | 9.4730 ms |  
| 7x7    | 8.8399 ms | 9.4827 ms | 

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
cargo run --example quick_bench     # Quick performance comparison
cargo run --example naive_demo      # Naive convolution examples
cargo run --example imagenet_bench  # Scaled ImageNet benchmarks
cargo run --example conv3d_benchmark # 3D convolution benchmarks

# Run with CUDA support (requires NVIDIA GPU and CUDA toolkit)
cargo build --release --features cuda
```

## Requirements

- Rust 1.70+
- For CUDA support: NVIDIA GPU, CUDA Toolkit 11.0+
- OpenBLAS (automatically installed via openblas-src)

## Architecture

The library is designed with modularity in mind:

- `naive.rs`: Direct/naive convolution implementations for reference and correctness
- `im2col.rs`: 2D image-to-column transformations
- `gemm.rs`: Matrix multiplication with CUDA/CPU backends
- `fft_conv.rs`: FFT-based convolution using rustfft
- `winograd.rs`: Winograd algorithm for 3x3 kernels
- `conv3d.rs`: 3D convolution algorithms and vol2col transformation

The CUDA backend automatically falls back to CPU implementation when GPU operations fail, ensuring reliability across different hardware configurations.

