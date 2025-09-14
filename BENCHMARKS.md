# Rust Convolution Library Benchmarks

This document describes the benchmarking tools available in the rust_conv_lib for comparing different convolution implementations.

## Benchmark Types

### 1. ImageNet Benchmark (`examples/imagenet_bench.rs`)

A comprehensive benchmarking framework that demonstrates how to measure convolution performance on ImageNet-scale inputs.

**Features:**
- Simulates real CNN layer configurations (ResNet, VGG)
- Measures GFLOPS (Giga Floating Point Operations Per Second)
- Estimates memory usage
- Compares multiple convolution methods
- Provides performance analysis framework

**Usage:**
```bash
cargo run --example imagenet_bench
```

**Sample Output:**
```
=== ImageNet-Scale Convolution Benchmark ===
Configuration: ResNet Conv1 (224x224x3 -> 112x112x64)
Method              Time (ms)    GFLOPS      Memory (MB)
Naive Direct        1234.5       0.85        156.2
im2col + GEMM       89.2         11.7        298.4
FFT                 45.6         22.9        512.1
Winograd 3x3        34.2         30.6        201.8
```

### 2. Criterion Benchmark (`benches/criterion_bench.rs`)

Statistical benchmarking using the Criterion.rs framework for precise performance measurement.

**Features:**
- Statistical analysis with confidence intervals
- Regression detection
- HTML reports with visualizations
- Multiple test configurations
- Scaling analysis

**Usage:**
```bash
# Run all benchmarks
cargo bench

# Run specific benchmark groups
cargo bench "Method Comparison"
cargo bench "Input Size Scaling"
cargo bench "Naive Convolution"

# Generate detailed reports
cargo bench --bench criterion_bench
```

**Reports:**
Criterion generates detailed HTML reports in `target/criterion/report/index.html`

## Convolution Methods Compared

### 1. Naive Direct Convolution
- **Best for:** Very small inputs, educational purposes
- **Characteristics:** Simple nested loops, minimal memory overhead
- **Performance:** Slowest but most straightforward

### 2. im2col + GEMM
- **Best for:** Medium to large inputs, when BLAS is available
- **Characteristics:** Memory-intensive, excellent parallelization
- **Performance:** Fast for most practical scenarios

### 3. FFT-based Convolution
- **Best for:** Large kernels and inputs
- **Characteristics:** High memory overhead, complex number arithmetic
- **Performance:** Asymptotically fastest for large operations

### 4. Winograd Convolution
- **Best for:** 3×3 kernels on small to medium inputs
- **Characteristics:** Reduced arithmetic operations, specific kernel sizes
- **Performance:** Often fastest for 3×3 convolutions

## Sample Benchmark Configurations

The benchmarks test various scenarios:

### Small Scale (Testing All Methods)
- 8×8, 16×16, 32×32 inputs
- 3×3 kernels
- Single channel

### Medium Scale (Practical Applications)
- 64×64, 128×128 inputs
- Multiple channels
- 3×3, 5×5 kernels

### Large Scale (Production-like)
- 224×224, 256×256 inputs
- Real CNN layer configurations
- Multiple channels and filters

### Scaling Analysis
- Input sizes: 16, 32, 64, 128, 256
- Fixed 3×3 kernels
- Performance scaling comparison

## Running Your Own Benchmarks

### Quick Start
```bash
# See benchmark guide
cargo run --example run_benchmarks

# Run a quick comparison
cargo bench "Method Comparison"

# Full benchmark suite (takes longer)
cargo bench
```

### Customizing Benchmarks

1. **Modify test configurations** in `benches/criterion_bench.rs`:
```rust
ConvConfig::new("Custom", channels, height, width, out_channels, kernel_size)
```

2. **Add new methods** by implementing the `ConvMethod` trait in `examples/imagenet_bench.rs`

3. **Adjust measurement parameters**:
```rust
group.measurement_time(Duration::from_secs(10));
group.sample_size(100);
```

## Understanding Results

### Criterion Output
```
Method Comparison (3x3)/naive
                        time:   [2.1234 ms 2.1456 ms 2.1678 ms]
                        change: [-2.3456% -1.2345% +0.1234%] (p = 0.23 > 0.05)
                        No change in performance detected.
```

### Performance Metrics
- **Time**: Average execution time per operation
- **GFLOPS**: Computational throughput
- **Memory**: Estimated memory usage
- **Scaling**: How performance changes with input size

### Expected Performance Characteristics

| Method | Small Inputs | Medium Inputs | Large Inputs | Memory |
|--------|-------------|---------------|--------------|---------|
| Naive | Acceptable | Slow | Very Slow | Low |
| im2col+GEMM | Good | Excellent | Excellent | High |
| FFT | Poor | Good | Excellent | Very High |
| Winograd | Excellent* | Good* | Good* | Medium |

*For 3×3 kernels only

## Hardware Considerations

Performance results depend on:
- **CPU**: Core count, cache size, instruction sets
- **Memory**: Bandwidth, latency
- **BLAS Library**: OpenBLAS, Intel MKL, Apple Accelerate
- **Compiler**: Optimization flags, target architecture

## Tips for Accurate Benchmarking

1. **Warm up the system** - Run benchmarks multiple times
2. **Stable environment** - Close other applications
3. **Consistent data** - Use the same input patterns
4. **Statistical significance** - Let Criterion run sufficient samples
5. **Multiple configurations** - Test various input sizes

## Troubleshooting

### Common Issues

1. **BLAS linking errors**: Install OpenBLAS or disable BLAS features
2. **Out of memory**: Reduce input sizes for large benchmarks
3. **Slow compilation**: Use `--release` for performance testing
4. **Inconsistent results**: Check for background processes

### Performance Tips

1. **Release builds**: Always use `cargo bench` or `cargo run --release`
2. **CPU features**: Enable target-specific optimizations
3. **Memory allocation**: Consider pre-allocating large arrays
4. **Parallelization**: Ensure Rayon thread pool is optimized

## Further Reading

- [Criterion.rs Documentation](https://docs.rs/criterion/)
- [Convolution Algorithm Comparison](https://arxiv.org/abs/1603.07285)
- [GEMM Optimization Techniques](https://github.com/flame/blis)
- [Winograd Convolution](https://arxiv.org/abs/1509.09308)
