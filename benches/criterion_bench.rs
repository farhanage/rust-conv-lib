// benches/criterion_bench.rs
// Use criterion to compare naive, im2col+GEMM, FFT and Winograd for a set of shapes.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::Array2;
use rust_conv_lib::{im2col_single, gemm, conv2d_fft, winograd_conv_2d_3x3, naive_conv_2d};

/// Test configuration for different convolution scenarios
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ConvConfig {
    name: String,
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    output_channels: usize,
    kernel_size: usize,
}

impl ConvConfig {
    fn new(name: &str, in_ch: usize, h: usize, w: usize, out_ch: usize, k: usize) -> Self {
        Self {
            name: name.to_string(),
            input_channels: in_ch,
            input_height: h,
            input_width: w,
            output_channels: out_ch,
            kernel_size: k,
        }
    }

    /// Generate test configurations of various sizes
    fn test_configs() -> Vec<ConvConfig> {
        vec![
            // Small inputs (for testing all methods)
            ConvConfig::new("Small 32x32", 1, 32, 32, 1, 3),
            
            // Medium inputs
            ConvConfig::new("Medium 128x128", 1, 128, 128, 1, 3),
            
            // Larger inputs (typical in modern CNNs)
            ConvConfig::new("Large 256x256", 1, 256, 256, 1, 3),
            
            // Different kernel sizes
            ConvConfig::new("3x3 kernel", 3, 256, 256, 64, 3),
            ConvConfig::new("5x5 kernel", 3, 256, 256, 64, 5),
            ConvConfig::new("7x7 kernel", 3, 256, 256, 64, 7),
        ]
    }
}

/// im2col + GEMM convolution implementation (single channel)
fn im2col_gemm_conv_2d(input: &Array2<f32>, kernel: &Array2<f32>) -> Array2<f32> {
    let (in_h, in_w) = input.dim();
    let (k_h, k_w) = kernel.dim();
    let out_h = in_h - k_h + 1;
    let out_w = in_w - k_w + 1;
    
    // Convert to 4D for im2col_single
    let input_clone = input.clone();
    let input_4d = input_clone.to_shape((1, 1, in_h, in_w)).unwrap();
    
    // Step 1: im2col transformation
    let patch_size = k_h * k_w;
    let mut im2col_matrix = Array2::<f32>::zeros((patch_size, out_h * out_w));
    im2col_single(
        input_4d.view(),
        k_h, k_w,
        0, 0,    // no padding
        1, 1,    // stride = 1
        im2col_matrix.view_mut()
    );
    
    // Step 2: Reshape kernel for matrix multiplication
    let kernel_clone = kernel.clone();
    let kernel_matrix = kernel_clone.to_shape((1, patch_size)).unwrap();
    
    // Step 3: Matrix multiplication (GEMM)
    let mut conv_output = Array2::<f32>::zeros((1, out_h * out_w));
    gemm(&kernel_matrix.view(), &im2col_matrix.view(), conv_output.view_mut()).unwrap();
    
    // Step 4: Reshape output
    conv_output.to_shape((out_h, out_w)).unwrap().to_owned()
}

/// Generate random test data for a given configuration
fn generate_test_data(config: &ConvConfig) -> (Array2<f32>, Array2<f32>) {
    // Generate simple test data (in practice, you'd use proper random data)
    let input = Array2::from_elem(
        (config.input_height, config.input_width),
        0.5
    );
    
    let kernel = Array2::from_elem(
        (config.kernel_size, config.kernel_size),
        0.1
    );
    
    (input, kernel)
}

/// Benchmark naive convolution
fn bench_naive_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("Naive Convolution");
    
    // Configure for slower benchmarks
    group.sample_size(10)
         .measurement_time(std::time::Duration::from_secs(10))
         .warm_up_time(std::time::Duration::from_secs(2));
    
    for config in ConvConfig::test_configs().iter().take(3) { // Reduced to smaller configs for naive
        let (input, kernel) = generate_test_data(config);
        
        group.bench_with_input(
            BenchmarkId::new("naive", &config.name),
            &(input, kernel),
            |b, (input, kernel)| {
                b.iter(|| {
                    black_box(naive_conv_2d(black_box(input), black_box(kernel)))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark im2col + GEMM convolution
fn bench_im2col_gemm_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("im2col + GEMM Convolution");
    
    // Configure for medium-speed benchmarks
    group.sample_size(20)
         .measurement_time(std::time::Duration::from_secs(8))
         .warm_up_time(std::time::Duration::from_secs(2));
    
    for config in ConvConfig::test_configs() {
        let (input, kernel) = generate_test_data(&config);
        
        group.bench_with_input(
            BenchmarkId::new("im2col_gemm", &config.name),
            &(input, kernel),
            |b, (input, kernel)| {
                b.iter(|| {
                    black_box(im2col_gemm_conv_2d(black_box(input), black_box(kernel)))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark FFT convolution
fn bench_fft_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Convolution");
    
    // Configure for medium-speed benchmarks
    group.sample_size(20)
         .measurement_time(std::time::Duration::from_secs(8))
         .warm_up_time(std::time::Duration::from_secs(2));
    
    for config in ConvConfig::test_configs() {
        let (input, kernel) = generate_test_data(&config);
        
        group.bench_with_input(
            BenchmarkId::new("fft", &config.name),
            &(input, kernel),
            |b, (input, kernel)| {
                b.iter(|| {
                    black_box(conv2d_fft(black_box(input), black_box(kernel)))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark Winograd convolution (3x3 kernels only)
fn bench_winograd_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("Winograd Convolution");
    
    // Configure for medium-speed benchmarks
    group.sample_size(20)
         .measurement_time(std::time::Duration::from_secs(8))
         .warm_up_time(std::time::Duration::from_secs(2));
    
    // Only test configs with 3x3 kernels
    for config in ConvConfig::test_configs().iter().filter(|c| c.kernel_size == 3) {
        let (input, kernel) = generate_test_data(config);
        
        group.bench_with_input(
            BenchmarkId::new("winograd", &config.name),
            &(input, kernel),
            |b, (input, kernel)| {
                b.iter(|| {
                    black_box(winograd_conv_2d_3x3(black_box(input), black_box(kernel)))
                })
            },
        );
    }
    
    group.finish();
}

/// Comparison benchmark across all methods for 3x3 kernels
fn bench_comparison_3x3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Method Comparison (3x3)");
    
    // Configure for comparison benchmarks
    group.sample_size(30)
         .measurement_time(std::time::Duration::from_secs(10))
         .warm_up_time(std::time::Duration::from_secs(2));
    
    // Test on a medium-sized input with 3x3 kernel
    let config = ConvConfig::new("Comparison 64x64", 1, 64, 64, 1, 3);
    let (input, kernel) = generate_test_data(&config);
    
    group.bench_function("naive", |b| {
        b.iter(|| naive_conv_2d(black_box(&input), black_box(&kernel)))
    });
    
    group.bench_function("im2col_gemm", |b| {
        b.iter(|| im2col_gemm_conv_2d(black_box(&input), black_box(&kernel)))
    });
    
    group.bench_function("fft", |b| {
        b.iter(|| conv2d_fft(black_box(&input), black_box(&kernel)))
    });
    
    group.bench_function("winograd", |b| {
        b.iter(|| winograd_conv_2d_3x3(black_box(&input), black_box(&kernel)))
    });
    
    group.finish();
}

/// Scaling benchmark - test how methods scale with input size
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Input Size Scaling");
    
    // Configure for scaling benchmarks
    group.sample_size(15)
         .measurement_time(std::time::Duration::from_secs(12))
         .warm_up_time(std::time::Duration::from_secs(2));
    
    let sizes = vec![16, 32, 64, 128, 256];
    
    for size in sizes {
        let config = ConvConfig::new(&format!("{}x{}", size, size), 1, size, size, 1, 3);
        let (input, kernel) = generate_test_data(&config);
        
        // Only test faster methods for larger sizes
        if size <= 64 {
            group.bench_with_input(
                BenchmarkId::new("naive", size),
                &(input.clone(), kernel.clone()),
                |b, (input, kernel)| {
                    b.iter(|| naive_conv_2d(black_box(input), black_box(kernel)))
                },
            );
        }
        
        group.bench_with_input(
            BenchmarkId::new("im2col_gemm", size),
            &(input.clone(), kernel.clone()),
            |b, (input, kernel)| {
                b.iter(|| im2col_gemm_conv_2d(black_box(input), black_box(kernel)))
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("fft", size),
            &(input.clone(), kernel.clone()),
            |b, (input, kernel)| {
                b.iter(|| conv2d_fft(black_box(input), black_box(kernel)))
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("winograd", size),
            &(input, kernel),
            |b, (input, kernel)| {
                b.iter(|| winograd_conv_2d_3x3(black_box(input), black_box(kernel)))
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(10))
        .warm_up_time(std::time::Duration::from_secs(3));
    targets = bench_naive_conv,
    bench_im2col_gemm_conv,
    bench_fft_conv,
    bench_winograd_conv,
    bench_comparison_3x3,
    bench_scaling
);
criterion_main!(benches);
