// examples/imagenet_bench.rs
// Sketch of a benchmarking example (not runnable in this repo) that shows how you'd
// measure convs on ImageNet-sized inputs. The user must provide preprocessed images
// and a data loader; we provide the harness.

use ndarray::{Array2, Array4};
use rust_conv_lib::{im2col_single, gemm, conv2d_fft, winograd_conv_2d_3x3};
use std::time::{Duration, Instant};

/// Benchmark configuration for different convolution scenarios
#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub batch_size: usize,
    pub input_channels: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub output_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub name: String,
}

impl BenchConfig {
    /// Common ImageNet configurations (scaled down for fast demonstration)
    pub fn imagenet_configs() -> Vec<BenchConfig> {
        vec![
            // Small-scale versions of ResNet-like layers 
            BenchConfig {
                batch_size: 2,
                input_channels: 3,
                input_height: 64,
                input_width: 64,
                output_channels: 16,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                name: "ResNet-style Conv1 (64x64x3 -> 64x64x16)".to_string(),
            },
            // Small-scale versions of VGG-like layers
            BenchConfig {
                batch_size: 2,
                input_channels: 3,
                input_height: 64,
                input_width: 64,
                output_channels: 16,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                name: "VGG-style Conv1 (64x64x3 -> 64x64x16)".to_string(),
            },
        ]
    }
}

/// Benchmark results for a single test
#[derive(Debug)]
pub struct BenchResult {
    pub method_name: String,
    pub config: BenchConfig,
    pub duration: Duration,
    pub throughput_gflops: f64,
    pub memory_mb: f64,
}

/// Trait for different convolution implementations
pub trait ConvMethod {
    fn name(&self) -> &str;
    fn run_conv(&self, input: &Array4<f32>, kernel: &Array4<f32>) -> Array4<f32>;
}

/// Naive direct convolution implementation
pub struct NaiveConv;

impl ConvMethod for NaiveConv {
    fn name(&self) -> &str {
        "Naive Direct"
    }

    fn run_conv(&self, input: &Array4<f32>, kernel: &Array4<f32>) -> Array4<f32> {
        // Note: This is a simplified implementation
        // In practice, you'd implement proper direct convolution with padding/stride
        let (batch, in_ch, in_h, in_w) = input.dim();
        let (out_ch, _, k_h, k_w) = kernel.dim();
        
        // Simplified output calculation (assuming stride=1, padding=0)
        let out_h = in_h - k_h + 1;
        let out_w = in_w - k_w + 1;
        
        let mut output = Array4::zeros((batch, out_ch, out_h, out_w));
        
        for b in 0..batch {
            for oc in 0..out_ch {
                for ic in 0..in_ch {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    output[[b, oc, oh, ow]] += 
                                        input[[b, ic, oh + kh, ow + kw]] * 
                                        kernel[[oc, ic, kh, kw]];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        output
    }
}

/// im2col + GEMM convolution implementation
pub struct Im2colGemmConv;

impl ConvMethod for Im2colGemmConv {
    fn name(&self) -> &str {
        "im2col + GEMM"
    }

    fn run_conv(&self, input: &Array4<f32>, kernel: &Array4<f32>) -> Array4<f32> {
        let (batch, in_ch, in_h, in_w) = input.dim();
        let (out_ch, _, k_h, k_w) = kernel.dim();
        
        // Calculate output dimensions (assuming padding=1, stride=1 for demo)
        let pad_h = 1;
        let pad_w = 1;
        let stride_h = 1;
        let stride_w = 1;
        
        let out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
        let mut output = Array4::zeros((batch, out_ch, out_h, out_w));
        
        // Process each batch item separately
        for b in 0..batch {
            // Extract single batch input
            let batch_input = input.slice(s![b..b+1, .., .., ..]).to_owned();
            
            // Create im2col matrix: (k_h * k_w * in_ch) x (out_h * out_w)
            let patch_size = k_h * k_w * in_ch;
            let im2col_cols = out_h * out_w;
            let mut im2col_matrix = Array2::zeros((patch_size, im2col_cols));
            
            // Apply im2col transformation
            im2col_single(
                batch_input.view(),
                k_h,
                k_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                im2col_matrix.view_mut(),
            );
            
            // Reshape kernel to (out_ch, k_h * k_w * in_ch)
            let kernel_reshaped = kernel.clone().to_shape((out_ch, patch_size)).unwrap().to_owned();
            
            // Perform GEMM: kernel_reshaped × im2col_matrix = output_matrix
            // Result shape: (out_ch, out_h * out_w)
            let mut output_matrix = Array2::zeros((out_ch, im2col_cols));
            
            // Use the library's gemm function with CUDA/CPU fallback
            match gemm(&kernel_reshaped.view(), &im2col_matrix.view(), output_matrix.view_mut()) {
                Ok(()) => {
                    // GEMM succeeded (either CUDA or CPU fallback)
                },
                Err(e) => {
                    eprintln!("GEMM failed: {}", e);
                    // This shouldn't happen with our fallback, but just in case
                    panic!("GEMM operation failed");
                }
            }
            
            // Reshape output back to (out_ch, out_h, out_w)
            for oc in 0..out_ch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let col = oh * out_w + ow;
                        output[[b, oc, oh, ow]] = output_matrix[[oc, col]];
                    }
                }
            }
        }
        
        output
    }
}

/// FFT-based convolution implementation
pub struct FFTConv;

impl ConvMethod for FFTConv {
    fn name(&self) -> &str {
        "FFT"
    }

    fn run_conv(&self, input: &Array4<f32>, kernel: &Array4<f32>) -> Array4<f32> {
        let (batch, in_ch, in_h, in_w) = input.dim();
        let (out_ch, _, k_h, k_w) = kernel.dim();
        
        // Calculate valid convolution output dimensions (with padding=1, stride=1)
        let pad_h = 1;
        let pad_w = 1;
        let out_h = (in_h + 2 * pad_h - k_h) + 1;
        let out_w = (in_w + 2 * pad_w - k_w) + 1;
        let mut output = Array4::zeros((batch, out_ch, out_h, out_w));
        
        // Process each batch and channel combination
        for b in 0..batch {
            for oc in 0..out_ch {
                let mut channel_sum = Array2::zeros((out_h, out_w));
                
                for ic in 0..in_ch {
                    // Extract 2D slices
                    let input_slice = input.slice(s![b, ic, .., ..]).to_owned();
                    let kernel_slice = kernel.slice(s![oc, ic, .., ..]).to_owned();
                    
                    // Apply manual padding to input slice to match expected behavior
                    let padded_input = {
                        let mut padded = Array2::zeros((in_h + 2 * pad_h, in_w + 2 * pad_w));
                        padded.slice_mut(s![pad_h..pad_h+in_h, pad_w..pad_w+in_w]).assign(&input_slice);
                        padded
                    };
                    
                    // Perform FFT convolution (which gives full convolution)
                    let full_conv_result = conv2d_fft(&padded_input, &kernel_slice);
                    
                    // Extract the valid convolution region from the full convolution result
                    // For a full convolution of size (H + K - 1), the valid region starts at (K-1)
                    let valid_start_h = k_h - 1;
                    let valid_start_w = k_w - 1;
                    let valid_conv = full_conv_result.slice(s![
                        valid_start_h..valid_start_h + out_h,
                        valid_start_w..valid_start_w + out_w
                    ]).to_owned();
                    
                    // Accumulate across input channels
                    channel_sum = channel_sum + valid_conv;
                }
                
                // Assign the accumulated result to the output
                output.slice_mut(s![b, oc, .., ..]).assign(&channel_sum);
            }
        }
        
        output
    }
}

/// Winograd convolution implementation (for 3x3 kernels)
pub struct WinogradConv;

impl ConvMethod for WinogradConv {
    fn name(&self) -> &str {
        "Winograd 3x3"
    }

    fn run_conv(&self, input: &Array4<f32>, kernel: &Array4<f32>) -> Array4<f32> {
        let (batch, in_ch, in_h, in_w) = input.dim();
        let (out_ch, _, k_h, k_w) = kernel.dim();
        
        // Only works for 3x3 kernels
        if k_h != 3 || k_w != 3 {
            panic!("Winograd implementation only supports 3x3 kernels");
        }
        
        let out_h = in_h - k_h + 1;
        let out_w = in_w - k_w + 1;
        let mut output = Array4::zeros((batch, out_ch, out_h, out_w));
        
        // Process each channel pair separately
        for b in 0..batch {
            for oc in 0..out_ch {
                for ic in 0..in_ch {
                    let input_slice = input.slice(s![b, ic, .., ..]).to_owned();
                    let kernel_slice = kernel.slice(s![oc, ic, .., ..]).to_owned();
                    
                    // This is where you'd call your Winograd convolution
                    let conv_result = winograd_conv_2d_3x3(&input_slice, &kernel_slice);
                    output.slice_mut(s![b, oc, .., ..]).assign(&conv_result);
                }
            }
        }
        
        output
    }
}

/// Calculate GFLOPS for convolution operation
fn calculate_gflops(config: &BenchConfig, duration: Duration) -> f64 {
    let ops = config.batch_size as f64 * 
              config.output_channels as f64 * 
              config.input_channels as f64 * 
              config.kernel_size as f64 * 
              config.kernel_size as f64 * 
              ((config.input_height - config.kernel_size + 2 * config.padding) / config.stride + 1) as f64 * 
              ((config.input_width - config.kernel_size + 2 * config.padding) / config.stride + 1) as f64 * 
              2.0; // multiply-add
    
    ops / duration.as_secs_f64() / 1e9
}

/// Estimate memory usage in MB
fn estimate_memory_mb(config: &BenchConfig) -> f64 {
    let input_size = config.batch_size * config.input_channels * config.input_height * config.input_width;
    let kernel_size = config.output_channels * config.input_channels * config.kernel_size * config.kernel_size;
    let output_h = (config.input_height - config.kernel_size + 2 * config.padding) / config.stride + 1;
    let output_w = (config.input_width - config.kernel_size + 2 * config.padding) / config.stride + 1;
    let output_size = config.batch_size * config.output_channels * output_h * output_w;
    
    (input_size + kernel_size + output_size) as f64 * 4.0 / (1024.0 * 1024.0) // 4 bytes per f32
}

/// Run benchmark for a single configuration and method
fn benchmark_method(
    method: &dyn ConvMethod,
    config: &BenchConfig,
    warmup_runs: usize,
    bench_runs: usize,
) -> BenchResult {
    println!("Benchmarking {} on {}", method.name(), config.name);
    
    // Generate random input and kernel (using simple values for speed)
    let input = Array4::<f32>::from_elem(
        (config.batch_size, config.input_channels, config.input_height, config.input_width),
        0.5, // Simple constant value for faster initialization
    );
    
    let kernel = Array4::<f32>::from_elem(
        (config.output_channels, config.input_channels, config.kernel_size, config.kernel_size),
        0.1, // Simple constant value for faster initialization
    );
    
    // Quick warmup runs
    for _ in 0..warmup_runs {
        let _ = method.run_conv(&input, &kernel);
    }
    
    // Benchmark runs
    let start = Instant::now();
    for _ in 0..bench_runs {
        let _ = method.run_conv(&input, &kernel);
    }
    let total_duration = start.elapsed();
    
    let avg_duration = total_duration / bench_runs as u32;
    let throughput = calculate_gflops(config, avg_duration);
    let memory = estimate_memory_mb(config);
    
    BenchResult {
        method_name: method.name().to_string(),
        config: config.clone(),
        duration: avg_duration,
        throughput_gflops: throughput,
        memory_mb: memory,
    }
}

/// Print benchmark results in a nice table format
fn print_results(results: &[BenchResult]) {
    println!("\n=== Benchmark Results ===");
    println!("{:<20} {:<50} {:<12} {:<12} {:<10}", 
             "Method", "Configuration", "Time (ms)", "GFLOPS", "Memory (MB)");
    println!("{}", "-".repeat(110));
    
    for result in results {
        println!("{:<20} {:<50} {:<12.2} {:<12.2} {:<10.1}", 
                 result.method_name,
                 result.config.name,
                 result.duration.as_millis(),
                 result.throughput_gflops,
                 result.memory_mb);
    }
}

fn main() {
    println!("=== ImageNet-Scale Convolution Benchmark (Fast Demo) ===");
    println!("This is a demonstration of how to benchmark different convolution methods");
    println!("on CNN-like inputs. Configurations are scaled down for fast execution.");
    println!("In a real scenario, you would:");
    println!("1. Load actual ImageNet preprocessed data");
    println!("2. Use real trained model weights");
    println!("3. Implement complete convolution functions");
    println!("4. Add proper error handling and validation");
    println!();
    
    let configs = BenchConfig::imagenet_configs();
    let methods: Vec<Box<dyn ConvMethod>> = vec![
        Box::new(NaiveConv),
        Box::new(Im2colGemmConv),
        Box::new(FFTConv),
    ];
    
    let mut all_results = Vec::new();
    
    // Use fewer iterations for faster demo
    let warmup_runs = 1;  // Reduced from 2
    let bench_runs = 3;   // Reduced from 5
    
    for config in &configs {
        println!("\n--- Configuration: {} ---", config.name);
        
        for method in &methods {
            // Skip Winograd for non-3x3 kernels
            if method.name() == "Winograd 3x3" && config.kernel_size != 3 {
                continue;
            }
            
            let result = benchmark_method(method.as_ref(), config, warmup_runs, bench_runs);
            all_results.push(result);
        }
        
        // Add Winograd for 3x3 kernels
        if config.kernel_size == 3 {
            let winograd = WinogradConv;
            let result = benchmark_method(&winograd, config, warmup_runs, bench_runs);
            all_results.push(result);
        }
    }
    
    print_results(&all_results);
    
    println!("\n=== Performance Analysis (Demo Results) ===");
    println!("Note: These are placeholder implementations for demonstration.");
    println!("Actual performance will depend on:");
    println!("- Real implementation optimizations");
    println!("- Hardware specifications (CPU, memory, BLAS library)");
    println!("- Input data characteristics");
    println!("- Compiler optimizations");
    println!("\nFor production benchmarking:");
    println!("- Scale up to real ImageNet sizes (224x224, etc.)");
    println!("- Use optimized BLAS libraries (Intel MKL, OpenBLAS)");
    println!("- Test on your target hardware");
    println!("- Validate correctness of results");
    println!("- Use statistical analysis (try: cargo bench)");
}

// Note: You would need to add these imports at the top for ndarray slicing
use ndarray::s;
