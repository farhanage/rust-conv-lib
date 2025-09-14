// examples/run_benchmarks.rs
// Example showing how to run the different benchmarking tools

use std::process::Command;

fn main() {
    println!("=== Rust Convolution Library Benchmarking Guide ===\n");
    
    println!("This library provides three main benchmarking tools:\n");
    
    println!("1. Quick Benchmark (examples/quick_bench.rs)");
    println!("   - Very fast demonstration (runs in seconds)");
    println!("   - Tests small input sizes (8x8, 16x16, 32x32)");
    println!("   - Shows relative performance and correctness");
    println!("   - Run with: cargo run --example quick_bench\n");
    
    println!("2. ImageNet Benchmark (examples/imagenet_bench.rs)");
    println!("   - Demonstrates benchmarking on CNN-scale inputs");
    println!("   - Shows typical CNN layer configurations (ResNet, VGG)");
    println!("   - Provides performance analysis framework (scaled down for speed)");
    println!("   - Run with: cargo run --example imagenet_bench\n");
    
    println!("3. Criterion Benchmark (benches/criterion_bench.rs)");
    println!("   - Statistical benchmarking with criterion.rs");
    println!("   - Compares naive, im2col+GEMM, FFT, and Winograd methods");
    println!("   - Tests scaling behavior across different input sizes");
    println!("   - Generates HTML reports with detailed statistics");
    println!("   - Run with: cargo bench\n");
    
    println!("=== Available Benchmark Commands ===\n");
    
    println!("# Run the fastest benchmark for immediate results:");
    println!("cargo run --example quick_bench\n");
    
    println!("# Run the ImageNet-style benchmark demo:");
    println!("cargo run --example imagenet_bench\n");
    
    println!("# Run all Criterion benchmarks:");
    println!("cargo bench\n");
    
    println!("# Run specific benchmark groups:");
    println!("cargo bench \"Method Comparison\"");
    println!("cargo bench \"Input Size Scaling\"");
    println!("cargo bench \"Naive Convolution\"\n");
    
    println!("# Generate benchmark reports:");
    println!("cargo bench --bench criterion_bench");
    println!("# Reports will be generated in target/criterion/\n");
    
    println!("=== Understanding the Results ===\n");
    
    println!("Criterion benchmarks will show:");
    println!("- Time: Average execution time per iteration");
    println!("- Throughput: Operations per second");
    println!("- Statistical confidence intervals");
    println!("- Performance regression detection");
    println!("- HTML reports with detailed visualizations\n");
    
    println!("ImageNet benchmark will show:");
    println!("- GFLOPS (Giga Floating Point Operations Per Second)");
    println!("- Memory usage estimates");
    println!("- Comparison across different layer configurations");
    println!("- Scalability analysis\n");
    
    println!("=== Performance Expectations ===\n");
    
    println!("Expected performance characteristics:");
    println!("- Naive: Slowest, but most memory efficient for small inputs");
    println!("- im2col+GEMM: Fast for medium to large inputs, good parallelization");
    println!("- FFT: Best for large kernels and inputs, high memory overhead");
    println!("- Winograd: Fastest for 3x3 kernels on small to medium inputs\n");
    
    println!("=== Running Benchmarks Now ===\n");
    
    // Try to run the quick benchmark as a demonstration
    println!("Running quick benchmark demonstration...");
    match Command::new("cargo")
        .args(&["run", "--example", "quick_bench"])
        .status()
    {
        Ok(status) => {
            if status.success() {
                println!("✓ Quick benchmark completed successfully");
            } else {
                println!("⚠ Quick benchmark completed with warnings");
            }
        }
        Err(e) => {
            println!("⚠ Could not run quick benchmark: {}", e);
            println!("You can run it manually with: cargo run --example quick_bench");
        }
    }
    
    println!("\nFor detailed performance analysis, run:");
    println!("cargo bench");
    println!("\nBenchmark reports will be available in target/criterion/report/index.html");
}
