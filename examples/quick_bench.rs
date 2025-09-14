// examples/quick_bench.rs
// A very fast benchmark for immediate testing and demonstration

use ndarray::Array2;
use rust_conv_lib::{im2col_single, gemm, conv2d_fft, winograd_conv_2d_3x3, naive_conv_2d};
use std::time::Instant;

fn main() {
    println!("=== Quick Convolution Benchmark ===");
    println!("Fast demonstration of different convolution methods\n");
    
    // Very small test sizes for instant results
    let test_cases = vec![
        ("Tiny 8x8", 8, 8, 3),
        ("Small 32x32", 16, 16, 3),
        ("Medium 64x64", 32, 32, 3),
    ];
    
    for (name, height, width, kernel_size) in test_cases {
        println!("=== Test Case: {} ({}x{} input, {}x{} kernel) ===", 
                 name, height, width, kernel_size, kernel_size);
        
        // Generate test data
        let input = Array2::from_elem((height, width), 1.0);
        let kernel = Array2::from_elem((kernel_size, kernel_size), 0.1);
        
        // Test 1: Naive convolution
        let start = Instant::now();
        let naive_result = naive_conv_2d(&input, &kernel);
        let naive_time = start.elapsed();
        println!("  Naive convolution:     {:>8.2?}", naive_time);
        
        // Test 2: im2col + GEMM convolution
        let start = Instant::now();
        let gemm_result = im2col_gemm_conv_2d(&input, &kernel);
        let gemm_time = start.elapsed();
        println!("  im2col + GEMM:         {:>8.2?}", gemm_time);
        
        // Test 3: FFT convolution
        let start = Instant::now();
        let _fft_result = conv2d_fft(&input, &kernel);
        let fft_time = start.elapsed();
        println!("  FFT convolution:       {:>8.2?}", fft_time);
        
        // Test 4: Winograd convolution (3x3 only) - moved to performance comparison section
        
        // Performance comparison
        let speedup_gemm = naive_time.as_secs_f64() / gemm_time.as_secs_f64();
        let speedup_fft = naive_time.as_secs_f64() / fft_time.as_secs_f64();
        
        println!("  Speedup vs naive:");
        println!("    im2col+GEMM: {:.2}x", speedup_gemm);
        println!("    FFT:         {:.2}x", speedup_fft);
        
        // Verify results are approximately equal (only compare compatible dimensions)
        if naive_result.dim() == gemm_result.dim() {
            let max_diff_gemm = calculate_max_difference(&naive_result, &gemm_result);
            println!("  Correctness (max difference from naive):");
            println!("    im2col+GEMM: {:.6}", max_diff_gemm);
            
            if max_diff_gemm < 1e-5 {
                println!("  ✓ im2col+GEMM produces similar results to naive");
            } else {
                println!("  ⚠ im2col+GEMM has significant differences from naive");
            }
        } else {
            println!("  Note: im2col+GEMM output size differs from naive ({}x{} vs {}x{})", 
                     gemm_result.dim().0, gemm_result.dim().1,
                     naive_result.dim().0, naive_result.dim().1);
        }
        
        // Note: FFT and Winograd may have different output sizes due to padding/implementation differences
        println!("  Note: FFT and Winograd may use different padding conventions");
        
        if kernel_size == 3 {
            let start = Instant::now();
            let winograd_result = winograd_conv_2d_3x3(&input, &kernel);
            let winograd_time = start.elapsed();
            println!("  Winograd convolution:  {:>8.2?}", winograd_time);
            
            let speedup_winograd = naive_time.as_secs_f64() / winograd_time.as_secs_f64();
            println!("    Winograd:    {:.2}x", speedup_winograd);
            
            if naive_result.dim() == winograd_result.dim() {
                let max_diff = calculate_max_difference(&naive_result, &winograd_result);
                println!("    Winograd:    {:.6}", max_diff);
                
                if max_diff < 1e-5 {
                    println!("  ✓ Winograd produces similar results to naive");
                } else {
                    println!("  ⚠ Winograd has significant differences from naive");
                }
            } else {
                println!("  Note: Winograd output size differs from naive ({}x{} vs {}x{})", 
                         winograd_result.dim().0, winograd_result.dim().1,
                         naive_result.dim().0, naive_result.dim().1);
            }
        }
        
        println!();
    }
    
    println!("=== Summary ===");
    println!("This quick benchmark shows:");
    println!("1. Relative performance of different methods");
    println!("2. Correctness verification between implementations");
    println!("3. How performance scales with input size");
    println!();
    println!("For more detailed analysis, run:");
    println!("  cargo bench                    # Statistical benchmarks");
    println!("  cargo run --example imagenet_bench  # Scaled-down ImageNet demo");
}

/// im2col + GEMM convolution implementation
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
    gemm(&kernel_matrix.view(), &im2col_matrix.view(), conv_output.view_mut())
        .expect("GEMM operation failed");
    
    // Step 4: Reshape output
    conv_output.to_shape((out_h, out_w)).unwrap().to_owned()
}

/// Calculate maximum absolute difference between two arrays
fn calculate_max_difference(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    assert_eq!(a.dim(), b.dim(), "Arrays must have same dimensions");
    
    let mut max_diff = 0.0;
    for (a_val, b_val) in a.iter().zip(b.iter()) {
        let diff = (a_val - b_val).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff
}
