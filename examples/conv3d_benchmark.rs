use rust_conv_lib::conv3d::*;
use ndarray::Array;
use std::time::Instant;

fn main() {
    println!("3D Convolution Performance Benchmark");
    println!("====================================");

    // Test parameters
    let input_sizes = vec![(1, 16, 16, 16), (1, 32, 32, 32), (2, 64, 64, 64)];
    let kernel_size = (3, 3, 3);
    
    for (channels, depth, height, width) in input_sizes {
        println!("\nInput size: {}x{}x{}x{}", channels, depth, height, width);
        
        // Create test data
        let input_size = channels * depth * height * width;
        let input_data: Vec<f32> = (0..input_size).map(|x| (x as f32) / input_size as f32).collect();
        let input = Array::from_shape_vec((channels, depth, height, width), input_data).unwrap();
        
        let kernel_elements = channels * kernel_size.0 * kernel_size.1 * kernel_size.2;
        let kernel_data = vec![1.0 / kernel_elements as f32; kernel_elements];
        let kernel = create_3d_kernel(&kernel_data, 1, channels, kernel_size.0, kernel_size.1, kernel_size.2);
        
        // Benchmark Naive implementation
        let start = Instant::now();
        let result_naive = naive_conv3d(&input.view(), &kernel.view(), kernel_size, (1, 1, 1), (0, 0, 0));
        let time_naive = start.elapsed();
        
        // Benchmark GEMM implementation
        let start = Instant::now();
        let result_gemm = vol2col_gemm_conv3d(&input.view(), &kernel.view(), kernel_size, (1, 1, 1), (0, 0, 0));
        let time_gemm = start.elapsed();
        
        // Verify results match
        let mut results_match = true;
        let tolerance = 1e-5;
        if result_naive.len() == result_gemm.len() {
            for (a, b) in result_naive.iter().zip(result_gemm.iter()) {
                if (a - b).abs() > tolerance {
                    results_match = false;
                    break;
                }
            }
        } else {
            results_match = false;
        }
        
        println!("  Naive convolution:    {:>8.2} ms", time_naive.as_secs_f64() * 1000.0);
        println!("  Vol2col+GEMM:         {:>8.2} ms", time_gemm.as_secs_f64() * 1000.0);
        println!("  Speedup:              {:>8.2}x", time_naive.as_secs_f64() / time_gemm.as_secs_f64());
        println!("  Results match:        {}", results_match);
        println!("  Output shape:         {:?}", result_naive.dim());
    }
    
    println!("\n3D convolution benchmarks completed!");
    println!("\nNote: The vol2col+GEMM implementation benefits from CUDA acceleration");
    println!("when available, and optimized BLAS operations for matrix multiplication.");
}
