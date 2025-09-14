use rust_conv_lib::conv3d::*;
use ndarray::Array;

fn main() {
    println!("Testing 3D Convolution implementations...");
    
    // Create a simple 3D input: 1 channel, 4x4x4 volume
    let input_data: Vec<f32> = (0..64).map(|x| x as f32).collect();
    let input = Array::from_shape_vec((1, 4, 4, 4), input_data).unwrap();
    
    // Create a simple 3D kernel: 1 output channel, 1 input channel, 2x2x2 kernel
    let kernel_data = vec![1.0; 8]; // All ones
    let kernel = create_3d_kernel(&kernel_data, 1, 1, 2, 2, 2);
    let kernel_shape = (2, 2, 2);
    
    println!("Input shape: {:?}", input.dim());
    println!("Kernel shape: {:?}", kernel.dim());
    
    // Test naive 3D convolution
    println!("\n=== Testing Naive 3D Convolution ===");
    let naive_conv = NaiveConv3D;
    let result_naive = naive_conv.conv3d(&input.view(), &kernel.view(), kernel_shape, (1, 1, 1), (0, 0, 0));
    println!("Naive result shape: {:?}", result_naive.dim());
    println!("Naive result [0,0,0,0]: {}", result_naive[[0, 0, 0, 0]]);
    println!("Naive result [0,2,2,2]: {}", result_naive[[0, 2, 2, 2]]);
    
    // Test vol2col + GEMM 3D convolution
    println!("\n=== Testing Vol2col + GEMM 3D Convolution ===");
    let gemm_conv = Vol2colGemmConv3D;
    let result_gemm = gemm_conv.conv3d(&input.view(), &kernel.view(), kernel_shape, (1, 1, 1), (0, 0, 0));
    println!("GEMM result shape: {:?}", result_gemm.dim());
    println!("GEMM result [0,0,0,0]: {}", result_gemm[[0, 0, 0, 0]]);
    println!("GEMM result [0,2,2,2]: {}", result_gemm[[0, 2, 2, 2]]);
    
    // Compare results
    println!("\n=== Comparing Results ===");
    let tolerance = 1e-4;
    
    // Compare naive vs GEMM
    let mut naive_gemm_match = true;
    for i in 0..result_naive.len() {
        let diff = (result_naive.as_slice().unwrap()[i] - result_gemm.as_slice().unwrap()[i]).abs();
        if diff > tolerance {
            naive_gemm_match = false;
            break;
        }
    }
    println!("Naive vs GEMM match: {}", naive_gemm_match);
    
    // Test with padding
    println!("\n=== Testing with Padding ===");
    let result_padded = naive_conv.conv3d(&input.view(), &kernel.view(), kernel_shape, (1, 1, 1), (1, 1, 1));
    println!("Padded result shape: {:?}", result_padded.dim());
    
    // Test with stride
    println!("\n=== Testing with Stride ===");
    let result_strided = naive_conv.conv3d(&input.view(), &kernel.view(), kernel_shape, (2, 2, 2), (0, 0, 0));
    println!("Strided result shape: {:?}", result_strided.dim());
    
    // Test vol2col function directly
    println!("\n=== Testing vol2col Function ===");
    let col_matrix = vol2col(&input.view(), kernel_shape, (1, 1, 1), (0, 0, 0));
    println!("Vol2col matrix shape: {:?}", col_matrix.dim());
    
    println!("\n3D Convolution tests completed successfully!");
}
