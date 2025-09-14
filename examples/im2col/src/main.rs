use ndarray::{array, Array2};
use rust_conv_lib::im2col_single;
use std::time::Instant;

fn main() {
    // Example input (3×3 image) - reshape to 4D: (batch=1, channels=1, height=3, width=3)
    let input_2d = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];
    
    // Reshape to 4D array (1, 1, 3, 3)
    let input_4d = input_2d.to_shape((1, 1, 3, 3)).unwrap();
    
    // Kernel parameters
    let k_h = 2; // kernel height
    let k_w = 2; // kernel width
    let pad_h = 0; // padding height
    let pad_w = 0; // padding width  
    let stride_h = 1; // stride height
    let stride_w = 1; // stride width
    
    // Calculate output dimensions
    let out_h = (3 + 2 * pad_h - k_h) / stride_h + 1; // (3 + 0 - 2) / 1 + 1 = 2
    let out_w = (3 + 2 * pad_w - k_w) / stride_w + 1; // (3 + 0 - 2) / 1 + 1 = 2
    let patch_size = k_h * k_w * 1; // 2 * 2 * 1 = 4 (channels=1)
    
    // Create output matrix
    let mut output = Array2::<f32>::zeros((patch_size, out_h * out_w));
    
    let start = Instant::now();
    im2col_single(
        input_4d.view(),
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        output.view_mut()
    );
    let duration = start.elapsed();

    println!("Input (reshaped to 4D):\n{:?}", input_4d);
    println!("Output (im2col matrix):\n{:?}", output);
    println!("Execution time: {:?}", duration);
}
