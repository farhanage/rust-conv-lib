use ndarray::array;
use rust_conv_lib::conv2d_fft;
use std::time::Instant;

fn main() {
    // Example input (3×3 image)
    let input = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];

    // Example kernel (2×2 filter)
    let kernel = array![
        [1.0, 0.0],
        [0.0, -1.0]
    ];

    let start = Instant::now();
    let output = gemm(&input, &kernel);
    let duration = start.elapsed();

    println!("Output:\n{output:?}");
    println!("Execution time: {:?}", duration);
}
